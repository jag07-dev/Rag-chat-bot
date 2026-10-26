import io
import os
import re

import faiss
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄", layout="wide")

load_dotenv()

# ---- Settings (chosen by measurement; see README "Evaluation") -------------
CHUNK_SIZE = 60        # words per chunk
CHUNK_OVERLAP = 15     # words shared between neighbouring chunks
TOP_K = 8              # chunks sent to the LLM
POOL_SIZE = 30         # candidates taken from each search before merging
RRF_K = 60             # constant for Reciprocal Rank Fusion
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_setting(name, default=None):
    """Read from the environment (.env locally), then Streamlit secrets (cloud)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return default


GROQ_API_KEY = get_setting("GROQ_API_KEY")
GROQ_MODEL = get_setting("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Add it to the .env file (or to Streamlit secrets when deployed).")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


embedding_model = load_embedding_model()


def extract_pages_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return [
        (number, page.extract_text() or "")
        for number, page in enumerate(reader.pages, start=1)
    ]


def split_pages_into_chunks(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split into overlapping word chunks, remembering which pages each chunk covers."""
    words = []
    for page_number, text in pages:
        words.extend((word, page_number) for word in text.split())

    chunks = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if piece:
            chunks.append({
                "text": " ".join(word for word, _ in piece),
                "pages": sorted({page for _, page in piece}),
            })
    return chunks


def create_chunk_embeddings(texts, model):
    return model.encode_document(texts, convert_to_numpy=True, normalize_embeddings=True)


def create_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


@st.cache_resource(show_spinner="Creating document embeddings...")
def build_document_index(file_bytes, file_name):
    """Build the vector index and keyword index once per uploaded file."""
    pages = extract_pages_from_pdf(io.BytesIO(file_bytes))
    chunks = split_pages_into_chunks(pages)
    if not chunks:
        return None, None, None
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_chunk_embeddings(texts, embedding_model)
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    return create_faiss_index(embeddings), chunks, bm25


def retrieve_relevant_chunks(question, model, index, chunks, bm25, top_k=TOP_K):
    """Hybrid search: vector similarity + BM25 keywords, merged with Reciprocal Rank Fusion."""
    q = model.encode_query(
        question, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32").reshape(1, -1)
    _, indices = index.search(q, min(POOL_SIZE, len(chunks)))
    dense = [int(i) for i in indices[0] if i != -1]

    keyword_scores = bm25.get_scores(tokenize(question))
    ranked_kw = sorted(
        range(len(keyword_scores)), key=lambda i: keyword_scores[i], reverse=True
    )[:POOL_SIZE]
    sparse = [i for i in ranked_kw if keyword_scores[i] > 0]

    fused = {}
    for ranking in (dense, sparse):
        for rank, chunk_index in enumerate(ranking, start=1):
            fused[chunk_index] = fused.get(chunk_index, 0.0) + 1.0 / (RRF_K + rank)

    best = sorted(fused, key=fused.get, reverse=True)[:top_k]
    return [chunks[i] for i in best]


def generate_answer(question, relevant_chunks, client, model_name):
    context = "\n\n".join(chunk["text"] for chunk in relevant_chunks)

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question using only the context
provided from the uploaded document.

If the answer is not available in the context, say:
"I could not find that information in the document."

If the document mentions the topic but does not give
the exact detail asked for, say what the document does
state instead of refusing.

Do not use outside knowledge.
Keep the answer clear and concise.
"""

    user_prompt = f"""
Document context:

{context}

Question:

{question}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


st.title("📄 RAG Document Chatbot")

st.write("Upload a PDF document and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.messages = []
        st.session_state.uploaded_file_name = uploaded_file.name

    st.success(f"Uploaded successfully: {uploaded_file.name}")

    faiss_index, document_chunks, bm25_index = build_document_index(
        uploaded_file.getvalue(), uploaded_file.name
    )

    if faiss_index is None:
        st.error(
            "No readable text was found in this PDF. "
            "It may be scanned or image-based."
        )
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a question about the document")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        relevant_chunks = retrieve_relevant_chunks(
            user_question, embedding_model, faiss_index, document_chunks, bm25_index
        )

        try:
            with st.chat_message("assistant"):
                with st.spinner("Searching the document..."):
                    answer = generate_answer(
                        user_question, relevant_chunks, groq_client, GROQ_MODEL
                    )

                if not answer or not answer.strip():
                    st.error("The model returned an empty response.")
                    st.stop()

                st.markdown(answer)

                best_pages = relevant_chunks[0]["pages"]
                other_pages = sorted(
                    {p for chunk in relevant_chunks[1:] for p in chunk["pages"]}
                    - set(best_pages)
                )
                caption = "Best match: page(s) " + ", ".join(map(str, best_pages))
                if other_pages:
                    caption += " | Also retrieved: " + ", ".join(map(str, other_pages))
                st.caption(caption)

        except Exception as error:
            st.error(
                "Unable to generate an answer. Please check your API key, "
                "model configuration, and internet connection."
            )
            st.caption(str(error))
            st.stop()

        st.session_state.messages.append({"role": "assistant", "content": answer})