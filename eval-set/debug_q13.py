from eval_rag import extract_text_from_pdf, split_text_into_chunks, build_index, search, EMBED_MODEL
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(EMBED_MODEL)
chunks = split_text_into_chunks(extract_text_from_pdf("rag_demo_company_handbook.pdf"))
index = build_index(chunks, model)
for ci, score in search("How many days of parental leave do I get?", model, index, 3):
    print("SCORE:", round(score, 2))
    print("HAS PARENTAL ROW:", "Parental leave" in chunks[ci])
    print(chunks[ci])
    print("-" * 60) 