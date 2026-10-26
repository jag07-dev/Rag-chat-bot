"""
Evaluation harness for the PDF RAG chatbot.

Retrieval metrics need no API key. Answer metrics (--generate) need GROQ_API_KEY.

Examples:
  python eval_rag.py --pdf handbook.pdf --check-evidence      # validate the eval set (no models needed)
  python eval_rag.py --pdf handbook.pdf                       # retrieval baseline (vector search)
  python eval_rag.py --pdf handbook.pdf --hybrid              # vector + keyword (BM25) search
  python eval_rag.py --pdf handbook.pdf --sweep               # compare chunk sizes
  python eval_rag.py --pdf handbook.pdf --generate            # answers, no threshold
  python eval_rag.py --pdf handbook.pdf --generate --chunk-size 60 --overlap 15 --hybrid
"""
import argparse
import json
import os
import re

REFUSAL = "i could not find that information in the document"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

USE_HYBRID = False   # set from --hybrid
USE_RERANK = False   # set from --rerank
BM25 = None          # keyword index, rebuilt whenever the chunks change
CHUNKS = None        # current chunk list, needed by the reranker
RERANKER = None      # cross-encoder, loaded on first use
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 30     # candidates taken from hybrid search before reranking


def norm(text):
    """Lowercase, unify special hyphens and spaces, collapse whitespace."""
    for ch in ("\u2010", "\u2011", "\u2013", "\u2014"):
        text = text.replace(ch, "-")
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).lower()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", norm(text))


# ---- Pipeline functions: copied from your app so results match it ----------
def extract_text_from_pdf(pdf_path):
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    pages = [p.extract_text() for p in reader.pages if p.extract_text()]
    return "\n".join(pages)


def split_text_into_chunks(text, chunk_size=180, chunk_overlap=40):
    words = text.split()
    chunks = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def build_index(chunks, model):
    import faiss
    emb = model.encode_document(chunks, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    return index


def build_bm25(chunks):
    from rank_bm25 import BM25Okapi
    return BM25Okapi([tokenize(c) for c in chunks])


def search(question, model, index, k):
    """Vector search only. Returns [(chunk_index, cosine_similarity), ...]."""
    q = model.encode_query(question, convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q.astype("float32").reshape(1, -1), k)
    return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]


def hybrid_search(question, model, index, bm25, k, pool=30, rrf_k=60):
    """Vector + keyword search merged with Reciprocal Rank Fusion.
    Returns [(chunk_index, fused_score), ...]; fused scores are NOT similarities."""
    dense = [ci for ci, _ in search(question, model, index, pool)]
    kw_scores = bm25.get_scores(tokenize(question))
    ranked_kw = sorted(range(len(kw_scores)), key=lambda i: kw_scores[i], reverse=True)[:pool]
    sparse = [i for i in ranked_kw if kw_scores[i] > 0]   # ignore chunks with no keyword overlap
    fused = {}
    for ranking in (dense, sparse):
        for rank, ci in enumerate(ranking, start=1):
            fused[ci] = fused.get(ci, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]


def rerank(question, ranked, k):
    """Score each (question, chunk) pair with a cross-encoder and keep the best k.
    Returns [(chunk_index, cross_encoder_score), ...]; scores are NOT similarities."""
    global RERANKER
    if RERANKER is None:
        from sentence_transformers import CrossEncoder
        RERANKER = CrossEncoder(RERANK_MODEL)
    pairs = [(question, CHUNKS[ci]) for ci, _ in ranked]
    scores = RERANKER.predict(pairs)
    order = sorted(range(len(ranked)), key=lambda i: scores[i], reverse=True)[:k]
    return [(ranked[i][0], float(scores[i])) for i in order]


def retrieve(question, model, index, k):
    if USE_RERANK:
        # Wider candidate pool from hybrid search, then the cross-encoder picks the best k
        pool = hybrid_search(question, model, index, BM25, RERANK_POOL) if USE_HYBRID \
            else search(question, model, index, RERANK_POOL)
        return rerank(question, pool, k)
    if USE_HYBRID:
        return hybrid_search(question, model, index, BM25, k)
    return search(question, model, index, k)


def generate_answer(question, chunks, client, model_name):
    system_prompt = (
        "You are a document question-answering assistant.\n"
        "Answer the user's question using only the context provided from the uploaded document.\n"
        'If the answer is not available in the context, say:\n'
        '"I could not find that information in the document."\n'
        "If the document mentions the topic but does not give the exact detail asked for, "
        "say what the document does state instead of refusing.\n"
        "Do not use outside knowledge.\nKeep the answer clear and concise."
    )
    user_prompt = f"Document context:\n\n{chr(10).join(chunks)}\n\nQuestion:\n\n{question}"
    r = client.chat.completions.create(
        model=model_name, temperature=0,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
    )
    return r.choices[0].message.content or ""


# ---- Evaluation ------------------------------------------------------------
def load_eval_set(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_evidence(eval_set, chunks):
    """Sanity-check the eval set against the chunks (no embeddings needed)."""
    print(f"{len(chunks)} chunks\n")
    bad = 0
    for item in eval_set:
        if item["type"] == "unanswerable":
            print(f"{item['id']}  unanswerable")
            continue
        homes = []
        for ev in item["evidence"]:
            homes.append([i for i, c in enumerate(chunks) if norm(ev) in norm(c)])
        missing = [item["evidence"][j] for j, h in enumerate(homes) if not h]
        flag = "OK " if not missing else "MISSING"
        if missing:
            bad += 1
        print(f"{item['id']}  {flag}  evidence in chunks {homes}")
    print(f"\n{bad} item(s) with evidence not found in any chunk.")


def evaluate_retrieval(eval_set, model, index, chunks, ks=(1, 3, 5)):
    max_k = max(ks)
    answerable = [i for i in eval_set if i["type"] != "unanswerable"]
    unanswerable = [i for i in eval_set if i["type"] == "unanswerable"]

    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    ans_top1, failures = [], []

    for item in answerable:
        ranked = retrieve(item["question"], model, index, max_k)
        ans_top1.append(ranked[0][1])
        evid = [norm(e) for e in item["evidence"]]

        first_rank = None
        for pos, (ci, _) in enumerate(ranked, start=1):
            if any(e in norm(chunks[ci]) for e in evid):
                first_rank = pos
                break
        rr_sum += 1.0 / first_rank if first_rank else 0.0

        for k in ks:
            top_text = norm(" ".join(chunks[ci] for ci, _ in ranked[:k]))
            if all(e in top_text for e in evid):
                hits[k] += 1
        top3_text = norm(" ".join(chunks[ci] for ci, _ in ranked[:3]))
        if not all(e in top3_text for e in evid):
            failures.append((item["id"], item["question"], ranked[:3]))

    unans_top1 = [retrieve(i["question"], model, index, 1)[0][1] for i in unanswerable]

    n = len(answerable)
    return {
        "n_chunks": len(chunks),
        "hit": {k: hits[k] / n for k in ks},
        "mrr": rr_sum / n,
        "ans_top1": ans_top1,
        "unans_top1": unans_top1,
        "failures": failures,
    }


def print_retrieval(result):
    mode = "vector only"
    if USE_HYBRID:
        mode = "hybrid (vector + keyword)"
    if USE_RERANK:
        mode += " + cross-encoder rerank"
    print(f"Search mode: {mode}")
    print(f"Chunks: {result['n_chunks']}")
    for k, v in result["hit"].items():
        print(f"  hit@{k}: {v:.0%}")
    print(f"  MRR:    {result['mrr']:.2f}")
    if USE_HYBRID or USE_RERANK:
        print("\n(Similarity-score statistics are skipped in this mode: scores are not cosine similarities.)")
    else:
        a, u = result["ans_top1"], result["unans_top1"]
        print(f"\nTop-1 similarity, answerable questions:   min {min(a):.2f}  mean {sum(a)/len(a):.2f}")
        print(f"Top-1 similarity, unanswerable questions: max {max(u):.2f}  mean {sum(u)/len(u):.2f}")
        if min(a) > max(u):
            print(f"  -> separable. A threshold between {max(u):.2f} and {min(a):.2f} would work, e.g. {(min(a)+max(u))/2:.2f}")
        else:
            print("  -> overlap: no single threshold separates them cleanly; consider a reranker or a stricter prompt.")
    if result["failures"]:
        print("\nRetrieval misses at top-3:")
        for qid, q, ranked in result["failures"]:
            print(f"  {qid}: {q}  (top chunks: {[(c, round(s, 3)) for c, s in ranked]})")


def evaluate_generation(eval_set, model, index, chunks, top_k, threshold, model_name):
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    correct = refused_ok = n_ans = n_unans = 0
    for item in eval_set:
        ranked = retrieve(item["question"], model, index, top_k)
        if threshold is not None and not USE_HYBRID and ranked[0][1] < threshold:
            answer = "I could not find that information in the document."
        else:
            answer = generate_answer(item["question"], [chunks[ci] for ci, _ in ranked], client, model_name)
        a = norm(answer)

        if item["type"] == "unanswerable":
            n_unans += 1
            ok = REFUSAL in a
            refused_ok += ok
        else:
            n_ans += 1
            ok = all(any(norm(alt) in a for alt in group) for group in item["answer_keywords"])
            correct += ok
        if not ok:
            print(f"FAIL {item['id']}: {item['question']}\n     got: {answer.strip()[:200]}\n")

    print(f"Answerable questions answered correctly: {correct}/{n_ans} ({correct/n_ans:.0%})")
    print(f"Unanswerable questions correctly refused: {refused_ok}/{n_unans} ({refused_ok/n_unans:.0%})")


def main():
    global USE_HYBRID, USE_RERANK, BM25, CHUNKS
    from dotenv import load_dotenv
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--eval-set", default="eval_set.json")
    ap.add_argument("--chunk-size", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--check-evidence", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--hybrid", action="store_true", help="vector + BM25 keyword search")
    ap.add_argument("--rerank", action="store_true", help="cross-encoder reranking of the top candidates")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    USE_HYBRID = args.hybrid
    USE_RERANK = args.rerank

    eval_set = load_eval_set(args.eval_set)
    text = extract_text_from_pdf(args.pdf)

    if args.check_evidence:
        check_evidence(eval_set, split_text_into_chunks(text, args.chunk_size, args.overlap))
        return

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    if args.sweep:
        print(f"Search mode: {'hybrid' if USE_HYBRID else 'vector only'}{' + rerank' if USE_RERANK else ''}")
        print(f"{'chunk/overlap':<14}{'chunks':>7}{'hit@1':>8}{'hit@3':>8}{'MRR':>7}")
        for size, ov in [(60, 15), (100, 20), (180, 40), (250, 50)]:
            chunks = split_text_into_chunks(text, size, ov)
            CHUNKS = chunks
            if USE_HYBRID:
                BM25 = build_bm25(chunks)
            r = evaluate_retrieval(eval_set, model, build_index(chunks, model), chunks)
            print(f"{f'{size}/{ov}':<14}{r['n_chunks']:>7}{r['hit'][1]:>8.0%}{r['hit'][3]:>8.0%}{r['mrr']:>7.2f}")
        return

    chunks = split_text_into_chunks(text, args.chunk_size, args.overlap)
    CHUNKS = chunks
    index = build_index(chunks, model)
    if USE_HYBRID:
        BM25 = build_bm25(chunks)

    if args.generate:
        evaluate_generation(eval_set, model, index, chunks, args.top_k, args.threshold,
                            os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    else:
        print_retrieval(evaluate_retrieval(eval_set, model, index, chunks))


if __name__ == "__main__":
    main()
