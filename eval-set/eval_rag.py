"""
Evaluation harness for the PDF RAG chatbot.

Retrieval metrics need no API key. Answer metrics (--generate) need GROQ_API_KEY.

Examples:
  python eval_rag.py --pdf handbook.pdf --check-evidence      # validate the eval set (no models needed)
  python eval_rag.py --pdf handbook.pdf                       # retrieval baseline
  python eval_rag.py --pdf handbook.pdf --sweep               # compare chunk sizes
  python eval_rag.py --pdf handbook.pdf --generate            # answers, no threshold
  python eval_rag.py --pdf handbook.pdf --generate --threshold 0.35
"""
import argparse
import json
import os
import re

REFUSAL = "i could not find that information in the document"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def norm(text):
    """Lowercase and collapse whitespace so line breaks never break matching."""
    return re.sub(r"\s+", " ", text).lower()


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


def search(question, model, index, k):
    q = model.encode_query(question, convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q.astype("float32").reshape(1, -1), k)
    return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]


def generate_answer(question, chunks, client, model_name):
    system_prompt = (
        "You are a document question-answering assistant.\n"
        "Answer the user's question using only the context provided from the uploaded document.\n"
        'If the answer is not available in the context, say:\n'
        '"I could not find that information in the document."\n'
        "Do not use outside knowledge.\n"
        "If the document mentions the topic but does not give the exact detail asked for, say what the document does state instead of refusing.\n"
        "Keep the answer clear and concise."
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
            text = norm(" ".join(chunks))
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
        ranked = search(item["question"], model, index, max_k)
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

    unans_top1 = [search(i["question"], model, index, 1)[0][1] for i in unanswerable]

    n = len(answerable)
    result = {
        "n_chunks": len(chunks),
        "hit": {k: hits[k] / n for k in ks},
        "mrr": rr_sum / n,
        "ans_top1": ans_top1,
        "unans_top1": unans_top1,
        "failures": failures,
    }
    return result


def print_retrieval(result):
    print(f"Chunks: {result['n_chunks']}")
    for k, v in result["hit"].items():
        print(f"  hit@{k}: {v:.0%}")
    print(f"  MRR:    {result['mrr']:.2f}")
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
            print(f"  {qid}: {q}  (top chunks: {[(c, round(s, 2)) for c, s in ranked]})")


def evaluate_generation(eval_set, model, index, chunks, top_k, threshold, model_name):
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    correct = refused_ok = n_ans = n_unans = 0
    for item in eval_set:
        ranked = search(item["question"], model, index, top_k)
        if threshold is not None and ranked[0][1] < threshold:
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
            ok = all(any(alt in a for alt in group) for group in item["answer_keywords"])
            correct += ok
        if not ok:
            print(f"FAIL {item['id']}: {item['question']}\n     got: {answer.strip()[:200]}\n")

    print(f"Answerable questions answered correctly: {correct}/{n_ans} ({correct/n_ans:.0%})")
    print(f"Unanswerable questions correctly refused: {refused_ok}/{n_unans} ({refused_ok/n_unans:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--eval-set", default="eval_set.json")
    ap.add_argument("--chunk-size", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--check-evidence", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    eval_set = load_eval_set(args.eval_set)
    text = extract_text_from_pdf(args.pdf)

    if args.check_evidence:
        check_evidence(eval_set, split_text_into_chunks(text, args.chunk_size, args.overlap))
        return

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    if args.sweep:
        print(f"{'chunk/overlap':<14}{'chunks':>7}{'hit@1':>8}{'hit@3':>8}{'MRR':>7}")
        for size, ov in [(60, 15), (100, 20), (180, 40), (250, 50)]:
            chunks = split_text_into_chunks(text, size, ov)
            r = evaluate_retrieval(eval_set, model, build_index(chunks, model), chunks)
            print(f"{f'{size}/{ov}':<14}{r['n_chunks']:>7}{r['hit'][1]:>8.0%}{r['hit'][3]:>8.0%}{r['mrr']:>7.2f}")
        return

    chunks = split_text_into_chunks(text, args.chunk_size, args.overlap)
    index = build_index(chunks, model)

    if args.generate:
        evaluate_generation(eval_set, model, index, chunks, args.top_k, args.threshold,
                            os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    else:
        print_retrieval(evaluate_retrieval(eval_set, model, index, chunks))


if __name__ == "__main__":
    main()
