import argparse
import json
import os
import re
import time

from dotenv import load_dotenv

import eval_rag as E

load_dotenv()

JUDGE_PROMPT = """You are checking whether an answer is supported by the context.

Context:
{context}

Answer:
{answer}

Is every factual claim in the answer supported by the context?
Reply with only JSON: {{"supported": true or false, "reason": "short reason"}}"""


def percentile(values, p):
    """Linear-interpolated percentile, e.g. percentile(xs, 75) for P75."""
    s = sorted(values)
    if not s:
        return None
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def judge_faithfulness(client, model, context, answer):
    r = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(context=context, answer=answer)}],
    )
    text = r.choices[0].message.content or ""
    match = re.search(r"\{.*\}", text, re.S)
    try:
        return bool(json.loads(match.group(0))["supported"])
    except Exception:
        return None  # judge output could not be parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--eval-set", default="eval_set_long.json")
    ap.add_argument("--chunk-size", type=int, default=60)
    ap.add_argument("--overlap", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()
    

    from groq import Groq
    from sentence_transformers import SentenceTransformer

    E.USE_HYBRID = True
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    gen_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    judge_model = os.getenv("JUDGE_MODEL", gen_model)

    eval_set = E.load_eval_set(args.eval_set)
    text = E.extract_text_from_pdf(args.pdf)
    model = SentenceTransformer(E.EMBED_MODEL)

    t0 = time.perf_counter()
    chunks = E.split_text_into_chunks(text, args.chunk_size, args.overlap)
    index = E.build_index(chunks, model)
    E.BM25 = E.build_bm25(chunks)
    index_seconds = time.perf_counter() - t0

    retrieval_s, generation_s, total_s, context_words = [], [], [], []
    supported = unsupported = unparsed = 0

    for item in eval_set:
        if item["type"] == "unanswerable":
            continue
        q = item["question"]

        start = time.perf_counter()
        ranked = E.retrieve(q, model, index, args.top_k)
        after_retrieval = time.perf_counter()

        ctx = [chunks[ci] for ci, _ in ranked]
        answer = E.generate_answer(q, ctx, client, gen_model)
        end = time.perf_counter()

        retrieval_s.append(after_retrieval - start)
        generation_s.append(end - after_retrieval)
        total_s.append(end - start)
        context_words.append(sum(len(c.split()) for c in ctx))

        verdict = judge_faithfulness(client, judge_model, "\n\n".join(ctx), answer)
        if verdict is False:
            print(f"\nUNSUPPORTED {item['id']}: {q}")
            print(f"ANSWER: {answer}\n")
            print("CONTEXT:\n" + "\n\n".join(ctx) + "\n" + "-" * 60)
        if verdict is None:
            unparsed += 1
        elif verdict:
            supported += 1
        else:
            unsupported += 1

    n = len(total_s)
    judged = supported + unsupported
    print(f"Questions measured: {n}")
    print(f"Indexing time (embed + keyword index, {len(chunks)} chunks): {index_seconds:.1f} s")
    print()
    print(f"{'Stage':<12}{'P50 (s)':>10}{'P75 (s)':>10}{'P95 (s)':>10}")
    for name, values in [("Retrieval", retrieval_s), ("Generation", generation_s), ("Total", total_s)]:
        print(f"{name:<12}{percentile(values,50):>10.2f}{percentile(values,75):>10.2f}{percentile(values,95):>10.2f}")
    print()
    print(f"Faithfulness: {supported}/{judged} answers judged fully supported "
          f"({supported/judged:.0%})" if judged else "Faithfulness: no answers judged")
    print(f"Judge output not parseable: {unparsed}")
    print(f"Average context sent to LLM: {sum(context_words)/n:.0f} words per question")


if __name__ == "__main__":
    main()