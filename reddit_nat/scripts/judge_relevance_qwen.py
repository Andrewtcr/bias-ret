"""Run Qwen3.5-35B-A3B as a second relevance judge.

Reads OpenAI-format requests from judge_relevance/requests_part_*.jsonl and
writes judgments_qwen.jsonl with query_id, article_id, and grade. Request
files require rehydrated post text and judge_relevance.py generate.
Completed pairs are skipped when resuming. Adjust tensor/pipeline parallelism
for available GPUs; defaults use four-way pipeline parallelism.

Usage:
  python reddit_nat/scripts/judge_relevance_qwen.py --tp 1 --pp 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JUDGE_DIR = REPO / "reddit_nat" / "results" / "judge_relevance"
OUT_PATH = JUDGE_DIR / "judgments_qwen.jsonl"


def load_pairs() -> list[dict]:
    pairs: list[dict] = []
    for part in sorted(JUDGE_DIR.glob("requests_part_*.jsonl")):
        with open(part) as f:
            for line in f:
                d = json.loads(line)
                inputs = d["body"]["input"]
                system = next((c["content"][0]["text"]
                               for c in inputs if c["role"] == "system"), "")
                user = next((c["content"][0]["text"]
                             for c in inputs if c["role"] == "user"), "")
                qid, aid = d["custom_id"].split("::", 1)
                pairs.append({"query_id": qid, "article_id": aid,
                              "system": system, "user": user})
    return pairs


def already_done(out_path: Path) -> set[tuple[str, str]]:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    # grade=None rows are parse failures kept for debugging;
                    # they are NOT done — re-judge them on resume.
                    if d.get("grade") is None:
                        continue
                    done.add((d["query_id"], d["article_id"]))
                except Exception:
                    continue
    return done


def parse_grade(text: str) -> int | None:
    """Robust grade parser: try strict JSON, then fallback to digit search."""
    text = text.strip()
    try:
        d = json.loads(text)
        g = int(d.get("grade", -1))
        if g in (0, 1):
            return g
    except Exception:
        pass
    # Fallback: find first 0 or 1 after "grade"
    import re
    m = re.search(r'"?grade"?\s*:\s*([01])', text)
    if m:
        return int(m.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=4,
                    help="Pipeline parallelism degree")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--mem-util", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=5_000,
                    help="Process this many pairs per vLLM.generate() call (also = checkpoint granularity).")
    args = ap.parse_args()

    print(f"loading pairs...")
    pairs = load_pairs()
    print(f"  {len(pairs):,} total pairs")
    done = already_done(OUT_PATH)
    print(f"  {len(done):,} already done (resuming)")
    pairs = [p for p in pairs if (p["query_id"], p["article_id"]) not in done]
    print(f"  {len(pairs):,} remaining")

    if not pairs:
        print("nothing to do")
        return

    from vllm import LLM, SamplingParams

    print(f"loading {args.model} (tp={args.tp} pp={args.pp})...")
    t0 = time.perf_counter()
    llm = LLM(model=args.model,
              tensor_parallel_size=args.tp,
              pipeline_parallel_size=args.pp,
              dtype="bfloat16",
              max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=args.mem_util,
              trust_remote_code=True)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    t_start = time.perf_counter()
    n_done = 0
    out_fp = open(OUT_PATH, "a")
    parse_fails = 0
    grade_counts = {0: 0, 1: 0, None: 0}

    for chunk_start in range(0, len(pairs), args.chunk):
        chunk = pairs[chunk_start:chunk_start + args.chunk]
        t_chunk = time.perf_counter()
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": p["system"]},
                 {"role": "user", "content": p["user"]}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            for p in chunk
        ]
        outputs = llm.generate(prompts, sp)
        for p, o in zip(chunk, outputs):
            text = o.outputs[0].text
            grade = parse_grade(text)
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            if grade is None:
                parse_fails += 1
            out_fp.write(json.dumps({
                "query_id": p["query_id"], "article_id": p["article_id"],
                "grade": grade, "raw": text if grade is None else None,
            }) + "\n")
        out_fp.flush()
        n_done += len(chunk)
        dt_chunk = time.perf_counter() - t_chunk
        dt_total = time.perf_counter() - t_start
        eta = (len(pairs) - n_done) / (n_done / dt_total) if n_done > 0 else float("inf")
        print(f"[{n_done:,}/{len(pairs):,}] "
              f"chunk {dt_chunk:.0f}s  "
              f"total {dt_total/60:.1f}m  "
              f"eta {eta/60:.1f}m  "
              f"grade_0={grade_counts.get(0,0):,} "
              f"grade_1={grade_counts.get(1,0):,} "
              f"parse_fail={parse_fails:,}")

    out_fp.close()
    print(f"DONE. {n_done:,} judgments written to {OUT_PATH}")


if __name__ == "__main__":
    main()
