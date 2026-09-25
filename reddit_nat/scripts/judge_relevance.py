"""Binary LLM-judged relevance on Reddit (query, retrieved-article) pairs.

Three subcommands:
  generate  build requests_part_NN.jsonl, dedupe (q, doc) pairs, print sample + cost
  submit    upload batch to OpenAI (billable API request)
  poll      wait until terminal; download output_<id>.jsonl + parse to judgments.jsonl

Builds the binary relevance-judge batch with:
  - binary 0/1 schema (no rationale, no confidence)
  - reasoning_effort: minimal
  - Reddit-specific prompt (no reference answer; Reddit has no gold)
  - top-10 retrievals across 5 dense + BM25, deduped by (query_id, article_id)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.judge import estimate_batch_tokens, extract_response_text  # noqa: E402
from reddit_nat import lib  # noqa: E402

CFG_PATH = REPO / "reddit_nat" / "configs" / "judge_relevance.yaml"
RUN_DIR = REPO / "reddit_nat" / "results"
JUDGE_DIR = RUN_DIR / "judge_relevance"
QUERIES_CSV = REPO / "reddit_nat" / "data" / "reddit_nat_raw.csv"
CORPUS_JSONL = REPO / "allsides_synth" / "data" / "corpus.jsonl"
BM25_CSV = lib.bm25_candidate_file(RUN_DIR)


def load_config() -> dict[str, Any]:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def load_queries(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(QUERIES_CSV)
    missing = sorted({"title", "selftext"} - set(df.columns))
    if missing:
        raise SystemExit(f"{missing} not in {QUERIES_CSV.name}: the release withholds Reddit post "
                         "text. Rehydrate it first; see reddit_nat/README.md.")
    df = df[df["ideology"].isin(cfg["keep_ideologies"])].copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["selftext"] = df["selftext"].fillna("").astype(str)
    return df


def load_corpus_index() -> dict[str, dict]:
    index = {}
    with open(CORPUS_JSONL) as f:
        for line in f:
            d = json.loads(line)
            index[d["article_id"]] = {
                "heading": d.get("heading", ""),
                "source": d.get("source", ""),
                "text": d.get("text", ""),
                "stance": d.get("stance", ""),
            }
    return index


def load_candidates() -> pd.DataFrame:
    cols = ["query_id", "encoder", "rank", "retrieved_article_id"]
    dense = lib.load_dense_candidates(RUN_DIR, usecols=cols)
    bm25 = pd.read_csv(BM25_CSV, usecols=cols)
    return pd.concat([dense, bm25], ignore_index=True)


def build_pairs(cfg: dict) -> list[dict]:
    """Return list of {query_id, article_id, query_text, doc_text} for unique pairs."""
    queries = load_queries(cfg)
    keep_qids = set(queries["id"])
    cands = load_candidates()
    cands = cands[cands["query_id"].isin(keep_qids) & (cands["rank"] <= cfg["top_k_to_judge"])]
    pairs = (
        cands[["query_id", "retrieved_article_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"queries (lib+con): {len(queries)}")
    print(f"retrieved rows @ k<={cfg['top_k_to_judge']}: {len(cands)}")
    print(f"unique (q, doc) pairs to judge: {len(pairs)}")

    queries = queries.drop_duplicates(subset=["id"], keep="first")
    qmap = queries.set_index("id")[["title", "selftext"]].to_dict("index")
    corpus = load_corpus_index()

    qmax = cfg["query_max_chars"]
    dmax = cfg["doc_max_chars"]

    out = []
    for _, row in pairs.iterrows():
        q = qmap[row["query_id"]]
        title = (q["title"] or "").strip()
        selftext = (q["selftext"] or "").strip()
        query_text = (title + ("\n\n" + selftext if selftext else "")).strip()[:qmax]

        doc = corpus.get(row["retrieved_article_id"])
        if doc is None:
            continue
        heading = (doc["heading"] or "").strip()
        body = (doc["text"] or "").strip()
        source = (doc["source"] or "").strip()
        doc_text = ((f"[{source}] " if source else "") + heading + "\n\n" + body).strip()[:dmax]

        out.append({
            "query_id": row["query_id"],
            "article_id": row["retrieved_article_id"],
            "query_text": query_text,
            "doc_text": doc_text,
        })
    print(f"pairs with corpus hits: {len(out)}")
    return out


def build_judge_prompt(query_text: str, doc_text: str) -> str:
    return (
        "Question (Reddit post):\n"
        f"{query_text}\n\n"
        "Retrieved article:\n"
        f"{doc_text}\n\n"
        "Does the article help the Reddit user reach an answer to their question?\n"
        "Grade 1 if on-topic and at least partly informative; 0 otherwise.\n"
        "Return JSON: {\"grade\": 0 or 1}."
    )


def build_batch(pairs: list[dict], cfg: dict) -> list[dict]:
    schema = cfg["response_schema"]
    records = []
    for p in pairs:
        prompt = build_judge_prompt(p["query_text"], p["doc_text"])
        records.append({
            "custom_id": f"{p['query_id']}::{p['article_id']}",
            "method": "POST",
            "url": cfg["endpoint"],
            "body": {
                "model": cfg["model"],
                "max_output_tokens": cfg["max_output_tokens"],
                "reasoning": {"effort": cfg["reasoning_effort"]},
                "input": [
                    {"role": "system", "content": [{"type": "input_text", "text": cfg["system_prompt"]}]},
                    {"role": "user",   "content": [{"type": "input_text", "text": prompt}]},
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema["name"],
                        "schema": schema["schema"],
                        "strict": True,
                    }
                },
            },
        })
    return records


def estimate_cost(records: list[dict], cfg: dict) -> dict:
    tok = estimate_batch_tokens(records, cfg["model"])
    in_tok = tok["input_tokens"]
    out_tok = len(records) * cfg["expected_output_tokens_per_request"]
    in_cost = in_tok / 1e6 * cfg["pricing"]["input_per_million"]
    out_cost = out_tok / 1e6 * cfg["pricing"]["output_per_million"]
    return {
        "n_requests": len(records),
        "input_tokens": in_tok,
        "output_tokens_estimate": out_tok,
        "input_cost": in_cost,
        "output_cost": out_cost,
        "total_cost": in_cost + out_cost,
        "estimation_method": tok["estimation_method"],
    }


def cmd_generate(args) -> None:
    cfg = load_config()
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(cfg)
    if args.limit:
        pairs = pairs[: args.limit]
        print(f"[--limit] truncated to {len(pairs)} pairs (dry-run mode)")
    records = build_batch(pairs, cfg)

    # OpenAI batch API caps at 50,000 requests per batch. Chunk to ~40k each.
    CHUNK = cfg.get("requests_per_batch", 40000)
    n_chunks = (len(records) + CHUNK - 1) // CHUNK
    for i in range(n_chunks):
        chunk = records[i * CHUNK: (i + 1) * CHUNK]
        out_path = JUDGE_DIR / f"requests_part_{i+1:02d}.jsonl"
        with open(out_path, "w") as f:
            for r in chunk:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {out_path} ({len(chunk)} requests)")

    print("\n=== SAMPLE PROMPT (first request) ===")
    sys_text = records[0]["body"]["input"][0]["content"][0]["text"]
    user_text = records[0]["body"]["input"][1]["content"][0]["text"]
    print(f"--- system ---\n{sys_text}")
    print(f"--- user ---\n{user_text[:1500]}{'...' if len(user_text) > 1500 else ''}")
    print(f"--- schema ---\n{json.dumps(records[0]['body']['text'], indent=2)}")

    print("\n=== COST ESTIMATE ===")
    est = estimate_cost(records, cfg)
    for k, v in est.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")


def cmd_submit(args) -> None:
    from openai import OpenAI

    cfg = load_config()
    parts = sorted(JUDGE_DIR.glob("requests_part_*.jsonl"))
    if not parts:
        sys.exit(f"no requests_part_*.jsonl in {JUDGE_DIR}; run `generate` first")

    client = OpenAI()
    batch_ids = []
    for p in parts:
        print(f"uploading {p.name}...")
        up = client.files.create(file=p.open("rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=up.id,
            endpoint=cfg["endpoint"],
            completion_window="24h",
            metadata={"experiment": "reddit_relevance", "part": p.stem},
        )
        print(f"  {p.name}: file={up.id} batch={batch.id} ({batch.status})")
        batch_ids.append(batch.id)
    (JUDGE_DIR / "batch_ids.txt").write_text("\n".join(batch_ids) + "\n")
    print(f"  wrote {JUDGE_DIR/'batch_ids.txt'}")


def cmd_poll(args) -> None:
    from openai import OpenAI

    ids_path = JUDGE_DIR / "batch_ids.txt"
    if not ids_path.exists():
        sys.exit(f"missing {ids_path}; run `submit` first")
    bids = [x.strip() for x in ids_path.read_text().splitlines() if x.strip()]

    client = OpenAI()
    # Poll until all terminal
    while True:
        statuses = []
        for bid in bids:
            b = client.batches.retrieve(bid)
            statuses.append((bid, b))
        line = " | ".join(f"{b.status} {b.request_counts.completed}/{b.request_counts.total}"
                          for _, b in statuses)
        print(f"[{time.strftime('%H:%M:%S')}] {line}")
        if all(b.status in ("completed", "failed", "expired", "cancelled") for _, b in statuses):
            break
        time.sleep(60)

    # Write to a temp file and only replace judgments.jsonl when every batch
    # completed and every response parsed: lib.filtered_per_query_lean treats
    # a missing judgment as grade 0, so a silently partial file skews the lean gap.
    judgments_path = JUDGE_DIR / "judgments.jsonl"
    tmp_path = judgments_path.with_suffix(".jsonl.tmp")
    not_completed = [(bid, b.status) for bid, b in statuses if b.status != "completed"]
    failed = [(bid, b.request_counts.failed) for bid, b in statuses
              if b.status == "completed" and (b.request_counts.failed or 0) > 0]
    n_skipped = 0
    n_written = 0
    with open(tmp_path, "w") as fout:
        for bid, b in statuses:
            if b.status != "completed":
                continue
            out_path = JUDGE_DIR / f"output_{bid[-8:]}.jsonl"
            content = client.files.content(b.output_file_id)
            out_path.write_bytes(content.read())
            with open(out_path) as fin:
                for line in fin:
                    d = json.loads(line)
                    cid = d.get("custom_id")
                    resp = d.get("response", {}).get("body")
                    if not resp:
                        print(f"missing response body {cid}", file=sys.stderr)
                        n_skipped += 1
                        continue
                    text = extract_response_text(resp)
                    if not text.strip():
                        print(f"empty response {cid}", file=sys.stderr)
                        n_skipped += 1
                        continue
                    try:
                        grade = int(json.loads(text)["grade"])
                    except Exception as e:
                        print(f"parse fail {cid}: {e}", file=sys.stderr)
                        n_skipped += 1
                        continue
                    qid, aid = cid.split("::", 1)
                    fout.write(json.dumps({"query_id": qid, "article_id": aid, "grade": grade}) + "\n")
                    n_written += 1
    if not_completed or failed or n_skipped:
        problems = []
        if not_completed:
            problems.append("batches not completed: "
                            + ", ".join(f"{bid}={st}" for bid, st in not_completed))
        if failed:
            problems.append("failed requests (see each batch's error_file_id): "
                            + ", ".join(f"{bid}={n}" for bid, n in failed))
        if n_skipped:
            problems.append(f"{n_skipped} responses skipped (missing/empty/unparseable)")
        sys.exit(
            f"ERROR: judgments incomplete ({'; '.join(problems)}). "
            f"Partial output kept at {tmp_path}; {judgments_path} was NOT replaced."
        )
    tmp_path.replace(judgments_path)
    print(f"wrote {judgments_path} ({n_written} judgments)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--limit", type=int, default=0, help="dry-run with first N pairs")
    sub.add_parser("submit")
    sub.add_parser("poll")
    args = ap.parse_args()

    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "submit":
        cmd_submit(args)
    elif args.cmd == "poll":
        cmd_poll(args)


if __name__ == "__main__":
    main()
