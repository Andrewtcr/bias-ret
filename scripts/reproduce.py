"""Rebuild paper figures and tables from committed data, without model inference.

Run with the same Python environment used for `pip install -e .`. Each step
runs from the repository root and must refresh its declared outputs. Detailed
stdout/stderr is saved under figs/output/logs/; failures stop the run.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Step:
    name: str
    group: str
    script: str
    outputs: tuple[str, ...]
    args: tuple[str, ...] = ()


def plots(*names: str) -> tuple[str, ...]:
    return tuple(f"figs/output/{name}.{ext}" for name in names for ext in ("pdf", "png"))


# Tables precede figures that consume them; lexical regression follows scoring.
STEPS = (
    Step("political-gold-test", "tables", "figs/scripts/analyze_political_gold.py",
         ("figs/results/gold_filtered_paired_test_k10.csv",)),
    Step("lexical-scores", "tables", "allsides_synth/scripts/analyze_lexical_scores.py",
         tuple(f"allsides_synth/results/{name}.csv" for name in
               ("lexical_logodds", "cell_lex_asymmetry"))),
    Step("lexical-regression", "tables", "allsides_synth/scripts/analyze_lexical_regression.py",
         ("allsides_synth/results/lex_regression_per_encoder_gold_filtered.csv",)),
    Step("hcmagic-statistics", "tables", "hcmagic/scripts/analyze_retrieval.py",
         ("hcmagic/results/hcmagic_synth_paired.csv", "hcmagic/results/hcmagic_real_paired.csv"),
         ("--config", "hcmagic/configs/retrieval.yaml")),
    Step("dialect-lexical-synthetic", "tables", "hcmagic/scripts/analyze_lexical_synthetic.py",
         tuple(f"hcmagic/results/lex_residual_{name}.csv" for name in
               ("summary", "token_logodds"))),
    Step("dialect-lexical-natural", "tables", "hcmagic/scripts/analyze_lexical_natural.py",
         tuple(f"hcmagic/results/lex_residual_real_{name}.csv" for name in
               ("summary", "token_logodds"))),
    Step("reddit-lexical-regression", "tables", "reddit_nat/scripts/analyze_lexical.py",
         ("reddit_nat/results/lex_residual_summary.csv", "reddit_nat/results/lex_residual.md"),
         ("--from-scores",)),
    Step("reddit-lexical-regression-gold", "tables", "reddit_nat/scripts/analyze_lexical.py",
         ("reddit_nat/results/lex_residual_gold_summary.csv",),
         ("--gold-filtered",)),
    Step("query-lengths", "tables", "figs/scripts/analyze_query_lengths.py",
         ("figs/results/query_length_stats.csv", "figs/results/hcmagic_dropped_queries.csv")),
    Step("judge-agreement", "tables", "reddit_nat/scripts/analyze_judge_agreement.py",
         tuple(f"reddit_nat/results/judge_relevance/{name}" for name in
               ("agreement_overall.csv", "kappa_per_encoder.csv",
                "lean_gap_by_filter.csv", "agreement_summary.md"))),
    Step("political-figure", "figures", "figs/scripts/plot_political_overview.py",
         plots("political_combined") + ("reddit_nat/results/gold_filtered_lean_summary.csv",)),
    Step("dialect-figure", "figures", "figs/scripts/plot_dialect_overview.py",
         plots("aalwme_combined")),
    Step("relevance-figures", "figures", "figs/scripts/plot_relevance_robustness.py",
         plots("gold_filter_unfiltered", "gold_filter_synth_llm", "gold_filter_reddit_full10",
               "gold_filter_relevance_dist", "gold_filter_confusion")
         + ("figs/results/gold_filter_conditions.csv",)),
    Step("political-depth", "figures", "figs/scripts/plot_political_retrieval.py",
         plots("political_ksweep")),
    Step("reddit-depth", "figures", "figs/scripts/plot_reddit_k_sweep.py",
         plots("real_arm_political_ksweep")),
    Step("dialect-depth", "figures", "hcmagic/scripts/plot_dialect_retrieval.py",
         plots("aalwme_synth_ksweep", "aalwme_real_unpaired_ksweep",
               "aalwme_translation_forest", "aalwme_translation_ksweep")),
    Step("pairing-similarity", "figures", "allsides_synth/scripts/plot_pairing_similarity.py",
         plots("pairing_sim_boxes_grouped")
         + ("allsides_synth/results/pairing_similarity/pairing_mean_summary.csv",)),
    Step("probing-figure", "figures", "probing/scripts/plot_paper_probes.py",
         plots("probe_combined_subset")),
)


def output_stamp(path: Path) -> tuple[int, int, int] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


def run_step(step: Step, root: Path = ROOT) -> None:
    """Execute one producer; reject failed, missing, empty, or stale outputs."""
    before = {name: output_stamp(root / name) for name in step.outputs}
    log_dir = root / "figs" / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{step.name}.log"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.setdefault("MPLCONFIGDIR", str(log_dir.parent / ".matplotlib"))
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    with log_path.open("w") as log:
        result = subprocess.run(
            [sys.executable, "-u", str(root / step.script), *step.args],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    if result.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(f"{step.name} exited {result.returncode}. Log: {log_path}\n{tail}")
    problems = []
    for name in step.outputs:
        after = output_stamp(root / name)
        if after is None or after[2] == 0:
            problems.append(f"missing or empty: {name}")
        elif after == before[name]:
            problems.append(f"not refreshed: {name}")
    if problems:
        raise RuntimeError(f"{step.name}: {'; '.join(problems)}. Log: {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("figures", "tables"),
                        help="Run one group; figures read the committed tables, and some figure "
                             "steps also write the tables they compute (see --list).")
    parser.add_argument("--list", action="store_true", help="List commands and expected outputs without running.")
    args = parser.parse_args()
    steps = [step for step in STEPS if args.only is None or step.group == args.only]
    if args.list:
        for step in steps:
            print(f"{step.group}: {step.name}\n  python {step.script} {' '.join(step.args)}".rstrip())
            for output in step.outputs:
                print(f"    -> {output}")
        return 0
    print("Rebuilding committed results and rendered figures; no model downloads or API calls.", flush=True)
    if args.only != "figures":
        print("Table calculations include the full HealthCareMagic bootstrap/permutation analysis.", flush=True)
    started = time.monotonic()
    try:
        for i, step in enumerate(steps, 1):
            print(f"[{i}/{len(steps)}] {step.name} ... (log: figs/output/logs/{step.name}.log)", flush=True)
            begin = time.monotonic()
            run_step(step)
            print(f"  OK: {len(step.outputs)} outputs, {time.monotonic() - begin:.1f}s", flush=True)
    except (RuntimeError, OSError) as exc:
        print(f"[reproduce] FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"[reproduce] OK: {len(steps)} steps, {sum(len(s.outputs) for s in steps)} outputs, "
          f"{time.monotonic() - started:.1f}s. Figures: {ROOT / 'figs/output'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
