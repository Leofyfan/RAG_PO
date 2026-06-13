from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from rag_po.data_prep.download import download_pheme_archive, verify_pheme_archive
from rag_po.data_prep.parse_pheme import parse_pheme_archive, write_processed_dataset
from rag_po.evaluation.visualize import build_html_report, find_latest_results_json, flatten_results
from rag_po.pipeline import ExperimentConfig, run_experiments


def _tuple_csv(value: str, cast=str):
    if value is None or value == "":
        return tuple()
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def prepare_limit_from_args(args: argparse.Namespace) -> int | None:
    return None if args.max_per_event_label < 0 else args.max_per_event_label


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--archive-path", default="data/pheme_raw/pheme-rnr-dataset.tar.bz2")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--events", default="", help="Comma-separated PHEME event ids. Empty means all parsed events.")
    parser.add_argument("--attacks", default="random", help="Comma-separated: random,semantic,llm,a1,a2,a3")
    parser.add_argument("--ratios", default="0.1", help="Comma-separated poison ratios, e.g. 0,0.1,0.3,0.5")
    parser.add_argument("--defenses", default="D0,D_all", help="Comma-separated: D0,D1,D2,D3,D4,D123,D34,D234,D_all,D_star")
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--max-per-event-label", type=int, default=30)
    parser.add_argument("--max-clean-docs", type=int, default=24)
    parser.add_argument("--max-rumour-pool", type=int, default=80)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--query-variants", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--chat-model", default="ecnu-plus", choices=["ecnu-plus", "ecnu-max"])
    parser.add_argument("--allow-fallback-judge", action="store_true")
    parser.add_argument("--resume-dir", default="", help="Existing result directory to resume; completed cell JSON files are skipped.")
    parser.add_argument("--api-timeout", type=int, default=120)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--max-llm-attack-posts", type=int, default=24, help="Cap posts requested from the LLM; remaining poison docs use diverse adaptive templates. 0 disables the cap.")
    parser.add_argument("--skip-llm-attack-api", action="store_true", help="Use adaptive fallback templates for poison text generation while keeping ECNU embeddings, summaries, and judges.")
    parser.add_argument("--api-key", default=None, help="Prefer ECNU_API_KEY env var; this flag is for local runs only.")


def command_download(args: argparse.Namespace) -> None:
    path = download_pheme_archive(args.out)
    ok = verify_pheme_archive(path)
    print(f"archive={path} bytes={path.stat().st_size} verified={ok}")


def command_prepare(args: argparse.Namespace) -> None:
    archive = Path(args.archive_path)
    if not archive.exists():
        download_pheme_archive(archive)
    docs = parse_pheme_archive(archive, max_per_event_label=prepare_limit_from_args(args))
    summary = write_processed_dataset(docs, args.processed_dir)
    print(f"processed_dir={args.processed_dir} docs={summary['num_docs']} events={len(summary['events'])}")


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        archive_path=args.archive_path,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        events=_tuple_csv(args.events, str),
        attacks=_tuple_csv(args.attacks, str),
        ratios=_tuple_csv(args.ratios, float),
        defenses=_tuple_csv(args.defenses, str),
        retrieve_k=args.retrieve_k,
        final_k=args.final_k,
        max_per_event_label=prepare_limit_from_args(args),
        max_clean_docs=args.max_clean_docs,
        max_rumour_pool=args.max_rumour_pool,
        embedding_batch_size=args.embedding_batch_size,
        parallelism=args.parallelism,
        query_variants=args.query_variants,
        repeats=args.repeats,
        chat_model=args.chat_model,
        allow_fallback_judge=args.allow_fallback_judge,
        resume_dir=args.resume_dir,
        api_timeout=args.api_timeout,
        api_retries=args.api_retries,
        max_llm_attack_posts=args.max_llm_attack_posts,
        skip_llm_attack_api=args.skip_llm_attack_api,
    )


def command_run(args: argparse.Namespace) -> None:
    out_dir = run_experiments(config_from_args(args), api_key=args.api_key or os.getenv("ECNU_API_KEY"))
    csv_path = flatten_results(out_dir / "results.json")
    print(f"results_dir={out_dir}")
    print(f"csv={csv_path}")


def command_smoke(args: argparse.Namespace) -> None:
    args.events = args.events or "germanwings-crash"
    args.attacks = args.attacks or "llm"
    args.ratios = args.ratios or "0.1"
    args.defenses = args.defenses or "D_all"
    args.max_per_event_label = min(args.max_per_event_label, 12)
    args.max_clean_docs = min(args.max_clean_docs, 8)
    args.max_rumour_pool = min(args.max_rumour_pool, 20)
    args.retrieve_k = min(args.retrieve_k, 6)
    args.final_k = min(args.final_k, 4)
    args.embedding_batch_size = min(args.embedding_batch_size, 8)
    args.parallelism = 1
    command_run(args)


def command_flatten(args: argparse.Namespace) -> None:
    csv_path = flatten_results(args.results_json, args.out)
    print(f"csv={csv_path}")


def command_visualize(args: argparse.Namespace) -> None:
    results_json = Path(args.results_json) if args.results_json else find_latest_results_json(args.root)
    html_path = build_html_report(results_json, args.out)
    print(f"html={html_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG poisoning attack/defense experiments with ECNU APIs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download-data", help="Download the PHEME RNR archive into this project.")
    p_download.add_argument("--out", default="data/pheme_raw/pheme-rnr-dataset.tar.bz2")
    p_download.set_defaults(func=command_download)

    p_prepare = sub.add_parser("prepare-data", help="Parse PHEME into JSONL under data/processed.")
    p_prepare.add_argument("--archive-path", default="data/pheme_raw/pheme-rnr-dataset.tar.bz2")
    p_prepare.add_argument("--processed-dir", default="data/processed")
    p_prepare.add_argument("--max-per-event-label", type=int, default=30)
    p_prepare.set_defaults(func=command_prepare)

    p_run = sub.add_parser("run", help="Run an experiment matrix.")
    _add_run_args(p_run)
    p_run.set_defaults(func=command_run)

    p_smoke = sub.add_parser("smoke", help="Run one tiny real ECNU-backed experiment with parallelism=1.")
    _add_run_args(p_smoke)
    p_smoke.set_defaults(
        func=command_smoke,
        events="germanwings-crash",
        attacks="llm",
        ratios="0.1",
        defenses="D_all",
        max_per_event_label=12,
        max_clean_docs=8,
        max_rumour_pool=20,
        retrieve_k=6,
        final_k=4,
        embedding_batch_size=8,
        parallelism=1,
    )

    p_flatten = sub.add_parser("flatten", help="Flatten results.json to CSV.")
    p_flatten.add_argument("results_json")
    p_flatten.add_argument("--out", default=None)
    p_flatten.set_defaults(func=command_flatten)

    p_visualize = sub.add_parser("visualize", help="Render a self-contained HTML/SVG report for results.json.")
    p_visualize.add_argument("results_json", nargs="?", help="Path to results.json. Empty means latest under --root.")
    p_visualize.add_argument("--out", default=None)
    p_visualize.add_argument("--root", default="experiments")
    p_visualize.set_defaults(func=command_visualize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
