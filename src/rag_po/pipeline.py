from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rag_po.attack.llm_generate import build_attack_prompt, build_llm_poisoned_corpus, fallback_generated_posts
from rag_po.attack.random_inject import build_random_poisoned_corpus
from rag_po.attack.semantic_inject import build_semantic_poisoned_corpus
from rag_po.data_prep.parse_pheme import build_event_queries, parse_pheme_archive, write_processed_dataset
from rag_po.evaluation.llm_judge import judge_summary
from rag_po.evaluation.metrics import defense_filter_metrics, generation_metrics_from_judge, retrieval_metrics
from rag_po.io_utils import read_json, read_jsonl, write_json, write_jsonl
from rag_po.models import DEFENSE_SPECS, TweetDocument
from rag_po.rag.ecnu_client import ECNUClient
from rag_po.rag.generator import generate_summary
from rag_po.rag.retriever import retrieve_with_defense


@dataclass(frozen=True)
class ExperimentConfig:
    archive_path: str = "data/pheme_raw/pheme-rnr-dataset.tar.bz2"
    processed_dir: str = "data/processed"
    results_dir: str = "experiments/results"
    events: tuple[str, ...] = ()
    attacks: tuple[str, ...] = ("random",)
    ratios: tuple[float, ...] = (0.1,)
    defenses: tuple[str, ...] = ("D0", "D_all")
    retrieve_k: int = 10
    final_k: int = 5
    max_per_event_label: int | None = 30
    max_clean_docs: int = 24
    max_rumour_pool: int = 80
    embedding_batch_size: int = 16
    parallelism: int = 1
    chat_model: str = "ecnu-plus"
    allow_fallback_judge: bool = False


def load_or_parse_docs(config: ExperimentConfig) -> tuple[list[TweetDocument], dict[str, str]]:
    processed = Path(config.processed_dir)
    embedded_path = processed / "documents.embedded.jsonl"
    docs_path = embedded_path if embedded_path.exists() else processed / "documents.jsonl"
    queries_path = processed / "queries.json"
    if docs_path.exists() and queries_path.exists():
        docs = [TweetDocument.from_dict(row) for row in read_jsonl(docs_path)]
        queries = dict(read_json(queries_path))
        return docs, queries
    docs = parse_pheme_archive(config.archive_path, max_per_event_label=config.max_per_event_label)
    write_processed_dataset(docs, processed)
    queries = build_event_queries(doc.event for doc in docs)
    return docs, queries


def ensure_embeddings(
    client: ECNUClient,
    docs: list[TweetDocument],
    queries: dict[str, str],
    config: ExperimentConfig,
) -> tuple[list[TweetDocument], dict[str, list[float]]]:
    texts: list[str] = []
    doc_indices: list[int] = []
    mutable_docs = list(docs)
    for idx, doc in enumerate(mutable_docs):
        if doc.embedding is None:
            doc_indices.append(idx)
            texts.append(doc.text)
    query_items = [(event, query) for event, query in queries.items()]
    query_texts = [query for _, query in query_items]

    vectors = client.embed_texts(texts + query_texts, batch_size=config.embedding_batch_size, parallelism=config.parallelism)
    doc_vectors = vectors[: len(texts)]
    query_vectors = vectors[len(texts) :]
    for idx, vector in zip(doc_indices, doc_vectors):
        mutable_docs[idx] = mutable_docs[idx].with_embedding(vector)
    query_embeddings = {event: vector for (event, _), vector in zip(query_items, query_vectors)}

    write_jsonl(Path(config.processed_dir) / "documents.embedded.jsonl", (doc.to_dict() for doc in mutable_docs))
    write_json(Path(config.processed_dir) / "query_embeddings.json", query_embeddings)
    return mutable_docs, query_embeddings


def select_events(docs: list[TweetDocument], queries: dict[str, str], requested: Iterable[str]) -> list[str]:
    available = sorted({doc.event for doc in docs if doc.event in queries})
    requested_tuple = tuple(requested)
    if not requested_tuple:
        return available
    missing = sorted(set(requested_tuple) - set(available))
    if missing:
        raise ValueError(f"Requested events not in dataset: {missing}; available={available}")
    return list(requested_tuple)


def _limit_docs(docs: list[TweetDocument], limit: int) -> list[TweetDocument]:
    if limit <= 0:
        return docs
    return docs[:limit]


def build_corpus_for_attack(
    client: ECNUClient,
    docs: list[TweetDocument],
    event: str,
    query: str,
    query_embedding: list[float],
    attack: str,
    ratio: float,
    config: ExperimentConfig,
) -> list[TweetDocument]:
    clean = _limit_docs([doc for doc in docs if doc.event == event and doc.label == "non-rumour"], config.max_clean_docs)
    rumour_pool = [doc for doc in docs if doc.event != event and doc.label == "rumour"]
    if not rumour_pool:
        rumour_pool = [doc for doc in docs if doc.label == "rumour"]
    rumour_pool = _limit_docs(rumour_pool, config.max_rumour_pool)
    if attack == "random":
        return build_random_poisoned_corpus(clean, rumour_pool, ratio=ratio, seed=13)
    if attack == "semantic":
        return build_semantic_poisoned_corpus(clean, rumour_pool, query_embedding=query_embedding, ratio=ratio)
    if attack == "llm":
        n_needed = max(1, int(len(clean) * ratio + 0.999)) if ratio > 0 else 0
        if n_needed == 0:
            return list(clean)
        prompt = build_attack_prompt(event, query, clean, n_needed)
        content = client.chat(
            [{"role": "system", "content": "You generate JSON for a controlled misinformation robustness experiment."}, {"role": "user", "content": prompt}],
            temperature=0.7,
            cache_namespace="llm_attacks",
        )
        try:
            posts = json.loads(content)
            if not isinstance(posts, list) or not all(isinstance(x, str) for x in posts):
                raise ValueError("not a list of strings")
        except Exception:
            posts = fallback_generated_posts(event, query, n_needed)
        corpus = build_llm_poisoned_corpus(clean, posts, event=event, query=query, ratio=ratio)
        generated = [doc for doc in corpus if doc.embedding is None]
        if generated:
            vectors = client.embed_texts([doc.text for doc in generated], batch_size=config.embedding_batch_size, parallelism=config.parallelism)
            by_id = {doc.doc_id: doc.with_embedding(vector) for doc, vector in zip(generated, vectors)}
            corpus = [by_id.get(doc.doc_id, doc) for doc in corpus]
        return corpus
    raise ValueError(f"Unknown attack: {attack}")


def run_single_experiment(
    client: ECNUClient,
    docs: list[TweetDocument],
    queries: dict[str, str],
    query_embeddings: dict[str, list[float]],
    event: str,
    attack: str,
    ratio: float,
    defense: str,
    clean_summary: str | None,
    config: ExperimentConfig,
) -> tuple[dict[str, object], str]:
    if defense not in DEFENSE_SPECS:
        raise ValueError(f"Unknown defense: {defense}")
    query = queries[event]
    query_embedding = query_embeddings[event]
    clean_docs = _limit_docs([doc for doc in docs if doc.event == event and doc.label == "non-rumour"], config.max_clean_docs)
    if clean_summary is None:
        clean_retrieved, clean_audit, _ = retrieve_with_defense(
            clean_docs, query_embedding, defense="D0", retrieve_k=config.retrieve_k, final_k=config.final_k
        )
        clean_summary = generate_summary(client, query, clean_retrieved, critical=False)
    corpus = build_corpus_for_attack(client, docs, event, query, query_embedding, attack, ratio, config)
    retrieved, audit, _ = retrieve_with_defense(
        corpus, query_embedding, defense=defense, retrieve_k=config.retrieve_k, final_k=config.final_k
    )
    critical = DEFENSE_SPECS[defense].critical_prompt
    summary = generate_summary(client, query, retrieved, critical=critical)
    judge = judge_summary(
        client,
        query=query,
        clean_summary=clean_summary,
        eval_summary=summary,
        retrieved_docs=retrieved,
        allow_fallback=config.allow_fallback_judge,
    )
    initial_by_id = {doc.doc_id: doc for doc in corpus}
    initial_docs = [initial_by_id[doc_id] for doc_id in audit.get("initial_ids", []) if doc_id in initial_by_id]
    result = {
        "event": event,
        "query": query,
        "attack": attack,
        "ratio": ratio,
        "defense": defense,
        "retrieved": [doc.to_dict() for doc in retrieved],
        "retrieval_metrics": retrieval_metrics(retrieved),
        "generation_metrics": generation_metrics_from_judge(judge),
        "defense_metrics": defense_filter_metrics(initial_docs, retrieved),
        "summary": summary,
        "clean_summary": clean_summary,
        "judge": judge,
        "audit": audit,
    }
    return result, clean_summary


def _compute_clean_summary(
    client: ECNUClient,
    docs: list[TweetDocument],
    queries: dict[str, str],
    query_embeddings: dict[str, list[float]],
    event: str,
    config: ExperimentConfig,
) -> str:
    clean_docs = _limit_docs([doc for doc in docs if doc.event == event and doc.label == "non-rumour"], config.max_clean_docs)
    retrieved, _, _ = retrieve_with_defense(
        clean_docs, query_embeddings[event], defense="D0", retrieve_k=config.retrieve_k, final_k=config.final_k
    )
    return generate_summary(client, queries[event], retrieved, critical=False)


def run_experiments(config: ExperimentConfig, api_key: str | None = None) -> Path:
    client = ECNUClient(api_key=api_key, chat_model=config.chat_model, cache_dir="data/cache/ecnu")
    docs, queries = load_or_parse_docs(config)
    events = select_events(docs, queries, config.events)
    docs, query_embeddings = ensure_embeddings(client, docs, queries, config)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(config.results_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", asdict(config))

    clean_summaries = {
        event: _compute_clean_summary(client, docs, queries, query_embeddings, event, config) for event in events
    }
    tasks = [
        (event, attack, ratio, defense)
        for event in events
        for attack in config.attacks
        for ratio in config.ratios
        for defense in config.defenses
    ]
    rows: list[dict[str, object]] = []

    def run_task(task: tuple[str, str, float, str]) -> dict[str, object]:
        event, attack, ratio, defense = task
        row, _ = run_single_experiment(
            client,
            docs,
            queries,
            query_embeddings,
            event,
            attack,
            ratio,
            defense,
            clean_summaries[event],
            config,
        )
        return row

    if max(1, config.parallelism) == 1 or len(tasks) <= 1:
        for task in tasks:
            row = run_task(task)
            rows.append(row)
            write_json(out_dir / f"{row['event']}-{row['attack']}-{row['ratio']:g}-{row['defense']}.json", row)
    else:
        with ThreadPoolExecutor(max_workers=max(1, config.parallelism)) as executor:
            futures = {executor.submit(run_task, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                write_json(out_dir / f"{row['event']}-{row['attack']}-{row['ratio']:g}-{row['defense']}.json", row)
    rows.sort(key=lambda row: (str(row["event"]), str(row["attack"]), float(row["ratio"]), str(row["defense"])))
    write_json(out_dir / "results.json", rows)
    return out_dir
