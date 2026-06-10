from __future__ import annotations

import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from rag_po.io_utils import write_json, write_jsonl
from rag_po.models import TweetDocument
from rag_po.text_utils import clean_tweet_text

DENY_WORDS = {
    "deny", "denies", "denied", "false", "fake", "hoax", "wrong", "untrue",
    "debunk", "debunked", "not true", "isn't true", "is not true", "rumor", "rumour",
}


def _safe_json_from_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> dict:
    extracted = tf.extractfile(member)
    if extracted is None:
        return {}
    try:
        return json.loads(extracted.read().decode("utf-8"))
    except Exception:
        return {}


def _path_parts(name: str) -> tuple[str, str, str] | None:
    parts = name.split("/")
    if any(part.startswith("._") for part in parts):
        return None
    if Path(name).name.startswith("."):
        return None
    if len(parts) >= 6 and parts[0] == "pheme-rnr-dataset":
        event, label, thread_id = parts[1], parts[2], parts[3]
    elif len(parts) >= 6 and parts[0] == "all-rnr-annotated-threads":
        event = parts[1].removesuffix("-all-rnr-threads")
        label, thread_id = parts[2], parts[3]
    else:
        return None
    if label not in {"rumours", "non-rumours"}:
        return None
    return event, label, thread_id


def _is_deny_reaction(text: str) -> bool:
    lower = (text or "").lower()
    return any(word in lower for word in DENY_WORDS)


def _doc_from_payload(event: str, label_dir: str, thread_id: str, payload: dict, source_path: str) -> TweetDocument | None:
    text = clean_tweet_text(str(payload.get("text") or ""))
    if not text:
        return None
    doc_id = str(payload.get("id") or Path(source_path).stem)
    label = "rumour" if label_dir == "rumours" else "non-rumour"
    return TweetDocument(
        doc_id=doc_id,
        event=event,
        label=label,
        text=text,
        created_at=str(payload.get("created_at") or ""),
        retweet_count=int(payload.get("retweet_count") or 0),
        favorite_count=int(payload.get("favorite_count") or 0),
        user=dict(payload.get("user") or {}),
        thread_id=str(thread_id),
        reaction_count=0,
        deny_count=0,
        metadata={"source_path": source_path},
    )


def parse_pheme_archive(archive_path: str | Path, max_per_event_label: int | None = None) -> list[TweetDocument]:
    """Parse PHEME source tweets by streaming the compressed archive once."""
    docs_by_key: dict[tuple[str, str, str], TweetDocument] = {}
    selected_keys: set[tuple[str, str, str]] = set()
    seen_per_event_label: Counter[tuple[str, str]] = Counter()
    reaction_counts: Counter[tuple[str, str, str]] = Counter()
    deny_counts: Counter[tuple[str, str, str]] = Counter()

    with tarfile.open(Path(archive_path), "r:*") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            parsed = _path_parts(member.name)
            if not parsed:
                continue
            event, label_dir, thread_id = parsed
            key = (event, label_dir, thread_id)
            label = "rumour" if label_dir == "rumours" else "non-rumour"
            per_key = (event, label)

            if "/source-tweet/" in member.name or "/source-tweets/" in member.name:
                if max_per_event_label is not None and seen_per_event_label[per_key] >= max_per_event_label:
                    continue
                payload = _safe_json_from_member(tf, member)
                doc = _doc_from_payload(event, label_dir, thread_id, payload, member.name)
                if doc is None:
                    continue
                docs_by_key[key] = doc
                selected_keys.add(key)
                seen_per_event_label[per_key] += 1
            elif "/reactions/" in member.name and key in selected_keys:
                reaction_counts[key] += 1
                reaction = _safe_json_from_member(tf, member)
                if _is_deny_reaction(str(reaction.get("text") or "")):
                    deny_counts[key] += 1

    docs: list[TweetDocument] = []
    for key, doc in docs_by_key.items():
        docs.append(
            TweetDocument(
                doc_id=doc.doc_id,
                event=doc.event,
                label=doc.label,
                text=doc.text,
                created_at=doc.created_at,
                retweet_count=doc.retweet_count,
                favorite_count=doc.favorite_count,
                user=doc.user,
                thread_id=doc.thread_id,
                reaction_count=int(reaction_counts[key]),
                deny_count=int(deny_counts[key]),
                metadata=doc.metadata,
            )
        )
    return sorted(docs, key=lambda doc: (doc.event, doc.label, doc.doc_id))


def event_display_name(event: str) -> str:
    return event.replace("-", " ").replace("_", " ").title()


def build_event_queries(events: Iterable[str]) -> dict[str, str]:
    return {
        event: f"Summarize the key facts and latest developments about {event_display_name(event)}."
        for event in sorted(set(events))
    }


def write_processed_dataset(docs: list[TweetDocument], out_dir: str | Path) -> dict[str, object]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "documents.jsonl", (doc.to_dict() for doc in docs))
    queries = build_event_queries(doc.event for doc in docs)
    write_json(out / "queries.json", queries)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"rumour": 0, "non-rumour": 0})
    for doc in docs:
        counts[doc.event][doc.label] += 1
    summary = {"num_docs": len(docs), "events": counts, "queries": queries}
    write_json(out / "summary.json", summary)
    return summary
