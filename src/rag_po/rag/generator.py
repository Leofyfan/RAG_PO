from __future__ import annotations

from rag_po.defense.critical_prompt import system_prompt
from rag_po.models import TweetDocument
from rag_po.rag.ecnu_client import ECNUClient
from rag_po.text_utils import compact_for_prompt


def build_context(docs: list[TweetDocument], max_doc_chars: int = 520, include_weights: bool = False) -> str:
    lines: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        user = doc.user or {}
        verified = "verified" if user.get("verified") else "unverified"
        defense_meta = doc.metadata.get("defense", {}) if isinstance(doc.metadata, dict) else {}
        weight_note = ""
        if include_weights and defense_meta:
            trust = defense_meta.get("trust")
            support = defense_meta.get("claim_support")
            cluster_size = defense_meta.get("cluster_size")
            parts = []
            if isinstance(trust, (int, float)):
                parts.append(f"trust={trust:.3f}")
            if isinstance(support, (int, float)):
                parts.append(f"claim_support={support:.3f}")
            if isinstance(cluster_size, int):
                parts.append(f"cluster_size={cluster_size}")
            if parts:
                weight_note = "; " + "; ".join(parts)
        lines.append(
            f"[{idx}] event={doc.event}; user={verified}; "
            f"reactions={doc.reaction_count}; denies={doc.deny_count}{weight_note}; "
            f"text={compact_for_prompt(doc.text, max_doc_chars)}"
        )
    return "\n".join(lines)


def build_summary_messages(
    query: str,
    docs: list[TweetDocument],
    critical: bool = False,
    prompt_mode: str = "critical",
) -> list[dict[str, str]]:
    context = build_context(docs, include_weights=prompt_mode == "tri_state")
    user_prompt = f"""Event query: {query}
Retrieved posts:
{context}

Write a concise factual event summary in English. Include only information supported by the posts."""
    return [
        {"role": "system", "content": system_prompt(critical, prompt_mode=prompt_mode)},
        {"role": "user", "content": user_prompt},
    ]


def generate_summary(
    client: ECNUClient,
    query: str,
    docs: list[TweetDocument],
    critical: bool = False,
    prompt_mode: str = "critical",
    temperature: float = 0.1,
) -> str:
    messages = build_summary_messages(query, docs, critical=critical, prompt_mode=prompt_mode)
    return client.chat(messages, temperature=temperature, cache_namespace="summaries", max_tokens=420)
