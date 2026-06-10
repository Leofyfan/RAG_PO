from __future__ import annotations

from rag_po.defense.critical_prompt import system_prompt
from rag_po.models import TweetDocument
from rag_po.rag.ecnu_client import ECNUClient
from rag_po.text_utils import compact_for_prompt


def build_context(docs: list[TweetDocument], max_doc_chars: int = 520) -> str:
    lines: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        label = "RUMOUR" if doc.is_rumour else "NON_RUMOUR"
        user = doc.user or {}
        verified = "verified" if user.get("verified") else "unverified"
        lines.append(
            f"[{idx}] label={label}; event={doc.event}; user={verified}; "
            f"reactions={doc.reaction_count}; denies={doc.deny_count}; text={compact_for_prompt(doc.text, max_doc_chars)}"
        )
    return "\n".join(lines)


def build_summary_messages(query: str, docs: list[TweetDocument], critical: bool = False) -> list[dict[str, str]]:
    context = build_context(docs)
    user_prompt = f"""Event query: {query}
Retrieved posts:
{context}

Write a concise factual event summary in English. Include only information supported by the posts."""
    return [
        {"role": "system", "content": system_prompt(critical)},
        {"role": "user", "content": user_prompt},
    ]


def generate_summary(
    client: ECNUClient,
    query: str,
    docs: list[TweetDocument],
    critical: bool = False,
    temperature: float = 0.1,
) -> str:
    messages = build_summary_messages(query, docs, critical=critical)
    return client.chat(messages, temperature=temperature, cache_namespace="summaries")
