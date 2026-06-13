from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class TweetDocument:
    """A source tweet or generated poison document used by the RAG pipeline."""

    doc_id: str
    event: str
    label: str
    text: str
    created_at: str = ""
    retweet_count: int = 0
    favorite_count: int = 0
    user: dict[str, Any] = field(default_factory=dict)
    thread_id: str = ""
    reaction_count: int = 0
    deny_count: int = 0
    embedding: list[float] | None = None
    attack: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_rumour(self) -> bool:
        return self.label == "rumour" or self.attack is not None

    def with_embedding(self, embedding: list[float]) -> "TweetDocument":
        return replace(self, embedding=embedding)

    def with_attack(self, attack: str) -> "TweetDocument":
        return replace(self, attack=attack, label="rumour")

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "event": self.event,
            "label": self.label,
            "text": self.text,
            "created_at": self.created_at,
            "retweet_count": self.retweet_count,
            "favorite_count": self.favorite_count,
            "user": self.user,
            "thread_id": self.thread_id,
            "reaction_count": self.reaction_count,
            "deny_count": self.deny_count,
            "embedding": self.embedding,
            "attack": self.attack,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TweetDocument":
        return cls(
            doc_id=str(data.get("doc_id") or data.get("id") or ""),
            event=str(data.get("event") or ""),
            label=str(data.get("label") or "non-rumour"),
            text=str(data.get("text") or ""),
            created_at=str(data.get("created_at") or ""),
            retweet_count=int(data.get("retweet_count") or 0),
            favorite_count=int(data.get("favorite_count") or 0),
            user=dict(data.get("user") or {}),
            thread_id=str(data.get("thread_id") or data.get("doc_id") or data.get("id") or ""),
            reaction_count=int(data.get("reaction_count") or 0),
            deny_count=int(data.get("deny_count") or 0),
            embedding=data.get("embedding"),
            attack=data.get("attack"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SearchResult:
    doc: TweetDocument
    score: float
    rank: int


@dataclass(frozen=True)
class DefenseSpec:
    name: str
    outlier: bool = False
    consistency: bool = False
    social: bool = False
    critical_prompt: bool = False
    mmr: bool = False
    dedup: bool = False
    trust_weighted: bool = False
    prompt_mode: str = "critical"


DEFENSE_SPECS: dict[str, DefenseSpec] = {
    "D0": DefenseSpec("D0"),
    "D1": DefenseSpec("D1", outlier=True),
    "D2": DefenseSpec("D2", consistency=True),
    "D3": DefenseSpec("D3", social=True),
    "D4": DefenseSpec("D4", critical_prompt=True),
    "D123": DefenseSpec("D123", outlier=True, consistency=True, social=True),
    "D34": DefenseSpec("D34", social=True, critical_prompt=True),
    "D234": DefenseSpec("D234", consistency=True, social=True, critical_prompt=True),
    "D_all": DefenseSpec("D_all", outlier=True, consistency=True, social=True, critical_prompt=True),
    "D_star": DefenseSpec(
        "D_star",
        critical_prompt=True,
        mmr=True,
        dedup=True,
        trust_weighted=True,
        prompt_mode="tri_state",
    ),
}
