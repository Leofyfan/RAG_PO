from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import requests

from rag_po.io_utils import ensure_parent

ECNU_BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
ECNU_CHAT_MODELS = {"ecnu-plus", "ecnu-max"}
ECNU_EMBEDDING_MODEL = "ecnu-embedding-small"


class ECNUAPIError(RuntimeError):
    pass


class DiskCache:
    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path | None:
        if self.root is None:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / namespace / f"{digest}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        path = self._path(namespace, key)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        if path is None:
            return
        ensure_parent(path)
        tmp = path.with_suffix(f".{threading.get_ident()}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


class ECNUClient:
    """OpenAI-compatible ECNU chat and embedding client with retries and caching."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = ECNU_BASE_URL,
        chat_model: str = "ecnu-plus",
        embedding_model: str = ECNU_EMBEDDING_MODEL,
        timeout: int = 120,
        retries: int = 3,
        cache_dir: str | Path | None = "data/cache/ecnu",
    ):
        self.api_key = api_key or os.getenv("ECNU_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ECNUAPIError("Missing ECNU_API_KEY environment variable.")
        if chat_model not in ECNU_CHAT_MODELS:
            raise ValueError(f"Unsupported ECNU chat model: {chat_model}")
        if embedding_model != ECNU_EMBEDDING_MODEL:
            raise ValueError(f"Unsupported ECNU embedding model: {embedding_model}")
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.retries = retries
        self.cache = DiskCache(cache_dir)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def embedding_payload(self, inputs: list[str]) -> dict[str, object]:
        return {"model": self.embedding_model, "input": inputs}

    def chat_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        model: str | None = None,
        thinking: bool = False,
    ) -> dict[str, object]:
        selected_model = model or self.chat_model
        if selected_model not in ECNU_CHAT_MODELS:
            raise ValueError(f"Unsupported ECNU chat model: {selected_model}")
        return {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }

    def _post_json(self, endpoint: str, payload: dict[str, object]) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(url, headers=self.headers, json=payload, timeout=self.timeout)
                if response.status_code >= 400:
                    raise ECNUAPIError(f"HTTP {response.status_code}: {response.text[:500]}")
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                sleep_s = min(12.0, 1.5 * (2 ** attempt)) + random.random() * 0.3
                time.sleep(sleep_s)
        raise ECNUAPIError(f"ECNU API request failed after retries: {last_error}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        normalized = [text[:8192] for text in texts]
        key = json.dumps({"model": self.embedding_model, "input": normalized}, ensure_ascii=False, sort_keys=True)
        cached = self.cache.get("embeddings", key)
        if cached is not None:
            return cached
        payload = self.embedding_payload(normalized)
        data = self._post_json("embeddings", payload)
        embeddings = [item["embedding"] for item in sorted(data.get("data", []), key=lambda row: row.get("index", 0))]
        if len(embeddings) != len(texts):
            raise ECNUAPIError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
        self.cache.set("embeddings", key, embeddings)
        return embeddings

    def embed_texts(self, texts: list[str], batch_size: int = 32, parallelism: int = 1) -> list[list[float]]:
        if not texts:
            return []
        batches: list[tuple[int, list[str]]] = [
            (start, texts[start : start + batch_size]) for start in range(0, len(texts), batch_size)
        ]
        results: list[list[float] | None] = [None] * len(texts)
        workers = max(1, parallelism)
        if workers == 1:
            for start, batch in batches:
                vectors = self.embed_batch(batch)
                results[start : start + len(batch)] = vectors
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self.embed_batch, batch): (start, batch) for start, batch in batches}
                for future in as_completed(futures):
                    start, batch = futures[future]
                    vectors = future.result()
                    results[start : start + len(batch)] = vectors
        return [vector for vector in results if vector is not None]

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        model: str | None = None,
        thinking: bool = False,
        cache_namespace: str = "chat",
    ) -> str:
        payload = self.chat_payload(messages, temperature=temperature, model=model, thinking=thinking)
        key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        cached = self.cache.get(cache_namespace, key)
        if cached is not None:
            return str(cached)
        data = self._post_json("chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise ECNUAPIError("Chat response had no choices")
        content = choices[0].get("message", {}).get("content")
        if content is None:
            content = ""
        self.cache.set(cache_namespace, key, content)
        return str(content)
