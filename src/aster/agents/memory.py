"""Scoped lexical retrieval and deterministic, provenance-preserving context compaction."""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import math
import re

from .events import digest, read_events, canonical_json
from .tools import sanitize


def terms(text):

    return re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text.lower())


@dataclass(frozen=True)
class MemoryItem:
    id: str
    scope_id: str
    text: str
    source: str
    verified: bool


class MemoryStore:
    def __init__(self, event_log, *, max_entries=2048, max_entry_chars=12000):
        if min(max_entries, max_entry_chars) < 1:
            raise ValueError("Memory needs positive resource bounds")
        self.log, self.max_entries, self.max_entry_chars = event_log, max_entries, max_entry_chars
        self._items = {}
        for event in read_events(event_log.path):
            if event["kind"] == "memory.added":
                item = MemoryItem(**event["payload"])
                self._items[item.id] = item
        if len(self._items) > max_entries:
            raise ValueError("Persisted memory exceeds configured capacity")

    def add(self, text, *, scope_id, source, verified=False):
        if (
            not isinstance(text, str)
            or not text
            or len(text) > self.max_entry_chars
            or not scope_id
            or not source
            or type(verified) is not bool
        ):
            raise ValueError("Invalid memory entry")
        data = {"scope_id": scope_id, "text": text, "source": source, "verified": verified}
        identifier = digest(data)
        if identifier in self._items:
            return self._items[identifier]
        if len(self._items) >= self.max_entries:
            raise ValueError("Memory is full; explicit retention/compaction is required")
        item = MemoryItem(identifier, **data)
        self.log.append("memory.added", thread_id=scope_id, payload=item.__dict__)
        self._items[identifier] = item
        return item

    def search(self, query, *, scope_id, limit=5, verified_only=False):
        if not query or not scope_id or not 1 <= limit <= 100:
            raise ValueError("Invalid memory query")
        items = [
            item
            for item in self._items.values()
            if item.scope_id == scope_id and (item.verified or not verified_only)
        ]
        if not items:
            return []
        documents = [Counter(terms(item.text)) for item in items]
        lengths = [sum(document.values()) for document in documents]
        average = sum(lengths) / len(lengths) or 1.0
        query_terms = set(terms(query))
        scored = []

        for item, frequencies, length in zip(items, documents, lengths):
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                document_count = sum(term in document for document in documents)
                idf = math.log(1 + (len(items) - document_count + 0.5) / (document_count + 0.5))
                score += (
                    idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average))
                )
            if score > 0:
                scored.append((score, item))
        return [
            {
                "id": item.id,
                "score": score,
                "source": item.source,
                "verified": item.verified,
                "view": sanitize(item.text),
            }
            for score, item in sorted(scored, key=lambda pair: (-pair[0], pair[1].id))[:limit]
        ]


class ContextCompactor:
    """Preserve system/current-user instructions and remove old call/result pairs
    together, retaining source-bound summaries within the token budget."""

    def __init__(self, *, max_summary_chars=2000):
        if max_summary_chars < 1:
            raise ValueError("Summary capacity must be positive")
        self.max_summary_chars = max_summary_chars

    def compact(self, messages, *, encode, max_tokens):
        selected, removed = list(messages), []
        while len(encode(selected)) > max_tokens and len(selected) > 2:
            count = (
                2
                if len(selected) > 3
                and selected[2].get("role") == "assistant"
                and selected[3].get("role") == "tool"
                else 1
            )
            removed.extend(selected[2 : 2 + count])
            del selected[2 : 2 + count]
        if len(encode(selected)) > max_tokens:
            raise ValueError("System/current user cannot fit within context budget")
        if not removed:
            return selected, {"removed_items": 0, "summary_included": False}
        summary = canonical_json(
            [
                {"role": item.get("role"), "excerpt": canonical_json(item.get("content"))[:200]}
                for item in removed
            ]
        )[: self.max_summary_chars]
        source_digest = digest(removed)
        included = False
        while summary:
            memory = {
                "role": "tool",
                "content": {
                    "trust": "untrusted_compacted_history",
                    "source_digest": source_digest,
                    "extractive_summary": summary,
                },
            }
            candidate = selected[:2] + [memory] + selected[2:]
            if len(encode(candidate)) <= max_tokens:
                selected, included = candidate, True
                break
            summary = summary[: len(summary) // 2]
        return selected, {
            "removed_items": len(removed),
            "summary_included": included,
            "source_digest": source_digest,
        }
