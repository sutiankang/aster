"""Non-executable chat formatting and a bounded JSON-schema grammar."""

from __future__ import annotations
from dataclasses import dataclass
import itertools
import json
from pathlib import Path


@dataclass(frozen=True)
class ChatTemplate:
    role_prefix: str = "<|"
    role_suffix: str = "|>\n"
    message_suffix: str = "\n"
    generation_role: str = "assistant"

    def render(self, messages):
        if not isinstance(messages, list) or not messages:
            raise ValueError("Chat needs a nonempty message list")
        output = []
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError(
                    "This chat format supports only explicit role/content text messages"
                )
            if message["role"] not in {"system", "user", "assistant", "tool"} or not isinstance(
                message["content"], str
            ):
                raise ValueError("Unknown role or unsupported multimodal message")
            output.append(
                self.role_prefix
                + message["role"]
                + self.role_suffix
                + message["content"]
                + self.message_suffix
            )
        output.append(self.role_prefix + self.generation_role + self.role_suffix)
        return "".join(output)

    def save_pretrained(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "chat_template.json").write_text(
            json.dumps({"schema_version": 1, **self.__dict__}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def from_pretrained(cls, directory):
        data = json.loads((Path(directory) / "chat_template.json").read_text(encoding="utf-8"))
        if data.pop("schema_version") != 1:
            raise ValueError("Unknown chat template version")
        return cls(**data)


class FiniteJSONGrammar:
    def __init__(self, schema, tokenizer, *, max_variants=1024, max_total_tokens=100000):
        if max_variants < 1 or max_total_tokens < 1:
            raise ValueError("Grammar resource budgets must be positive")
        self.schema = schema
        self._limit = max_variants
        values = self._values(schema)
        self._trie = {}
        total = 0
        for value in values:
            text = json.dumps(
                value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            if hasattr(tokenizer, "eos_token_id"):
                ids = [*ids, tokenizer.eos_token_id]
            total += len(ids)
            if not ids or total > max_total_tokens:
                raise ValueError("Grammar exceeds token expansion budget")
            node = self._trie
            for token in ids:
                node = node.setdefault(int(token), {})
            node[None] = True

    def _bounded(self, values):
        result = []
        for value in values:
            result.append(value)
            if len(result) > self._limit:
                raise ValueError("Grammar has too many finite alternatives")
        if not result:
            raise ValueError("Grammar has no valid values")
        return result

    def _values(self, schema):
        if not isinstance(schema, dict):
            raise ValueError("Schema must be an object")
        allowed = {
            "type",
            "const",
            "enum",
            "properties",
            "required",
            "additionalProperties",
            "items",
            "minItems",
            "maxItems",
            "minimum",
            "maximum",
        }
        if set(schema) - allowed:
            raise ValueError("Unsupported schema keyword")

        def typed(value):
            kind = schema.get("type")
            checks = {
                "string": lambda: isinstance(value, str),
                "integer": lambda: type(value) is int,
                "number": lambda: type(value) in {int, float},
                "boolean": lambda: type(value) is bool,
                "null": lambda: value is None,
                "array": lambda: isinstance(value, list),
                "object": lambda: isinstance(value, dict),
            }
            if kind is not None and (kind not in checks or not checks[kind]()):
                raise ValueError("const/enum member violates its declared JSON type")
            return value

        if "const" in schema:
            if set(schema) - {"const", "type"}:
                raise ValueError("const cannot silently ignore other constraints")
            return [typed(schema["const"])]
        if "enum" in schema:
            if set(schema) - {"enum", "type"} or not isinstance(schema["enum"], list):
                raise ValueError("Invalid finite enum schema")
            return self._bounded(typed(value) for value in schema["enum"])
        kind = schema.get("type")
        if kind == "boolean" and set(schema) == {"type"}:
            return [False, True]
        if kind == "null" and set(schema) == {"type"}:
            return [None]
        if kind == "integer" and set(schema) == {"type", "minimum", "maximum"}:
            low, high = schema["minimum"], schema["maximum"]
            if type(low) is not int or type(high) is not int or high - low + 1 > self._limit:
                raise ValueError("Invalid bounded integer range")
            return self._bounded(range(low, high + 1))
        if kind == "object":
            if (
                set(schema) - {"type", "properties", "required", "additionalProperties"}
                or schema.get("additionalProperties") is not False
            ):
                raise ValueError("Objects need explicit properties and additionalProperties=false")
            properties = schema.get("properties", {})
            required = schema.get("required", list(properties))
            if (
                not isinstance(properties, dict)
                or not isinstance(required, list)
                or set(required) != set(properties)
                or len(required) != len(properties)
            ):
                raise ValueError(
                    "Finite object grammar currently requires every property exactly once"
                )
            keys = sorted(properties)
            alternatives = [self._values(properties[key]) for key in keys]
            if math_product(len(values) for values in alternatives) > self._limit:
                raise ValueError("Object grammar expansion exceeds budget")
            return self._bounded(dict(zip(keys, row)) for row in itertools.product(*alternatives))
        if kind == "array" and set(schema) == {"type", "items", "minItems", "maxItems"}:
            low, high = schema["minItems"], schema["maxItems"]
            if type(low) is not int or type(high) is not int or not 0 <= low <= high <= 16:
                raise ValueError("Array grammar requires small explicit length bounds")
            choices = self._values(schema["items"])
            if sum(len(choices) ** length for length in range(low, high + 1)) > self._limit:
                raise ValueError("Array expansion exceeds budget")
            return self._bounded(
                list(row)
                for length in range(low, high + 1)
                for row in itertools.product(choices, repeat=length)
            )
        raise ValueError(
            "Unsupported/unbounded JSON schema; use finite enum/const/bounded containers"
        )

    def _node(self, prefix):
        node = self._trie
        for token in prefix:
            if token not in node:
                raise ValueError("Token prefix left the grammar")
            node = node[token]
        return node

    def allowed_tokens(self, prefix):
        return tuple(token for token in self._node(prefix) if token is not None)

    def accepting(self, prefix):
        return None in self._node(prefix)


def math_product(values):
    result = 1
    for value in values:
        result *= value
    return result
