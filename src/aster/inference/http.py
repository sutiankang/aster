"""Local HTTP/1.1 and SSE transport over the native inference engine."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import ipaddress
import json
import math
from urllib.parse import urlsplit

from .engine import OverloadedError
from .sampling import SamplingConfig
from .state import PrefixIdentity
from .structured import FiniteJSONGrammar


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _reject_constant(value):
    raise ValueError("Non-finite JSON numbers are forbidden")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_constant(value)
    return parsed


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


class HTTPServer:
    def __init__(
        self,
        engine,
        *,
        host="127.0.0.1",
        port=0,
        max_connections=128,
        max_body_bytes=1024 * 1024,
        io_timeout=15.0,
    ):
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = host == "localhost"
        if not is_loopback:
            raise ValueError(
                "Public binding requires an unimplemented authenticated gateway contract"
            )
        if not 0 <= port <= 65535 or min(max_connections, max_body_bytes, io_timeout) <= 0:
            raise ValueError("Invalid HTTP server limits")
        self.engine, self.host, self.port = engine, host, port
        self.max_connections, self.max_body_bytes, self.io_timeout = (
            max_connections,
            max_body_bytes,
            io_timeout,
        )
        self._server = None
        self._clients = set()
        self._closing = False

    @property
    def address(self):
        if self._server is None:
            raise RuntimeError("HTTP server has not started")
        host, port, *_ = self._server.sockets[0].getsockname()
        return host, port

    @property
    def url(self):
        host, port = self.address
        return f"http://{'[' + host + ']' if ':' in host else host}:{port}"

    async def start(self):
        await self.engine.start()
        self._server = await asyncio.start_server(
            self._connection, self.host, self.port, limit=65536
        )
        return self

    async def _send(self, writer, status, value):
        payload = _json(value)
        reason = {
            200: "OK",
            400: "Bad Request",
            403: "Forbidden",
            404: "Not Found",
            413: "Payload Too Large",
            429: "Too Many Requests",
            500: "Internal Server Error",
            503: "Service Unavailable",
        }[status]
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
            + payload
        )
        await asyncio.wait_for(writer.drain(), timeout=self.io_timeout)

    async def _parse(self, reader):
        headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), self.io_timeout)
        if len(headers) > 16384:
            raise ValueError("Headers too large")
        lines = headers.decode("ascii").split("\r\n")
        method, path, protocol = lines[0].split(" ")
        if protocol != "HTTP/1.1" or not path.startswith("/"):
            raise ValueError("Unsupported request format")
        fields = {}
        for line in lines[1:]:
            if not line:
                continue
            name, value = line.split(":", 1)
            name = name.lower().strip()
            if name in fields:
                raise ValueError("Duplicate HTTP header")
            fields[name] = value.strip()
        if "transfer-encoding" in fields:
            raise ValueError("Chunked request bodies are not supported")
        if "origin" in fields:
            raise PermissionError("Browser origins are not an authorized local client")
        authority = urlsplit("http://" + fields.get("host", ""))
        try:
            valid_host = ipaddress.ip_address(authority.hostname or "").is_loopback
        except ValueError:
            valid_host = authority.hostname == "localhost"
        if not valid_host or authority.username or authority.password or authority.path:
            raise PermissionError("Host must identify a loopback endpoint")
        length = int(fields.get("content-length", "0"))
        if length < 0 or length > self.max_body_bytes:
            raise OverflowError("Body exceeds limit")
        body = await asyncio.wait_for(reader.readexactly(length), self.io_timeout)
        data = (
            json.loads(
                body,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
                object_pairs_hook=_unique_pairs,
            )
            if body
            else {}
        )
        if not isinstance(data, dict):
            raise ValueError("JSON object required")
        return method, path, data

    def _request_options(self, data, *, chat=False):
        runner = self.engine.runner
        allowed = {
            "model",
            "prompt",
            "prompt_token_ids",
            "max_tokens",
            "temperature",
            "top_k",
            "top_p",
            "seed",
            "stream",
            "timeout_s",
            "request_id",
            "eos_token_ids",
            "min_tokens",
            "repetition_penalty",
            "logit_bias",
            "response_format",
            "messages",
        }
        if set(data) - allowed:
            raise ValueError("Unsupported completion fields")
        model_name = data.get("model", runner.policy_artifact_id)
        if hasattr(runner, "resolve_model_identity"):
            identity = runner.resolve_model_identity(model_name)
        else:
            if model_name != runner.policy_artifact_id:
                raise ValueError("Unknown model artifact")
            identity = PrefixIdentity(runner.policy_artifact_id)
        if chat:
            if (
                runner.chat_template is None
                or runner.tokenizer is None
                or "prompt" in data
                or "prompt_token_ids" in data
            ):
                raise ValueError(
                    "Chat requires an explicit artifact template/tokenizer and messages only"
                )
            ids = runner.tokenizer.encode(runner.chat_template.render(data.get("messages")))
        elif "messages" in data:
            raise ValueError("messages require the chat endpoint")
        elif "prompt" in data and "prompt_token_ids" in data:
            raise ValueError("Choose prompt text or prompt token IDs")
        elif "prompt_token_ids" in data:
            ids = data["prompt_token_ids"]
        else:
            if not isinstance(data.get("prompt"), str) or runner.tokenizer is None:
                raise ValueError("Text input requires the artifact tokenizer")
            ids = runner.tokenizer.encode(data["prompt"])
        if not isinstance(ids, (list, tuple)):
            raise ValueError("Prompt IDs must be an array")
        if type(data.get("stream", False)) is not bool:
            raise ValueError("stream must be boolean")
        bias = data.get("logit_bias", {})
        if not isinstance(bias, dict):
            raise ValueError("logit_bias must be an object")
        config = SamplingConfig(
            max_new_tokens=data.get("max_tokens", 32),
            temperature=data.get("temperature", 1.0),
            top_k=data.get("top_k", 0),
            top_p=data.get("top_p", 1.0),
            seed=data.get("seed", 0),
            eos_token_ids=tuple(data.get("eos_token_ids", ())),
            min_new_tokens=data.get("min_tokens", 0),
            repetition_penalty=data.get("repetition_penalty", 1.0),
            logit_bias=tuple((int(token), value) for token, value in bias.items()),
        )
        grammar = None
        if "response_format" in data:
            form = data["response_format"]
            if (
                not isinstance(form, dict)
                or set(form) != {"type", "json_schema"}
                or form["type"] != "json_schema"
                or runner.tokenizer is None
            ):
                raise ValueError("Only finite json_schema response format is supported")
            declaration = form["json_schema"]
            if (
                not isinstance(declaration, dict)
                or set(declaration) - {"name", "schema", "strict"}
                or declaration.get("strict", True) is not True
                or "schema" not in declaration
            ):
                raise ValueError("JSON schema format must be strict")
            grammar = FiniteJSONGrammar(declaration["schema"], runner.tokenizer)
        return ids, config, grammar, identity

    @staticmethod
    def _response(result, chat=False):
        choice = {"index": 0, "finish_reason": result.stop_reason}
        choice.update(
            {"message": {"role": "assistant", "content": result.text}}
            if chat
            else {"text": result.text}
        )
        return {
            "id": result.request_id,
            "object": "chat.completion" if chat else "text_completion",
            "model": result.policy_artifact_id,
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(result.prompt_token_ids),
                "completion_tokens": len(result.token_ids),
                "total_tokens": len(result.prompt_token_ids) + len(result.token_ids),
            },
            "aster": {
                "token_ids": result.token_ids,
                "raw_model_logprobs": result.raw_model_logprobs,
                "behavior_logprobs": result.behavior_logprobs,
                "sampling_transform_order": result.sampling_transform_order,
                "metrics": result.metrics(),
                "error_code": result.error_code,
                "adapter_id": result.adapter_id,
            },
        }

    async def _stream(self, writer, handle, *, chat=False):
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\nX-Accel-Buffering: no\r\n\r\n"
        )
        await asyncio.wait_for(writer.drain(), self.io_timeout)
        async for event in handle:
            choice = {"index": 0, "finish_reason": None}
            choice.update(
                {"delta": {"role": "assistant", "content": event.text}}
                if chat
                else {"text": event.text}
            )
            data = {
                "id": event.request_id,
                "object": "chat.completion.chunk" if chat else "text_completion.chunk",
                "model": event.policy_artifact_id,
                "choices": [choice],
                "aster": {
                    "index": event.index,
                    "token_id": event.token_id,
                    "raw_model_logprob": event.raw_model_logprob,
                    "behavior_logprob": event.behavior_logprob,
                },
            }
            writer.write(b"data: " + _json(data) + b"\n\n")
            await asyncio.wait_for(writer.drain(), self.io_timeout)
        result = await handle.result()
        terminal = self._response(result, chat)
        terminal["object"] = "chat.completion.chunk" if chat else "text_completion.chunk"
        terminal["choices"] = [
            {
                "index": 0,
                "finish_reason": result.stop_reason,
                **({"delta": {}} if chat else {"text": ""}),
            }
        ]
        writer.write(b"data: " + _json(terminal) + b"\n\ndata: [DONE]\n\n")
        await asyncio.wait_for(writer.drain(), self.io_timeout)

    async def _connection(self, reader, writer):
        task = asyncio.current_task()
        admitted = len(self._clients) < self.max_connections and not self._closing
        self._clients.add(task)
        handle = None
        child_tasks = []
        headers_sent = False
        try:
            if not admitted:
                await self._send(writer, 503, {"error": {"code": "connection_limit"}})
                return
            method, path, data = await self._parse(reader)
            if method == "GET" and path in {"/health", "/ready"}:
                ready = self.engine.ready
                await self._send(
                    writer, 200 if ready else 503, {"status": "ready" if ready else "unavailable"}
                )
            elif method == "GET" and path == "/metrics":
                await self._send(writer, 200, self.engine.observation())
            elif method == "GET" and path == "/v1/models":
                await self._send(
                    writer,
                    200,
                    {"object": "list", "data": [{"id": self.engine.runner.policy_artifact_id}]},
                )
            elif method == "POST" and path.startswith("/v1/requests/") and path.endswith("/cancel"):
                identifier = path[len("/v1/requests/") : -len("/cancel")]
                await self._send(writer, 200, {"cancel_requested": self.engine.cancel(identifier)})
            elif method == "POST" and path in {"/v1/completions", "/v1/chat/completions"}:
                chat = path == "/v1/chat/completions"
                ids, config, grammar, identity = self._request_options(data, chat=chat)
                handle = await self.engine.submit(
                    ids,
                    config,
                    request_id=data.get("request_id"),
                    timeout_s=data.get("timeout_s", 60.0),
                    grammar=grammar,
                    identity=identity,
                )

                disconnected = asyncio.create_task(reader.read(1))
                child_tasks.append(disconnected)
                if data.get("stream", False):
                    headers_sent = True
                    response = asyncio.create_task(self._stream(writer, handle, chat=chat))
                else:

                    async def complete():

                        async for _ in handle:
                            pass
                        await self._send(writer, 200, self._response(await handle.result(), chat))

                    response = asyncio.create_task(complete())
                child_tasks.append(response)
                done, _ = await asyncio.wait(child_tasks, return_when=asyncio.FIRST_COMPLETED)
                if response in done:
                    await response
                else:
                    await handle.cancel()
            else:
                await self._send(writer, 404, {"error": {"code": "unsupported_route"}})
        except OverloadedError:
            await self._send(writer, 429, {"error": {"code": "admission_overload"}})
        except PermissionError:
            await self._send(writer, 403, {"error": {"code": "local_client_required"}})
        except OverflowError:
            if not headers_sent:
                await self._send(writer, 413, {"error": {"code": "payload_too_large"}})
        except (
            ValueError,
            TypeError,
            UnicodeError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ):
            if not headers_sent:
                await self._send(writer, 400, {"error": {"code": "invalid_request"}})
        except (ConnectionError, asyncio.TimeoutError):
            pass
        except Exception:
            if not headers_sent:
                await self._send(writer, 500, {"error": {"code": "internal_error"}})
        finally:
            for child in child_tasks:
                if not child.done():
                    child.cancel()
            if child_tasks:
                await asyncio.gather(*child_tasks, return_exceptions=True)
            if handle is not None:
                await handle.cancel()
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            self._clients.discard(task)

    async def close(self):
        self._closing = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        await self.engine.close()

        for client in tuple(self._clients):
            client.cancel()
        if self._clients:
            await asyncio.gather(*tuple(self._clients), return_exceptions=True)

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *exc):
        await self.close()
