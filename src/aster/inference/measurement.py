"""SSE client-side TTFT and inter-token latency measured with one monotonic clock."""

from __future__ import annotations
import asyncio
from dataclasses import dataclass
import ipaddress
import json
import math
import time
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ClientObservation:
    request_index: int
    started_at: float
    finished_at: float
    token_arrivals: tuple[float, ...]
    status: str
    expected_tokens: int
    output_tokens: int
    http_status: int | None

    @property
    def ttft(self):
        return self.token_arrivals[0] - self.started_at if self.token_arrivals else None

    @property
    def itl(self):
        return tuple(b - a for a, b in zip(self.token_arrivals, self.token_arrivals[1:]))


def percentile(values, probability):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


async def measure_http(
    url,
    prompts,
    *,
    max_new_tokens=16,
    concurrency=1,
    timeout_seconds=60.0,
    ttft_slo=None,
    itl_slo=None,
):
    location = urlsplit(url)
    host = location.hostname
    if (
        location.scheme != "http"
        or not host
        or location.username
        or location.password
        or location.path not in {"", "/"}
    ):
        raise ValueError("Measure a local HTTP root without credentials")
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host == "localhost"
    if not local or min(concurrency, max_new_tokens, timeout_seconds) <= 0 or not prompts:
        raise ValueError("Only explicit bounded local measurements are supported")
    if any(
        value is not None and (not math.isfinite(value) or value <= 0)
        for value in (ttft_slo, itl_slo)
    ):
        raise ValueError("SLO limits must be finite and positive")
    semaphore = asyncio.Semaphore(concurrency)

    async def single(index, prompt):
        async with semaphore:
            started, arrivals, status, code, output_tokens = time.monotonic(), [], "error", None, 0
            writer = None
            try:
                async with asyncio.timeout(timeout_seconds):
                    reader, writer = await asyncio.open_connection(host, location.port or 80)
                    body = json.dumps(
                        {
                            "prompt_token_ids": prompt,
                            "max_tokens": max_new_tokens,
                            "temperature": 0.0,
                            "stream": True,
                        }
                    ).encode()
                    writer.write(
                        f"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                        + body
                    )
                    await writer.drain()
                    headers = await reader.readuntil(b"\r\n\r\n")
                    code = int(headers.split(b" ", 2)[1])
                    if code == 200:
                        final = None
                        done = False
                        while line := await reader.readline():
                            if not line.startswith(b"data: "):
                                continue
                            payload = line[6:].strip()
                            if payload == b"[DONE]":
                                done = True
                                break
                            event = json.loads(payload)
                            if "token_id" in event.get("aster", {}):
                                arrivals.append(time.monotonic())
                            if "usage" in event:
                                final = event
                        output_tokens = len(arrivals)
                        if (
                            final is not None
                            and done
                            and final["usage"]["completion_tokens"] == output_tokens
                        ):
                            reason = final["choices"][0]["finish_reason"]
                            status = (
                                "ok"
                                if reason == "length" and output_tokens == max_new_tokens
                                else "incomplete"
                            )
                    else:
                        status = "http_error"
            except asyncio.TimeoutError:
                status = "timeout"
            except (ValueError, ConnectionError, asyncio.IncompleteReadError):
                status = "protocol_error"
            finally:
                if writer is not None:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except ConnectionError:
                        pass
            return ClientObservation(
                index,
                started,
                time.monotonic(),
                tuple(arrivals),
                status,
                max_new_tokens,
                output_tokens,
                code,
            )

    window_start = time.monotonic()
    records = await asyncio.gather(*(single(index, prompt) for index, prompt in enumerate(prompts)))
    window = time.monotonic() - window_start
    successes = [record for record in records if record.status == "ok"]
    good = [
        record
        for record in successes
        if (ttft_slo is None or record.ttft is not None and record.ttft <= ttft_slo)
        and (itl_slo is None or all(value <= itl_slo for value in record.itl))
    ]
    ttfts = [record.ttft for record in records if record.ttft is not None]
    latencies = [record.finished_at - record.started_at for record in records]
    return {
        "evidence_kind": "local_performance_observation",
        "clock": "client_monotonic_token_event",
        "measurement_window_seconds": window,
        "planned_requests": len(prompts),
        "actual_requests": len(records),
        "successful_requests": len(successes),
        "failed_requests": len(records) - len(successes),
        "good_requests": len(good),
        "throughput_tokens_per_second": sum(r.output_tokens for r in successes) / window,
        "goodput_requests_per_second": len(good) / window,
        "ttft_seconds": {str(p): percentile(ttfts, p) for p in (0.5, 0.95, 0.99)},
        "end_to_end_seconds": {str(p): percentile(latencies, p) for p in (0.5, 0.95, 0.99)},
        "records": [
            {**record.__dict__, "ttft_seconds": record.ttft, "itl_seconds": record.itl}
            for record in records
        ],
    }
