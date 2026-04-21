"""Parse structured (NDJSON) output from AI CLIs into UsageRecord events.

Each of Ralph's supported tools can emit per-assistant-message token usage on
stdout if invoked with the right flag:

    claude --print --output-format=stream-json --verbose
    ccs <profile> --print --output-format=stream-json --verbose (passthrough)
    amp  --execute <prompt> --stream-json
    opencode run --format json <prompt>

This module consumes the raw PTY byte stream, splits on newlines, and dispatches
JSON events through a tool-specific handler that emits two callback streams:

    on_record:  UsageRecord events (for accounting)
    on_display: human-readable text extracted from assistant messages (for console)

The parser is tolerant: partial lines are buffered until the next newline; lines
that don't parse as JSON are passed through to on_display verbatim so pre-amble
banners, spinners, and debug output still reach the user.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ralph.process import Tool
from ralph.usage import UsageRecord

logger = logging.getLogger(__name__)


OnRecord = Callable[[UsageRecord], None]
OnDisplay = Callable[[str], None]


@dataclass
class StreamUsageParser:
    """Feed raw output chunks; emit structured UsageRecord + display text events."""

    tool: Tool
    on_record: OnRecord | None = None
    on_display: OnDisplay | None = None
    session_id: str | None = None
    _buffer: str = field(default="", init=False)
    _saw_structured_event: bool = field(default=False, init=False)

    def feed(self, chunk: str) -> None:
        """Consume a chunk of raw output. Lines without a trailing newline are
        buffered until the next call.
        """
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line)

    def flush(self) -> None:
        """Drain any buffered partial line. Call once after the process exits."""
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""

    @property
    def saw_structured_event(self) -> bool:
        """True if at least one JSON event was successfully parsed. Used by the
        runner to decide whether stream-based usage is authoritative for this
        iteration (vs. falling back to a JSONL backfill scan)."""
        return self._saw_structured_event

    def _handle_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            self._emit_display(line + "\n")
            return

        if not stripped.startswith("{"):
            # Not JSON — pass through (spinners, banners, stderr diagnostics)
            self._emit_display(line + "\n")
            return

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            self._emit_display(line + "\n")
            return

        if not isinstance(data, dict):
            self._emit_display(line + "\n")
            return

        self._saw_structured_event = True

        if self.tool in (Tool.CLAUDE, Tool.CCS, Tool.AMP):
            self._handle_anthropic_event(data)
        elif self.tool == Tool.OPENCODE:
            self._handle_opencode_event(data)

    def _handle_anthropic_event(self, data: dict[str, Any]) -> None:
        """Parse claude/ccs/amp stream-json events (Anthropic shape)."""
        event_type = data.get("type")

        if event_type == "system" and data.get("subtype") == "init":
            sid = data.get("session_id")
            if isinstance(sid, str):
                self.session_id = sid
            return

        if event_type == "assistant":
            msg = data.get("message", {})
            if not isinstance(msg, dict):
                return

            # Extract text blocks for display
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if isinstance(text, str) and text:
                            self._emit_display(text)

            usage = msg.get("usage")
            if isinstance(usage, dict):
                record = self._record_from_anthropic_usage(usage, msg.get("model"))
                if record is not None:
                    self._emit_record(record)
            return

        if event_type == "result":
            # Final summary — per-message usage already captured on assistant
            # events. We intentionally don't double-count here.
            return

        # Tool-use/user events are not surfaced: they're noisy and carry no usage.

    def _handle_opencode_event(self, data: dict[str, Any]) -> None:
        """Parse opencode `run --format json` events (AI-SDK shape)."""
        # AI-SDK usage appears on step-finish-like events under "usage" or
        # within a nested "data" object. Schema evolves; probe common paths.
        usage_dict = data.get("usage")
        if not isinstance(usage_dict, dict):
            nested = data.get("data")
            if isinstance(nested, dict):
                maybe = nested.get("usage")
                if isinstance(maybe, dict):
                    usage_dict = maybe

        if isinstance(usage_dict, dict):
            record = self._record_from_opencode_usage(usage_dict, data)
            if record is not None:
                self._emit_record(record)

        # Display text extraction — events may carry "text" or "content" strings
        for key in ("text", "content", "delta"):
            val = data.get(key)
            if isinstance(val, str) and val:
                self._emit_display(val)
                break

    def _record_from_anthropic_usage(
        self, usage: dict[str, Any], model: Any
    ) -> UsageRecord | None:
        try:
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
            cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        except (ValueError, TypeError):
            logger.debug("Invalid token counts in stream event")
            return None

        if input_tokens == 0 and output_tokens == 0 and cache_creation == 0 and cache_read == 0:
            return None

        model_str = model if isinstance(model, str) else None

        from ralph.pricing import calculate_cost

        cost = calculate_cost(
            model=model_str,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
        )

        return UsageRecord(
            timestamp=datetime.now(timezone.utc),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            model=model_str,
            session_id=self.session_id,
            cost_usd=cost,
        )

    def _record_from_opencode_usage(
        self, usage: dict[str, Any], event: dict[str, Any]
    ) -> UsageRecord | None:
        # AI-SDK canonical keys vary across versions: promptTokens/completionTokens
        # (v3), inputTokens/outputTokens (v4+), or Anthropic-style input_tokens.
        def _pick(*keys: str) -> int:
            for k in keys:
                v = usage.get(k)
                if v is None:
                    continue
                try:
                    return int(v)
                except (ValueError, TypeError):
                    continue
            return 0

        input_tokens = _pick("promptTokens", "inputTokens", "input_tokens")
        output_tokens = _pick("completionTokens", "outputTokens", "output_tokens")
        cache_read = _pick("cachedPromptTokens", "cache_read_input_tokens")
        cache_creation = _pick("cache_creation_input_tokens")

        # Heuristic: some SDK versions count cache_read inside promptTokens.
        if cache_read and input_tokens >= cache_read:
            input_tokens -= cache_read

        if (
            input_tokens == 0
            and output_tokens == 0
            and cache_creation == 0
            and cache_read == 0
        ):
            return None

        model: str | None = None
        for path in (("model",), ("data", "model"), ("providerMetadata", "model")):
            cur: Any = event
            for key in path:
                if isinstance(cur, dict):
                    cur = cur.get(key)
                else:
                    cur = None
                    break
            if isinstance(cur, str):
                model = cur
                break

        from ralph.pricing import calculate_cost

        cost = calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
        )

        return UsageRecord(
            timestamp=datetime.now(timezone.utc),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            model=model,
            session_id=self.session_id,
            cost_usd=cost,
        )

    def _emit_record(self, record: UsageRecord) -> None:
        if self.on_record is not None:
            try:
                self.on_record(record)
            except Exception:
                logger.exception("on_record callback raised")

    def _emit_display(self, text: str) -> None:
        if self.on_display is not None:
            try:
                self.on_display(text)
            except Exception:
                logger.exception("on_display callback raised")


def make_tee_callback(
    parser: StreamUsageParser,
    passthrough: OnDisplay | None = None,
) -> OnDisplay:
    """Build an ``on_output`` handler for ``stream_process`` that feeds the
    parser and optionally tees a passthrough copy of the raw bytes.

    The parser's own ``on_display`` callback is what should reach the user's
    console — this wrapper's ``passthrough`` is for rare debugging cases (e.g.
    dumping raw NDJSON to a log file) and is off by default.
    """

    def _tee(chunk: str) -> None:
        if passthrough is not None:
            passthrough(chunk)
        parser.feed(chunk)

    return _tee
