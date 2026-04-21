"""Tests for the StreamUsageParser — per-tool NDJSON fixtures."""

from __future__ import annotations

import json

import pytest

from ralph.process import Tool
from ralph.usage import UsageRecord
from ralph.usage_stream import StreamUsageParser


def _run(parser: StreamUsageParser, chunks: list[str]) -> tuple[list[UsageRecord], list[str]]:
    records: list[UsageRecord] = []
    display: list[str] = []
    parser.on_record = records.append
    parser.on_display = display.append
    for c in chunks:
        parser.feed(c)
    parser.flush()
    return records, display


class TestClaudeStreamJson:
    """claude --print --output-format=stream-json --verbose shape."""

    def test_extracts_usage_from_assistant_event(self):
        assistant = json.dumps({
            "type": "assistant",
            "message": {
                "model": "claude-sonnet-4-20250514",
                "content": [{"type": "text", "text": "Hello world"}],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 200,
                    "cache_read_input_tokens": 1000,
                },
            },
        })
        parser = StreamUsageParser(tool=Tool.CLAUDE)
        records, display = _run(parser, [assistant + "\n"])

        assert len(records) == 1
        r = records[0]
        assert r.input_tokens == 100
        assert r.output_tokens == 50
        assert r.cache_creation_input_tokens == 200
        assert r.cache_read_input_tokens == 1000
        assert r.model == "claude-sonnet-4-20250514"
        assert r.rate_limited_tokens == 350  # excludes cache_read
        assert "Hello world" in "".join(display)

    def test_captures_session_id_from_system_init(self):
        init = json.dumps({
            "type": "system",
            "subtype": "init",
            "session_id": "sess-abc-123",
        })
        assistant = json.dumps({
            "type": "assistant",
            "message": {"usage": {"input_tokens": 10, "output_tokens": 5}},
        })
        parser = StreamUsageParser(tool=Tool.CLAUDE)
        records, _ = _run(parser, [init + "\n", assistant + "\n"])
        assert records[0].session_id == "sess-abc-123"

    def test_partial_lines_are_buffered(self):
        assistant = json.dumps({
            "type": "assistant",
            "message": {"usage": {"input_tokens": 10, "output_tokens": 5}},
        })
        parser = StreamUsageParser(tool=Tool.CLAUDE)
        # Split the line into two chunks mid-JSON — parser must buffer.
        half = len(assistant) // 2
        records, _ = _run(parser, [assistant[:half], assistant[half:] + "\n"])
        assert len(records) == 1
        assert records[0].input_tokens == 10

    def test_non_json_lines_pass_through_to_display(self):
        parser = StreamUsageParser(tool=Tool.CLAUDE)
        records, display = _run(
            parser, ["some banner text\n", "still loading...\n"]
        )
        assert records == []
        assert "some banner text" in "".join(display)
        assert "still loading" in "".join(display)

    def test_malformed_json_does_not_crash(self):
        parser = StreamUsageParser(tool=Tool.CLAUDE)
        records, display = _run(
            parser, ["{this is not valid json\n", '{"type":"assistant"}\n']
        )
        # Malformed line goes to display; well-formed-but-no-usage is silently dropped.
        assert records == []
        assert "{this is not valid json" in "".join(display)

    def test_zero_token_events_are_skipped(self):
        """An assistant event with zero tokens across the board is not recorded."""
        assistant = json.dumps({
            "type": "assistant",
            "message": {"usage": {
                "input_tokens": 0, "output_tokens": 0,
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            }},
        })
        parser = StreamUsageParser(tool=Tool.CLAUDE)
        records, _ = _run(parser, [assistant + "\n"])
        assert records == []

    def test_result_event_does_not_double_count(self):
        """Per-assistant usage already captured; result event is ignored."""
        assistant = json.dumps({
            "type": "assistant",
            "message": {"usage": {"input_tokens": 100, "output_tokens": 50}},
        })
        result = json.dumps({
            "type": "result",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        })
        parser = StreamUsageParser(tool=Tool.CLAUDE)
        records, _ = _run(parser, [assistant + "\n", result + "\n"])
        assert len(records) == 1


class TestCcsPassthrough:
    """CCS emits the same Anthropic shape when running a Claude account."""

    def test_ccs_uses_anthropic_parser(self):
        assistant = json.dumps({
            "type": "assistant",
            "message": {"usage": {"input_tokens": 42, "output_tokens": 7}},
        })
        parser = StreamUsageParser(tool=Tool.CCS)
        records, _ = _run(parser, [assistant + "\n"])
        assert len(records) == 1
        assert records[0].output_tokens == 7


class TestOpencodeFormatJson:
    """opencode run --format json — AI-SDK usage shape."""

    def test_v3_shape_prompt_completion_tokens(self):
        event = json.dumps({
            "type": "step-finish",
            "usage": {"promptTokens": 500, "completionTokens": 200},
            "model": "gemini-3-pro",
        })
        parser = StreamUsageParser(tool=Tool.OPENCODE)
        records, _ = _run(parser, [event + "\n"])
        assert len(records) == 1
        assert records[0].input_tokens == 500
        assert records[0].output_tokens == 200
        assert records[0].model == "gemini-3-pro"

    def test_v4_shape_input_output_tokens(self):
        event = json.dumps({
            "type": "step-finish",
            "usage": {"inputTokens": 300, "outputTokens": 100},
        })
        parser = StreamUsageParser(tool=Tool.OPENCODE)
        records, _ = _run(parser, [event + "\n"])
        assert records[0].input_tokens == 300
        assert records[0].output_tokens == 100

    def test_nested_data_usage(self):
        event = json.dumps({
            "type": "chunk",
            "data": {
                "usage": {"promptTokens": 50, "completionTokens": 25},
                "model": "gpt-5",
            },
        })
        parser = StreamUsageParser(tool=Tool.OPENCODE)
        records, _ = _run(parser, [event + "\n"])
        assert records[0].input_tokens == 50


class TestStreamSawStructured:
    """saw_structured_event gates 'is the live stream authoritative?'."""

    def test_false_until_a_json_event_parses(self):
        parser = StreamUsageParser(tool=Tool.CLAUDE)
        _run(parser, ["plain text only\n"])
        assert parser.saw_structured_event is False

    def test_true_after_any_json_event(self):
        parser = StreamUsageParser(tool=Tool.CLAUDE)
        _run(parser, ['{"type":"result"}\n'])
        assert parser.saw_structured_event is True
