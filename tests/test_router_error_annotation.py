"""Tests for router.py's _annotate_fallback_error — non-dict JSON error bodies."""
from __future__ import annotations

import json

from router import _annotate_fallback_error


def test_dict_error_body_gets_annotated() -> None:
    body = json.dumps({"error": {"message": "boom"}}).encode()
    result = _annotate_fallback_error(body, "some-model")
    parsed = json.loads(result)
    assert "has no configured route" in parsed["error"]["message"]
    assert "boom" in parsed["error"]["message"]


def test_json_array_body_does_not_crash() -> None:
    body = json.dumps([1, 2, 3]).encode()
    result = _annotate_fallback_error(body, "some-model")
    assert b"has no configured route" in result
    assert b"[1, 2, 3]" in result or b"[1,2,3]" in result


def test_json_string_body_does_not_crash() -> None:
    body = json.dumps("error string").encode()
    result = _annotate_fallback_error(body, "some-model")
    assert b"has no configured route" in result


def test_json_number_body_does_not_crash() -> None:
    body = b"42"
    result = _annotate_fallback_error(body, "some-model")
    assert b"has no configured route" in result


def test_json_null_body_does_not_crash() -> None:
    body = b"null"
    result = _annotate_fallback_error(body, "some-model")
    assert b"has no configured route" in result


def test_non_json_body_does_not_crash() -> None:
    body = b"not valid json at all"
    result = _annotate_fallback_error(body, "some-model")
    assert b"has no configured route" in result
    assert b"not valid json at all" in result
