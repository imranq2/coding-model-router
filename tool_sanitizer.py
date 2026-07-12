"""Tool-name sanitization for local (vllm-mlx) routes.

vllm-mlx enforces ^[A-Za-z0-9_-]{1,64}$ on tool names; Claude Code's tool names
(e.g. "mcp__playwright__browser_click") can violate this. Sanitize on the way in;
the caller restores original names in the response stream.
"""
from __future__ import annotations

import hashlib
import re

_VALID_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _sanitize_tool_name(name: str) -> str:
    """Map an arbitrary tool name to one that satisfies ^[A-Za-z0-9_-]{1,64}$."""
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    if len(sanitized) <= 64:
        return sanitized
    # Stable 64-char form: first 55 chars + '_' + 8-char SHA-256 prefix
    h = hashlib.sha256(name.encode()).hexdigest()[:8]
    return sanitized[:55] + "_" + h


def _sanitize_tools(body_json: dict) -> tuple[dict, dict]:
    """Sanitize tool names for vllm-mlx's [A-Za-z0-9_-]{1,64} constraint.

    Returns (modified_body_json, {sanitized_name: original_name}).
    Only entries that needed sanitization appear in the map.
    """
    tools = body_json.get("tools")
    if not tools:
        return body_json, {}

    mapping: dict[str, str] = {}
    new_tools: list = []
    used: set[str] = set()

    for tool in tools:
        original_name = tool.get("name", "")
        if _VALID_TOOL_NAME_RE.match(original_name):
            new_tools.append(tool)
            used.add(original_name)
        else:
            sanitized = _sanitize_tool_name(original_name)
            base, i = sanitized, 1
            while sanitized in used:
                suffix = f"_{i}"
                sanitized = base[: 64 - len(suffix)] + suffix
                i += 1
            used.add(sanitized)
            mapping[sanitized] = original_name
            new_tools.append({**tool, "name": sanitized})

    if not mapping:
        return body_json, {}
    return {**body_json, "tools": new_tools}, mapping
