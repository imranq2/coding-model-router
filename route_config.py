"""Route loading and lookup.

Routes are keyed by `claude_model` (the model name Claude Code sends) for O(1)
exact-match lookup. A route may additionally carry a `claude_model_pattern` regex,
checked only on an exact-match miss, so a single route keeps matching future
Claude Code model-id version bumps without a models.json edit + reinstall.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

log = logging.getLogger("model-router")

CONFIG_PATH = Path(
    os.environ.get("ROUTER_CONFIG", Path.home() / "model-router" / "router_config.json")
)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def _build_routes(config: dict) -> tuple[dict[str, dict], list[tuple[re.Pattern, dict]]]:
    """Build the exact-match route dict and the ordered pattern fallback list."""
    routes: dict[str, dict] = {}
    patterns: list[tuple[re.Pattern, dict]] = []
    for route in config.get("routes", []):
        key = route["claude_model"]
        if key in routes:
            log.warning("[model-router] duplicate route for model '%s' — later entry wins", key)
        routes[key] = route
        if pattern := route.get("claude_model_pattern"):
            patterns.append((re.compile(pattern), route))
    return routes, patterns


try:
    CONFIG: dict = load_config()
except FileNotFoundError:
    log.error(
        "[model-router] config not found at %s — run install-model-router first; starting with no routes",
        CONFIG_PATH,
    )
    CONFIG = {"routes": []}

ROUTES: dict[str, dict]
PATTERNS: list[tuple[re.Pattern, dict]]
ROUTES, PATTERNS = _build_routes(CONFIG)


def find_route(model: str) -> dict | None:
    """Exact match first (fast path), then the first matching claude_model_pattern."""
    if route := ROUTES.get(model):
        return route
    for pattern, route in PATTERNS:
        if pattern.search(model):
            return route
    return None


def _reload_routes() -> dict[str, dict]:
    """Reload routes from disk and return the updated exact-match routes dict."""
    global CONFIG, ROUTES, PATTERNS
    CONFIG = load_config()
    ROUTES, PATTERNS = _build_routes(CONFIG)
    return ROUTES
