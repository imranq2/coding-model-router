#!/bin/bash
# stop-model-router.sh — stop the local stack (vllm-mlx + router.py).
# Safe to run when nothing is up.
_killed=0
pkill -f "vllm-mlx" 2>/dev/null && _killed=1
[ "$_killed" = 1 ] && echo "✓ vllm-mlx stopped" || echo "vllm-mlx not running"
pkill -f "router.py" 2>/dev/null && echo "✓ router stopped" || echo "router not running"
