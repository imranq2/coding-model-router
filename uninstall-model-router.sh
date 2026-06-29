#!/bin/bash
set -u

# uninstall-model-router.sh — tear down the local model routing stack.
#   --purge   also delete downloaded models + pip-uninstall packages

echo "Stopping servers..."
pkill -f "vllm-mlx"  2>/dev/null || true
pkill -f "router.py" 2>/dev/null || true

PLIST="$HOME/Library/LaunchAgents/com.local-ai.plist"
if [ -f "$PLIST" ]; then
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
fi

# Strip the marked block from ~/.zshrc (removes claude-router + the four aliases).
ZRC="$HOME/.zshrc"
if [ -f "$ZRC" ]; then
  awk 'BEGIN{skip=0}
       /# >>> claude model routing >>>/{skip=1}
       skip==0{print}
       /# <<< claude model routing <<</{skip=0}' "$ZRC" > "$ZRC.tmp" && mv "$ZRC.tmp" "$ZRC"
fi

rm -f "$HOME/model-router/router_config.json"

if [ "${1:-}" = "--purge" ]; then
  echo "Purging downloaded models..."
  rm -rf "$HOME/.cache/huggingface/hub"/models--mlx-community--* 2>/dev/null || true
fi

echo "✓ Removed. Open a NEW terminal so claude-router is gone from your shell."
echo "  (Downloaded models kept unless you ran with --purge.)"
# Remove the bundle LAST — this script lives inside it (and so does the venv with all the
# pip packages, so deleting the bundle removes them too). cd out first so the working
# directory isn't the one being deleted.
cd "$HOME" && rm -rf "$HOME/model-router"
