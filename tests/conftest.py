from __future__ import annotations

import sys
from pathlib import Path

# Flat top-level modules (constants.py, route_config.py, etc.) live at repo root,
# not in a package — add it to sys.path so `import constants` etc. resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
