"""Enable ``python -m src.pipeline ...``."""

from __future__ import annotations

import sys

from src.pipeline.runner import main

if __name__ == "__main__":
    sys.exit(main())
