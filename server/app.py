"""
NL2SQL Arena — server/app.py
Required by openenv-core for multi-mode deployment discovery.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa: F401


def main() -> None:
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
