import sys
from pathlib import Path

# Add project root and waste-api to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
API_DIR = BASE_DIR / "waste-api"

for path in [str(API_DIR), str(BASE_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from app.main import app
