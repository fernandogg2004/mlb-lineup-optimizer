import sys
from pathlib import Path

# Ensure project root is importable so `import app.*` and `import src.*` work from tests.
sys.path.insert(0, str(Path(__file__).parent))
