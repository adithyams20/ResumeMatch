# conftest.py — pytest configuration for ResumeMatch tests.
# Adds the project root to sys.path so all imports resolve correctly.

import sys
from pathlib import Path

# Add the project root (parent of tests/) to sys.path
sys.path.insert(0, str(Path(__file__).parent))

