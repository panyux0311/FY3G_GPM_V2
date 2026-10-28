"""Run package tests for FY3G_GPM_V2."""
from pathlib import Path
import sys
import unittest

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
