"""Shared bootstrap for the runnable scripts: put the project root on sys.path
and silence noisy warnings."""
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

warnings.filterwarnings("ignore")
