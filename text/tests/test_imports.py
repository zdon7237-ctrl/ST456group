from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from novel_continuation import __version__


def test_package_exposes_version():
    assert isinstance(__version__, str)
    assert __version__
