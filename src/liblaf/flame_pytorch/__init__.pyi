from . import upstream
from ._version import __version__, __version_tuple__
from .config import FlameConfig
from .flame import FLAME

__all__ = ["FLAME", "FlameConfig", "__version__", "__version_tuple__", "upstream"]
