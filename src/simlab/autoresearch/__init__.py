"""Autoresearch runs controlled improvement loops over a fixed evaluation contract.

Version 1 optimizes only the runtime scenario prompt injected into task instructions.
"""

from simlab.autoresearch.manager import run_autoresearch

__all__ = ["run_autoresearch"]
