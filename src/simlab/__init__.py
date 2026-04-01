"""Simlab CLI — browse tool servers and generate docker-compose environments."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    __version__ = version("simulationlab")
except PackageNotFoundError:
    __version__ = "0.0.0"
