"""Pluggable model implementations for forest growth comparison.

Every concrete model subclasses ``base.BaseModel`` and exposes the same
``fit / predict / predict_samples / save / load`` API. The notebook
``04_model_comparison.ipynb`` iterates over them via the ``MODELS`` dict
in ``models.registry``.
"""

from .base import BaseModel
from .registry import MODELS, build

__all__ = ["BaseModel", "MODELS", "build"]
