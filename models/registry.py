"""Registry mapping short names → BaseModel classes.

The notebook iterates over this dict to drive the experiment loop.
"""

from __future__ import annotations

from .blr import BLRModel
from .bnn import BNNModel
from .exact_gp import ExactGPModel
from .gbm import GBMModel
from .hierarchical import HierarchicalModel
from .linear import LinearModel
from .rf import RFModel
from .svgp import SVGPModel

MODELS = {
    "linear": LinearModel,
    "blr": BLRModel,
    "hierarchical": HierarchicalModel,
    "exact_gp": ExactGPModel,
    "svgp": SVGPModel,
    "bnn": BNNModel,
    "rf": RFModel,
    "gbm": GBMModel,
}


def build(name: str, **kwargs):
    cls = MODELS[name]
    return cls(**kwargs)
