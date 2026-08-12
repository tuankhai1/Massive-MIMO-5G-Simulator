"""Experiment: handover/outage probability versus user speed."""

from __future__ import annotations

import numpy as np

from config import SystemConfig
from mobility import simulate_mobility


def run(cfg: SystemConfig) -> dict:
    """Run mobility + handover simulation and return per-speed metrics."""
    return simulate_mobility(cfg)
