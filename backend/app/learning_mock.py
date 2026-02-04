"""
Mock modules required by training checkpoints.
Provides a setup_learning_mock() that registers expected symbols.
"""

import sys
import types

from .learning_optimizer import Ranger2020


def setup_learning_mock() -> None:
    """Register learning.optimizer and learning.lr_scheduler modules."""
    learning = types.ModuleType("learning")
    learning_opt = types.ModuleType("learning.optimizer")
    learning_sch = types.ModuleType("learning.lr_scheduler")

    learning_opt.Ranger2020 = Ranger2020

    class DummyScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            return None

    learning_sch.GradualWarmupScheduler = DummyScheduler

    sys.modules.setdefault("learning", learning)
    sys.modules.setdefault("learning.optimizer", learning_opt)
    sys.modules.setdefault("learning.lr_scheduler", learning_sch)
