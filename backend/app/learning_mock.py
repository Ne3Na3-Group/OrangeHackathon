"""
Mock learning module to enable loading checkpoints that depend on the 'learning' package.
This allows loading model weights without having the original training codebase.
"""

import sys
import types
import torch.optim


class MockOptimizer(torch.optim.Optimizer):
    """Mock optimizer class for unpickling purposes"""
    def __init__(self, params=None, *args, **kwargs):
        if params is None:
            params = [torch.nn.Parameter(torch.zeros(1))]
        defaults = {'lr': 0.001}
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        pass
    
    def __setstate__(self, state):
        self.__dict__.update(state)


class MockScheduler:
    """Mock scheduler class for unpickling purposes"""
    def __init__(self, *args, **kwargs):
        pass
    
    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def step(self):
        pass


def setup_learning_mock():
    """
    Create and register mock 'learning' package in sys.modules.
    This allows torch.load to unpickle checkpoints that reference
    the learning package without having it installed.
    """
    if 'learning' in sys.modules:
        return  # Already set up
    
    # Create package structure
    learning = types.ModuleType('learning')
    learning.__path__ = []  # Make it a package
    
    # Add submodules
    submodules = ['scheduler', 'optimizer', 'trainer', 'model', 'utils', 'data', 'losses']
    for submod in submodules:
        mod = types.ModuleType(f'learning.{submod}')
        setattr(learning, submod, mod)
        sys.modules[f'learning.{submod}'] = mod
    
    # Add specific classes used in the checkpoint
    learning.optimizer.Ranger2020 = MockOptimizer
    learning.optimizer.AdamW = MockOptimizer
    learning.scheduler.CosineAnnealingLR = MockScheduler
    learning.scheduler.OneCycleLR = MockScheduler
    
    # Register the main module
    sys.modules['learning'] = learning
