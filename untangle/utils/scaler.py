"""Loss scaler utilities for mixed-precision training."""

import torch
from torch import Tensor
from torch.optim import Optimizer


class NativeScaler:
    """Native torch gradient scaler for mixed-precision training.

    This class provides a wrapper around torch.amp.GradScaler for handling
    gradient scaling in mixed-precision training.
    """

    state_dict_key = "amp_scaler"

    def __init__(self) -> None:
        self._scaler = torch.amp.GradScaler("cuda")

    def __call__(
        self,
        loss: Tensor,
        optimizer: Optimizer,
        need_update: bool,
    ) -> None:
        """Scales the loss, performs backward pass, and optionally updates.

        This method scales the given loss, performs a backward pass, and
        optionally steps the optimizer and updates the scaler.

        Args:
            loss: The loss tensor to be scaled and backpropagated.
            optimizer: The optimizer to step if an update is needed.
            need_update: If True, the optimizer will be stepped and the
                scaler will be updated.
        """
        self._scaler.scale(loss).backward()
        if need_update:
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self) -> dict:
        """Returns the state dictionary of the underlying GradScaler.

        Returns:
            A dictionary containing the current state of the GradScaler.
        """
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the state dictionary into the underlying GradScaler.

        Args:
            state_dict: A dictionary containing the state to be loaded.
        """
        self._scaler.load_state_dict(state_dict)
