"""Loss scaler utilities for mixed-precision training."""

import torch


class NativeScaler:
    """Native torch gradient scaler for mixed-precision training."""

    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")

    def __call__(
        self,
        loss,
        optimizer,
        need_update,
    ):
        self._scaler.scale(loss).backward()
        if need_update:
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
