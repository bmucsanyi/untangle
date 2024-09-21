"""Checkpoint saver.

Tracks top-n training checkpoints.
"""

import logging
import operator

import torch

logger = logging.getLogger(__name__)


class CheckpointSaver:
    """Checkpoint saver that tracks top-n training checkpoints."""

    def __init__(
        self,
        model,
        optimizer,
        loss_scaler,
        max_history,
        checkpoint_dir,
    ):
        # Objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.loss_scaler = loss_scaler

        # State
        # (filename, metric) tuples in order of decreasing performance
        self.checkpoint_files = []
        self.best_epoch = None
        self.best_metric = None

        # Config
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = "checkpoint"
        self.extension = "pt"
        self.cmp = operator.gt
        self.max_history = max_history

    def save_checkpoint(self, epoch, metric):
        # Save as last checkpoint
        last_filename = f"{self.checkpoint_prefix}_last.{self.extension}"
        last_save_path = self.checkpoint_dir / last_filename
        self._save(last_save_path, epoch, metric)

        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None

        if len(self.checkpoint_files) < self.max_history or self.cmp(
            metric, worst_file[1]
        ):
            # Save as among-the-best checkpoint
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)

            top_filename = f"{self.checkpoint_prefix}_{epoch}.{self.extension}"
            top_save_path = self.checkpoint_dir / top_filename
            top_save_path.hardlink_to(last_save_path)
            self.checkpoint_files.append((top_save_path, metric))
            self.checkpoint_files.sort(key=operator.itemgetter(1), reverse=True)

            checkpoints_str = "Current checkpoints:\n"
            for checkpoint_file in self.checkpoint_files:
                checkpoints_str += f"\t{checkpoint_file}\n"
            logger.info(checkpoints_str)

            if self.best_metric is None or self.cmp(metric, self.best_metric):
                self.best_epoch = epoch
                self.best_metric = metric
                best_filename = f"{self.checkpoint_prefix}_best.{self.extension}"
                best_save_path = self.checkpoint_dir / best_filename
                best_save_path.unlink(missing_ok=True)
                best_save_path.hardlink_to(top_save_path)

    def _save(self, save_path, epoch, metric):
        save_state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metric": metric,
        }

        if self.loss_scaler is not None:
            save_state["amp_scaler"] = self.loss_scaler.state_dict()

        torch.save(save_state, save_path)

    def _cleanup_checkpoints(self, num_checkpoints_to_delete):
        start_delete_index = self.max_history - num_checkpoints_to_delete

        if start_delete_index < 0 or len(self.checkpoint_files) <= start_delete_index:
            return

        checkpoints_to_delete = self.checkpoint_files[start_delete_index:]

        for checkpoint in checkpoints_to_delete:
            logger.info(f"Cleaning checkpoint: {checkpoint}")
            checkpoint[0].unlink()

        self.checkpoint_files = self.checkpoint_files[:start_delete_index]
