"""Implementations of supported uncertainty wrappers."""

__all__ = [
    "BaseCorrectnessPredictionWrapper",
    "BaseLossPredictionWrapper",
    "CEBaselineWrapper",
    "CorrectnessPredictionWrapper",
    "DDUWrapper",
    "DUQWrapper",
    "DeepCorrectnessPredictionWrapper",
    "DeepEnsembleWrapper",
    "DeepLossPredictionWrapper",
    "DirichletWrapper",
    "DistributionalWrapper",
    "EDLWrapper",
    "FastDeepEnsembleWrapper",
    "HETXLWrapper",
    "HetClassNNWrapper",
    "LaplaceWrapper",
    "LossPredictionWrapper",
    "MCDropoutWrapper",
    "MahalanobisWrapper",
    "ModelWrapper",
    "PostNetWrapper",
    "SNGPWrapper",
    "SWAGWrapper",
    "ShallowEnsembleWrapper",
    "SpecialWrapper",
    "TemperatureWrapper",
]

from .ce_baseline_wrapper import CEBaselineWrapper
from .correctness_prediction_wrapper import (
    BaseCorrectnessPredictionWrapper,
    CorrectnessPredictionWrapper,
    DeepCorrectnessPredictionWrapper,
)
from .ddu_wrapper import DDUWrapper
from .deep_ensemble_wrapper import DeepEnsembleWrapper
from .duq_wrapper import DUQWrapper
from .edl_wrapper import EDLWrapper
from .fast_deep_ensemble_wrapper import FastDeepEnsembleWrapper
from .hetclassnn_wrapper import HetClassNNWrapper
from .hetxl_wrapper import HETXLWrapper
from .laplace_wrapper import LaplaceWrapper
from .loss_prediction_wrapper import (
    BaseLossPredictionWrapper,
    DeepLossPredictionWrapper,
    LossPredictionWrapper,
)
from .mahalanobis_wrapper import MahalanobisWrapper
from .mc_dropout_wrapper import MCDropoutWrapper
from .model_wrapper import (
    DirichletWrapper,
    DistributionalWrapper,
    ModelWrapper,
    SpecialWrapper,
)
from .postnet_wrapper import PostNetWrapper
from .shallow_ensemble_wrapper import ShallowEnsembleWrapper
from .sngp_wrapper import SNGPWrapper
from .swag_wrapper import SWAGWrapper
from .temperature_wrapper import TemperatureWrapper
