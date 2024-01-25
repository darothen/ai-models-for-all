"""Shim for interfacing with ai-models package and related plugins."""

import dataclasses
from importlib.metadata import EntryPoint
from typing import Type

import ai_models
from ai_models import model  # noqa: F401 - needed for type annotations

AIModelType = Type[ai_models.model.Model]


@dataclasses.dataclass(frozen=True)
class AIModelPluginConfig:
    """Configuration information for ai-models plugins.

    Although the ai-models package provides a simple interface (ai_models.model.Model)
    for preparing and running models via plugins, the exact mechanics can differ
    a bit (e.g. the FourCastNet plugin inserts an auxiliary method to bypass directly
    exposing the Model sub-class), we directly map the models to package name and
    entrypoints to the implementation classes.

    Attributes:
        model_name: the name of an AI NWP model as it is exposed through the ai-models
            plugin architecture (e.g. "fourcastnetv2-small" for FourCastNet).
        plugin_package_name: the name of an AI NWP model as it is encoded in the name
            of the plugin which provides it (e.g. "fourcastnetv2" for FourCastNet).
        entrypoint: an EntryPoint which maps directly to the class implementing the
            ai-models.model.Model interface for the given model.
    """

    model_name: str
    plugin_package_name: str
    entry_point: EntryPoint


AI_MODELS_CONFIGS: dict[str, AIModelPluginConfig] = {
    # PanguWeather
    "panguweather": AIModelPluginConfig(
        "panguweather",
        "panguweather",
        EntryPoint(
            name="panguweather",
            group="ai_models.model",
            value="ai_models_panguweather.model:PanguWeather",
        ),
    ),
    # FourCastNet
    "fourcastnetv2-small": AIModelPluginConfig(
        "fourcastnetv2-small",
        "fourcastnetv2",
        EntryPoint(
            name="fourcastnetv2",
            group="ai_models.model",
            value="ai_models_fourcastnetv2.model:FourCastNetv2",
        ),
    ),
    # GraphCast
    "graphcast": AIModelPluginConfig(
        "graphcast",
        "graphcast",
        EntryPoint(
            name="graphcast",
            group="ai_models.model",
            value="ai_models_graphcast.model:GraphcastModel",
        ),
    ),
}

SUPPORTED_AI_MODELS = [
    plugin_config.model_name for plugin_config in AI_MODELS_CONFIGS.values()
]


def get_model_class(model_name: str) -> AIModelType:
    """Get the class initializer for an ai-models plugin."""
    return AI_MODELS_CONFIGS[model_name].entry_point.load()
