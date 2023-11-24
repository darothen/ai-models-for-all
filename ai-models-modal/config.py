import logging
import pathlib

import modal

# Define a list of which wrapped AI NWP models we want to make available to the
# user. As of 11/23/23, limit this to PanguWeather for dev purposes.
SUPPORTED_AI_MODELS = [
    "panguweather",
]

# Set up paths that can be mapped to our Volume in order to persist model
# assets after they've been downloaded once.
VOLUME_ROOT = pathlib.Path("/vol/ai-models")
AI_MODEL_ASSETS_DIR = VOLUME_ROOT / "assets"


# Set a default GPU that's large enough to work with any of the published models
# available to the ai-models package.
DEFAULT_GPU_CONFIG = modal.gpu.A100(memory=40)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a default logger with configs for working within a modal app."""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    # logger.propagate = False
    return logger
