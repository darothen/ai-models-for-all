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


# Read secrets locally from a ".env" file; this avoids the need to have users
# manually set them up in Modal, with the one downside that we do have to put
# all secrets into the same file (but we don't plan to have many).
# NOTE: Modal will try to read ".env" from the working directory, not from our
# module directory. So keep the ".env" in the repo root.
ENV_SECRETS = modal.Secret.from_dotenv()


def get_logger(
    name: str, level: int = logging.INFO, set_all: bool = False
) -> logging.Logger:
    """Set up a default logger with configs for working within a modal app."""

    if set_all:
        all_loggers = [
            logging.getLogger(name) for name in logging.root.manager.loggerDict
        ]
        for logger in all_loggers:
            logger.setLevel(level)

    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)

    # logger.propagate = False
    return logger
