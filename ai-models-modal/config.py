import datetime
import logging
import os
import pathlib

import modal

MAX_FCST_LEAD_TIME = 24 * 10  # 10 days


# Set up a cache for assets leveraged during model runtime.
CACHE_DIR = pathlib.Path("/cache")
# Root dir in cache for writing completed model outputs.
OUTPUT_ROOT_DIR = CACHE_DIR / "output"

# Set up paths that can be mapped to our Volume in order to persist model
# assets after they've been downloaded once.
# TODO: Should we have a separate Volume instance for the model assets?
AI_MODEL_ASSETS_DIR = CACHE_DIR / "assets"


# Set a default GPU that's large enough to work with any of the published models
# available to the ai-models package.
DEFAULT_GPU_CONFIG = modal.gpu.A100(memory=40)


# Read secrets locally from a ".env" file; this avoids the need to have users
# manually set them up in Modal, with the one downside that we do have to put
# all secrets into the same file (but we don't plan to have many).
# NOTE: Modal will try to read ".env" from the working directory, not from our
# module directory. So keep the ".env" in the repo root.
ENV_SECRETS = modal.Secret.from_dotenv()


def validate_env():
    """Validate that expected env vars from .env are imported correctly."""
    assert os.environ.get("CDS_API_KEY", "") != "YOUR_KEY_HERE"
    assert os.environ.get("GCS_SERVICE_ACCOUNT_INFO", "") != "YOUR_SERVICE_ACCOUNT_INFO"
    assert os.environ.get("GCS_BUCKET_NAME", "") != "YOUR_BUCKET_NAME"


def make_output_path(model_name: str, init_datetime: datetime.datetime) -> pathlib.Path:
    """Create a full path for writing a model output GRIB file."""
    filename = f"{model_name}.{init_datetime:%Y%m%d%H%M}.grib"
    return OUTPUT_ROOT_DIR / filename


def get_logger(
    name: str, level: int = logging.INFO, add_handler=False
) -> logging.Logger:
    """Set up a default logger with configs for working within a modal app."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if add_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s")
        )
        logger.addHandler(handler)

    # logger.propagate = False
    return logger


def set_logger_basic_config(level: int = logging.INFO):
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s")
    )
    logging.basicConfig(level=level, handlers=[handler])
