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

# Set up paths to archive initial conditions that are prepared for our model runs;
# for now, this is just the processed GFS/GDAS initial conditions that we produce.
INIT_CONDITIONS_DIR = CACHE_DIR / "initial_conditions"

# Set a default GPU that's large enough to work with any of the published models
# available to the ai-models package.
DEFAULT_GPU_CONFIG = modal.gpu.A100(memory=40)

# Set a default date to use when fetching sample data from ERA-5 to create templates
# for processing GFS/GDAS data; we need this because we have to sort GRIB messages by
# time when we prepare GraphCast inputs.
DEFAULT_GFS_TEMPLATE_MODEL_EPOCH = datetime.datetime(2024, 1, 1, 0, 0)

# Read secrets locally from a ".env" file; this avoids the need to have users
# manually set them up in Modal, with the one downside that we do have to put
# all secrets into the same file (but we don't plan to have many).
# NOTE: Modal will try to read ".env" from the working directory, not from our
# module directory. So keep the ".env" in the repo root.
ENV_SECRETS = modal.Secret.from_dotenv()

# Manually set all "forced" actions to run (e.g. re-processing initial conditions)
FORCE_OVERRIDE = True


def validate_env():
    """Validate that expected env vars from .env are imported correctly."""
    assert os.environ.get("CDS_API_KEY", "") != "YOUR_KEY_HERE"
    assert os.environ.get("GCS_SERVICE_ACCOUNT_INFO", "") != "YOUR_SERVICE_ACCOUNT_INFO"
    assert os.environ.get("GCS_BUCKET_NAME", "") != "YOUR_BUCKET_NAME"


def make_output_path(
    model_name: str, init_datetime: datetime.datetime, use_gfs: bool
) -> pathlib.Path:
    """Create a full path for writing a model output GRIB file."""
    src = "gfs" if use_gfs else "era5"
    filename = f"{model_name}.{src}.{init_datetime:%Y%m%d%H%M}.grib"
    return OUTPUT_ROOT_DIR / filename


def make_gfs_template_path(model_name: str) -> pathlib.Path:
    """Create a expected path where GFS/GDAS -> ERA-5 template should exist."""
    return AI_MODEL_ASSETS_DIR / f"{model_name}.input-template.grib2"


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
