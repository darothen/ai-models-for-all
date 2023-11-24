import os

from ai_models import model

from . import config
from .app import stub

logger = config.get_logger(__name__)


@stub.function(
    image=stub.image,
    secret=config.ENV_SECRETS,
)
def check_assets():
    assets = list(config.AI_MODEL_ASSETS_DIR.glob("**/*"))
    logger.info(f"Found {len(assets)} assets:")
    for i, asset in enumerate(assets, 1):
        logger.info(f"({i}) {asset}")
    logger.info(f"CDS API URL: {os.environ['CDSAPI_URL']}")
    logger.info(f"CDS API Key: {os.environ['CDSAPI_KEY']}")


@stub.function(
    image=stub.image,
    # volumes={VOLUME_ROOT: stub.volume},
    # gpu=DEFAULT_GPU_CONFIG, timeout=600
)
def generate_forecast(model_name: str = config.SUPPORTED_AI_MODELS[0]):
    # logger = config.get_logger()
    logger.info(f"Attempting to initialize model {model_name}...")
    # _ = model.load_model(model_name)
    logger.info("Generating forecast...")
    logger.info("Done!")


@stub.local_entrypoint()
def main():
    check_assets.remote()
    generate_forecast.remote()
