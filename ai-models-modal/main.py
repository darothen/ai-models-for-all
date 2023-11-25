import os
import pathlib

from ai_models import model

from . import config
from .app import stub

logger = config.get_logger(__name__, set_all=True)


@stub.function(
    image=stub.image,
    secret=config.ENV_SECRETS,
)
def check_assets():
    import cdsapi

    assets = list(config.AI_MODEL_ASSETS_DIR.glob("**/*"))
    logger.info(f"Found {len(assets)} assets:")
    for i, asset in enumerate(assets, 1):
        logger.info(f"({i}) {asset}")
    logger.info(f"CDS API URL: {os.environ['CDSAPI_URL']}")
    logger.info(f"CDS API Key: {os.environ['CDSAPI_KEY']}")

    client = cdsapi.Client()
    logger.info(client)

    test_cdsapirc = pathlib.Path("~/.cdsapirc").expanduser()
    logger.info(f"Test .cdsapirc: {test_cdsapirc} exists = {test_cdsapirc.exists()}")

    logger.info("Trying to import eccodes...")
    import eccodes


@stub.function(
    image=stub.image,
    secret=config.ENV_SECRETS,
    # volumes={VOLUME_ROOT: stub.volume},
    # gpu=DEFAULT_GPU_CONFIG, timeout=600
)
def generate_forecast(model_name: str = config.SUPPORTED_AI_MODELS[0]):
    logger.info(f"Attempting to initialize model {model_name}...")
    init_model = model.load_model(
        # Necessary arguments to instantiate a Model object
        input="cds",
        output="none",
        download_assets=False,
        name=model_name,
        # Additional arguments. These are generally set as object attributes
        # which are then referred to by various Model methods; unfortunately,
        # they're not clearly declared in the class documentation so there is
        # a bit of trial and error involved in figuring out what's needed.
        assets=config.AI_MODEL_ASSETS_DIR,
        date=20230701,
        time=0000,
        lead_time=12,
        # Unused arguments that are required by Model class methods to work.
        model_args={},
        assets_sub_directory=None,
        staging_dates=None,
        only_gpu=False,  # TODO: change this to True once we run on GPUs
    )
    logger.info("Generating forecast...")
    init_model.run()
    logger.info("Done!")


@stub.local_entrypoint()
def main():
    check_assets.remote()
    generate_forecast.remote()
