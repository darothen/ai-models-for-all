import datetime
import os
import pathlib

from ai_models import model

from . import config
from .app import stub

config.set_logger_basic_config()
logger = config.get_logger(__name__, add_handler=False)


@stub.function(
    image=stub.image,
    secret=config.ENV_SECRETS,
    gpu="T4",
    timeout=60,
)
def check_assets():
    import cdsapi
    import modal

    logger.info(f"Running locally -> {modal.is_local()}")

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

    logger.info("Getting GPU information...")
    import onnxruntime as ort

    logger.info(
        f"ort avail providers: {ort.get_available_providers()}"
    )  # output: ['CUDAExecutionProvider', 'CPUExecutionProvider']
    logger.info(f"onnxruntime device: {ort.get_device()}")  # output: GPU


@stub.function(
    image=stub.image,
    secret=config.ENV_SECRETS,
    # volumes={VOLUME_ROOT: stub.volume},
    gpu=config.DEFAULT_GPU_CONFIG,
    timeout=600,
)
def generate_forecast(
    model_name: str = config.SUPPORTED_AI_MODELS[0],
    init_datetime: datetime.datetime = datetime.datetime(2023, 7, 1, 0, 0),
):
    """Generate a forecast using the specified model."""
    logger.info(f"Attempting to initialize model {model_name}...")
    logger.info(f"   Run initialization datetime: {init_datetime}")
    out_pth = config.make_output_path(model_name, init_datetime)
    out_pth.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"   Model output path: {str(out_pth)}")
    init_model = model.load_model(
        # Necessary arguments to instantiate a Model object
        input="cds",
        output="file",
        download_assets=False,
        name=model_name,
        # Additional arguments. These are generally set as object attributes
        # which are then referred to by various Model methods; unfortunately,
        # they're not clearly declared in the class documentation so there is
        # a bit of trial and error involved in figuring out what's needed.
        assets=config.AI_MODEL_ASSETS_DIR,
        date=int(init_datetime.strftime("%Y%m%d")),
        time=init_datetime.hour,
        lead_time=12,
        path=str(out_pth),
        metadata={},  # Read by the output data handler
        # Unused arguments that are required by Model class methods to work.
        model_args={},
        assets_sub_directory=None,
        staging_dates=None,
        archive_requests=False,
        only_gpu=True,  # TODO: change this to True once we run on GPUs
    )
    logger.info("Generating forecast...")
    init_model.run()
    logger.info("Done!")

    # Double check that we successfully produced a model output file.
    logger.info(f"Checking output file {str(out_pth)}...")
    if out_pth.exists():
        logger.info("   Success!")
    else:
        logger.info("   Did not find expected output file.")


@stub.local_entrypoint()
def main():
    check_assets.remote()
    generate_forecast.remote()
