"""A Modal application for running `ai-models` weather forecasts."""
import datetime
import os
import pathlib

import modal
from ai_models import model

from . import config
from .app import stub, volume

config.set_logger_basic_config()
logger = config.get_logger(__name__, add_handler=False)


@stub.function(
    image=stub.image,
    secret=config.ENV_SECRETS,
    network_file_systems={str(config.CACHE_DIR): volume},
    gpu="T4",
    timeout=60,
    allow_cross_region_volumes=True,
)
def check_assets():
    import cdsapi

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

    logger.info(f"Checking contents on network file system at {config.CACHE_DIR}...")
    for i, asset in enumerate(config.CACHE_DIR.glob("**/*"), 1):
        logger.info(f"({i}) {asset}")


@stub.cls(
    secret=config.ENV_SECRETS,
    gpu=config.DEFAULT_GPU_CONFIG,
    network_file_systems={str(config.CACHE_DIR): volume},
    concurrency_limit=1,
    timeout=1_800,
)
class AIModel:
    def __init__(
        self,
        # TODO: Re-factor arguments into a well-structured dataclass.
        model_name: str = config.SUPPORTED_AI_MODELS[0],
        init_datetime: datetime.datetime = datetime.datetime(2023, 7, 1, 0, 0),
    ) -> None:
        self.model_name = model_name
        self.init_datetime = init_datetime
        self.out_pth = config.make_output_path(model_name, init_datetime)
        self.out_pth.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"   Run initialization datetime: {self.init_datetime}")
        logger.info(f"   Model output path: {str(self.out_pth)}")
        self.init_model = model.load_model(
            # Necessary arguments to instantiate a Model object
            input="cds",
            output="file",
            download_assets=False,
            name=self.model_name,
            # Additional arguments. These are generally set as object attributes
            # which are then referred to by various Model methods; unfortunately,
            # they're not clearly declared in the class documentation so there is
            # a bit of trial and error involved in figuring out what's needed.
            assets=config.AI_MODEL_ASSETS_DIR,
            date=int(self.init_datetime.strftime("%Y%m%d")),
            time=self.init_datetime.hour,
            lead_time=12,
            path=str(self.out_pth),
            metadata={},  # Read by the output data handler
            # Unused arguments that are required by Model class methods to work.
            model_args={},
            assets_sub_directory=None,
            staging_dates=None,
            archive_requests=False,
            only_gpu=True,
        )

    @modal.method()
    def run_model(self) -> None:
        self.init_model.run()


@stub.function(
    image=stub.image,
    secret=config.ENV_SECRETS,
    network_file_systems={str(config.CACHE_DIR): volume},
    allow_cross_region_volumes=True,
)
def generate_forecast(
    model_name: str = config.SUPPORTED_AI_MODELS[0],
    init_datetime: datetime.datetime = datetime.datetime(2023, 7, 1, 0, 0),
):
    """Generate a forecast using the specified model."""
    logger.info(f"Attempting to initialize model {model_name}...")
    ai_model = AIModel(model_name, init_datetime)

    logger.info("Generating forecast...")
    ai_model.run_model.remote()
    logger.info("Done!")

    # Double check that we successfully produced a model output file.
    logger.info(f"Checking output file {str(ai_model.out_pth)}...")
    if ai_model.out_pth.exists():
        logger.info("   Success!")
    else:
        logger.info("   Did not find expected output file.")


@stub.local_entrypoint()
def main():
    check_assets.remote()
    generate_forecast.remote()
