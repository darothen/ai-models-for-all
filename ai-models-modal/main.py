"""A Modal application for running `ai-models` weather forecasts."""
import datetime
import os
import pathlib

import modal
import ujson
from ai_models import model

from . import config, gcs
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
    """This is a placeholder function for testing that the application and credentials
    are all set up correctly and working as expected."""
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
    # NOTE: Right now, this will throw a UserWarning: "libexpat.so.1: cannot
    # open shared object file: No such file or directory." This is likely due to
    # something not being built correctly by mamba in the application image, but
    # it doesn't impact functionality at the moment.
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

    logger.info("Checking for access to GCS...")

    service_account_info = gcs.get_service_account_json("GCS_SERVICE_ACCOUNT_INFO")
    gcs_handler = gcs.GoogleCloudStorageHandler.with_service_account_info(
        service_account_info
    )
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    logger.info(f"Listing blobs in GCS bucket gs://{bucket_name}")
    blobs = list(gcs_handler.client.list_blobs(bucket_name))
    logger.info(f"Found {len(blobs)} blobs:")
    for i, blob in enumerate(blobs, 1):
        logger.info(f"({i}) {blob.name}")


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
        model_init: datetime.datetime = datetime.datetime(2023, 7, 1, 0, 0),
    ) -> None:
        self.model_name = model_name
        self.model_init = model_init
        self.out_pth = config.make_output_path(model_name, model_init)
        self.out_pth.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Run initialization datetime: {self.model_init}")
        logger.info(f"   Model output path: {str(self.out_pth)}")
        logger.info("Running model initialization / staging...")
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
            date=int(self.model_init.strftime("%Y%m%d")),
            time=self.model_init.hour,
            # TODO: allow user to specify desired forecast lead time, with limited
            # validation (e.g. < 10 days)
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
        logger.info("... done! Model is initialized and ready to run.")

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
    logger.info(f"Building model {model_name}...")
    ai_model = AIModel(model_name, init_datetime)
    logger.info(f"... model ready!")

    logger.info("Generating forecast...")
    ai_model.run_model.remote()
    logger.info("... forecast complete!")

    # Double check that we successfully produced a model output file.
    logger.info(f"Checking output file {str(ai_model.out_pth)}...")
    if ai_model.out_pth.exists():
        logger.info("   Success!")
    else:
        logger.info("   Did not find expected output file.")

    # Try to upload to Google Cloud Storage
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "")
    service_account_info = gcs.get_service_account_json("GCS_SERVICE_ACCOUNT_INFO")
    try:
        service_account_info: dict = ujson.loads(
            os.environ.get("GCS_SERVICE_ACCOUNT_INFO", "")
        )
    except ujson.JSONDecodeError:
        logger.warning("Could not parse 'GCS_SERVICE_ACCOUNT_INFO'")
        service_account_info = {}

    if (bucket_name is None) or (not service_account_info):
        logger.warning("Not able to access to Google Cloud Storage; skipping upload.")
        return

    logger.info("Attempting to upload to GCS bucket gs://{bucket_name}...")
    gcs_handler = gcs.GoogleCloudStorageHandler.with_service_account_info(
        service_account_info
    )
    dest_blob_name = ai_model.out_pth.name
    logger.info(f"Uploading to gs://{bucket_name}/{dest_blob_name}")
    gcs_handler.upload_blob(
        bucket_name,
        ai_model.out_pth,
        dest_blob_name,
    )
    logger.info("Checking that upload was successful...")
    target_blob = gcs_handler.client.bucket(bucket_name).blob(dest_blob_name)
    if target_blob.exists():
        logger.info("   Success!")
    else:
        logger.info(
            f"   Did not find expected blob ({dest_blob_name}) in GCS bucket"
            f" ({bucket_name})."
        )


@stub.local_entrypoint()
def main(
    model: str = "panguweather",
    lead_time: int = 12,
    model_init: datetime.datetime = datetime.datetime(2023, 7, 1, 0, 0),
    run_checks: bool = True,
    run_forecast: bool = True,
):
    """Entrypoint for triggering a remote ai-models weather forecast run.

    Parameters:
        model: short name for the model to run; must be one of ['panguweather',
            'fourcastnet_v2', 'graphcast']. Defaults to 'panguweather'.
        lead_time: number of hours to forecast into the future. Defaults to 12.
        model_init: datetime to use when initializing the model. Defaults to
            2023-07-01T00:00.
        run_checks: enable call to remote check_assets() for triaging the application
            runtime environment.
        run_forecast: enable call to remote generate_forecast() for running the actual
            forecast model.
    """
    # Quick sanity checks on model arguments; if we don't need to call out to our
    # remote apps, then we shouldn't!
    if model not in config.SUPPORTED_AI_MODELS:
        raise ValueError(
            f"User provided model_name '{model}' is not supported; must be one of"
            f" {config.SUPPORTED_AI_MODELS}."
        )

    if run_checks:
        check_assets.remote()
    if run_forecast:
        generate_forecast.remote(model_name=model, model_init=model_init)
