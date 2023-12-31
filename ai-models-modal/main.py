"""A Modal application for running `ai-models` weather forecasts."""
import datetime
import os
import pathlib

import modal
from ai_models import model

from . import ai_models_shim, config, gcs
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
def check_assets(skip_validate_env: bool = False):
    """This is a placeholder function for testing that the application and credentials
    are all set up correctly and working as expected."""
    import cdsapi

    if not skip_validate_env:
        config.validate_env()

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
        model_name: str = "panguweather",
        model_init: datetime.datetime = datetime.datetime(2023, 7, 1, 0, 0),
        lead_time: int = 12,
    ) -> None:
        self.model_name = model_name
        self.model_init = model_init

        # Cap forecast lead time to 10 days; the models may or may not work longer than
        # this, but this is an unnecessary foot-gun. A savvy user can disable this check
        # in-code.
        if lead_time > config.MAX_FCST_LEAD_TIME:
            logger.warning(
                f"Requested forecast lead time ({lead_time}) exceeds max; setting"
                f" to {config.MAX_FCST_LEAD_TIME}. You can manually set a higher limit in"
                "ai-models-modal/config.py::MAX_FCST_LEAD_TIME."
            )
            self.lead_time = config.MAX_FCST_LEAD_TIME
        else:
            self.lead_time = lead_time

        self.out_pth = config.make_output_path(model_name, model_init)
        self.out_pth.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Run initialization datetime: {self.model_init}")
        logger.info(f"   Forecast lead time: {self.lead_time}")
        logger.info(f"   Model output path: {str(self.out_pth)}")
        logger.info("Running model initialization / staging...")
        model_class = ai_models_shim.get_model_class(self.model_name)
        self.init_model = model_class(
            # Necessary arguments to instantiate a Model object
            input="cds",
            output="file",
            download_assets=False,
            # Additional arguments. These are generally set as object attributes
            # which are then referred to by various Model methods; unfortunately,
            # they're not clearly declared in the class documentation so there is
            # a bit of trial and error involved in figuring out what's needed.
            assets=config.AI_MODEL_ASSETS_DIR,
            date=int(self.model_init.strftime("%Y%m%d")),
            time=self.model_init.hour,
            lead_time=self.lead_time,
            path=str(self.out_pth),
            metadata={},  # Read by the output data handler
            # Unused arguments that are required by Model class methods to work.
            model_args={},
            assets_sub_directory=None,
            staging_dates=None,
            # TODO: Figure out if we can set up caching of model initial conditions
            # using the default interface.
            archive_requests=False,
            only_gpu=True,
            # Assumed set by GraphcastModel; produces additional auxiliary
            # output NetCDF files.
            debug=False,
        )
        logger.info("... done! Model is initialized and ready to run.")

    @modal.method()
    def run_model(self) -> None:
        self.init_model.run()


# This routine is made available as a stand-alone function, and it's up to the user
# to ensure that the path config.AI_MODEL_ASSETS_DIR exists and is mapped to the storage
# volume where assets should be cached. We provide this as a stand-alone function so
# that it can be called a cheaper, non-GPU instance and avoid wasting cycles outside
# of model inference on such a more expensive machine.
def _maybe_download_assets(model_name: str) -> None:
    from multiurl import download

    logger.info(f"Maybe retrieving assets for model {model_name}...")

    # For the requested model, retrieve the pretrained model weights and cache them to
    # our storage volume. We are generally replicating the code from
    # ai_models.model.Model.download_assets(), but with some hard-coded options;
    # that method is also originally written as an instance method, and we don't
    # want to run the actual initializer for a model type to access it since
    # that would require us to provide input/output options and otherwise
    # prepare more generally for a model inference run - something we're not
    # ready to do at this stage of setup.
    model_class = ai_models_shim.get_model_class(model_name)
    n_files = len(model_class.download_files)
    n_downloaded = 0
    for i, file in enumerate(model_class.download_files):
        asset = os.path.realpath(os.path.join(config.AI_MODEL_ASSETS_DIR, file))
        if not os.path.exists(asset):
            os.makedirs(os.path.dirname(asset), exist_ok=True)
            logger.info(f"({i}/{n_files}) downloading {asset}")
            download(
                model_class.download_url.format(file=file),
                asset + ".download",
            )
            os.rename(asset + ".download", asset)
            n_downloaded += 1
    if not n_downloaded:
        logger.info("   No assets need to be downloaded.")
    logger.info("... done retrieving assets.")


@stub.function(
    image=stub.image,
    secret=config.ENV_SECRETS,
    network_file_systems={str(config.CACHE_DIR): volume},
    allow_cross_region_volumes=True,
    timeout=1_800,
)
def generate_forecast(
    model_name: str = "panguweather",
    model_init: datetime.datetime = datetime.datetime(2023, 7, 1, 0, 0),
    lead_time: int = 12,
    skip_validate_env: bool = False,
):
    """Generate a forecast using the specified model."""

    if not skip_validate_env:
        config.validate_env()

    logger.info(f"Setting up model {model_name} conditions...")
    # Pre-emptively try to download assets from our cheaper CPU-only function, so that
    # we don't waste time on the GPU machine.
    _maybe_download_assets(model_name)
    ai_model = AIModel(model_name, model_init, lead_time)

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

    if (bucket_name is None) or (not service_account_info):
        logger.warning("Not able to access to Google Cloud Storage; skipping upload.")
        return

    logger.info(f"Attempting to upload to GCS bucket gs://{bucket_name}...")
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
    model_name: str = "panguweather",
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
    if model_name not in ai_models_shim.SUPPORTED_AI_MODELS:
        raise ValueError(
            f"User provided model_name '{model_name}' is not supported; must be one of"
            f" {ai_models_shim.SUPPORTED_AI_MODELS}."
        )

    if run_checks:
        check_assets.remote()
    if run_forecast:
        generate_forecast.remote(
            model_name=model_name, model_init=model_init, lead_time=lead_time
        )
