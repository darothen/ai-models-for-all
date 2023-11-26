"""Modal object definitions for reference by other application components."""
import os

import modal

from . import ai_models_shim, config

logger = config.get_logger(__name__)


def download_model_assets():
    """Download and cache the model weights necessary to run the model."""
    raise Exception(
        "This function is deprecated; assets will be downloaded on the first run of a model"
        " and saved to an NFS running within the application."
    )

    from ai_models import model
    from multiurl import download

    # For each model, retrieve the pretrained model weights and cache them to
    # our volume. We are generally replicating the code from
    # ai_models.model.Model.download_assets(), but with some hard-coded options;
    # that method is also originally written as an instance method, and we don't
    # want to run the actual initializer for a model type to access it since
    # that would require us to provide input/output options and otherwise
    # prepare more generally for a model inference run - something we're not
    # ready to do at this stage of setup.
    n_models = len(config.SUPPORTED_AI_MODELS)
    for i, model_name in enumerate(config.SUPPORTED_AI_MODELS, 1):
        logger.info(f"({i}/{n_models}) downloading assets for model {model_name}...")
        model_initializer = model.available_models()[model_name].load()
        for file in model_initializer.download_files:
            asset = os.path.realpath(os.path.join(config.AI_MODEL_ASSETS_DIR, file))
            if not os.path.exists(asset):
                os.makedirs(os.path.dirname(asset), exist_ok=True)
                logger.info("downloading %s", asset)
                download(
                    model_initializer.download_url.format(file=file),
                    asset + ".download",
                )
                os.rename(asset + ".download", asset)


# Set up the image that we'll use for performing model inference.
# NOTE: We use a somewhat convoluted build procedure here, but after much trial
# and error, this seems to reliably build a working application. The biggest
# issue we ran into was getting onnx to detect our GPU and NVIDIA libraries
# correctly. To achieve this, we manually install via mamba a known, working
# combination of CUDA and cuDNN. We also have to be careful when we install the
# library for model-specific plugins to ai-models; these tended to re-install
# the CPU-only onnxruntime library, so we manually uninstall that and purposely
# install the onnxrtuntime-gpu library instead.
inference_image = (
    modal.Image
    # Micromamba will be much faster than conda, but we need to pin to
    # Python=3.10 to ensure ai-models' dependencies work correctly.
    .micromamba(python_version="3.10")
    .micromamba_install(
        "cudatoolkit=11.8",
        "cudnn<=8.7.0",
        "eccodes",
        channels=[
            "conda-forge",
        ],
    )
    # Run several successive pip installs; this makes it a little bit easier to
    # handle the dependencies and final tweaks across different plugins.
    # (1) Install ai-models and its dependencies.
    .pip_install(
        [
            "ai-models",
            "google-cloud-storage",
            "onnx==1.15.0",
            "ujson",
        ]
    )
    # (2) GraphCast has some additional requirements - mostly around building a
    # properly configured version of JAX that can run on GPUs - so we take care
    # of those here.
    .pip_install(
        ["jax[cuda11_pip]", "git+https://github.com/deepmind/graphcast.git"],
    )
    # (3) Install the ai-models plugins enabled for this package.
    .pip_install(
        [
            "ai-models-" + plugin_config.plugin_package_name
            for plugin_config in ai_models_shim.AI_MODELS_CONFIGS.values()
        ]
    )
    .run_commands("pip uninstall -y onnxruntime")
    # (4) Ensure that we're using the ONNX GPU-enabled runtime.
    .pip_install("onnxruntime-gpu==1.16.3")
    # Generate a blank .cdsapirc file so that we can override credentials with
    # environment variables later on. This is necessary because the ai-models
    # package input handler ultimately uses climetlab.sources.CDSAPIKeyPrompt to
    # create a client to the CDS API, and it has a hard-coded prompt check
    # which requires user interaction if this file doesn't exist.
    # TODO: Patch climetlab to allow env var overrides for CDS API credentials.
    .run_commands("touch /root/.cdsapirc")
)

# Set up a storage volume for sharing model outputs between processes.
# TODO: Explore adding a modal.Volume to cache model weights since it should be
# much faster for loading them at runtime.
volume = modal.NetworkFileSystem.persisted("ai-models-cache")

stub = modal.Stub(name="ai-models-for-all", image=inference_image)
