"""Modal object definitions for reference by other application components."""
import os

import modal

from . import config

logger = config.get_logger(__name__)


# NOTE: In the first pass here, let's store the model assets directly in the
# image that we build. We can figure out optimizations later on.
# stub.volume = modal.Volume.persisted("ai-models-assets")
def download_model_assets():
    """Download and cache the model weights necessary to run the model."""
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
inference_image = (
    modal.Image
    # Micromamba will be much faster than conda, buy we need to pin to
    # Python=3.10 to ensure ai-models' dependencies work correctly.
    .micromamba(python_version="3.10")
    .micromamba_install(
        "cudatoolkit",
        channels=[
            "conda-forge",
        ],
    )
    .pip_install(
        [
            "ai-models",
        ]
        + ["ai-models-" + model for model in config.SUPPORTED_AI_MODELS]
    )
    .run_function(download_model_assets)
)


stub = modal.Stub(name="ai-models-for-all", image=inference_image)
