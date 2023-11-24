# `ai-models` For All

This repository is intended to provide a template that users can adapt to start generating their own weather forecasts leveraging the "pure AI" NWP systems recently developed and open-sourced by Huawei, Nvidia, and Google DeepMind. We boot-strap on top of the fantastic [`ai-models`](https://github.com/ecmwf-lab/ai-models) repository published by ECMWF Labs, but implement our own wrappers to help end-users quickly get started generating their own forecasts.

## Quick-Start Guide - Modal

1. Set up your preferred python environment and install the `modal` client package:
   ```shell 
   $ pip install modal
   ```
2. Set up an account and credentials [modal](https://modal.com/). Be sure that you've set up your API token correctly, see their ["Getting Started"](https://modal.com/home) page for instructions.
3. Build and deploy an image of the `ai-models-modal` application; from the top-level directory, execute
   ```shell
   $ modal deploy ai-models-modal.main 
   ```
4. Run the test application, which should log the filenames of assets pre-cached with the image:
   ```shell
   $ modal run ai-models-modal.main
   ```

## Configuring `cdsapi`

We need access to the [Copernicus Data Store](https://cds.climate.copernicus.eu/)
to retrieve historical ERA-5 data to use when initializing our forecasts. The
easiest way to set this up would be to have the user retrieve their credentials
from [here](https://cds.climate.copernicus.eu/api-how-to) and save them to a
local file, `~/.cdsapirc`. But that's a tad inconvenient to build into our
application image. Instead, we can just set the environment variables
**CDSAPI_URL** and **CDSAPI_KEY**.
