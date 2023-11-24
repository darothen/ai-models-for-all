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