# `ai-models` For All

This repository is intended to provide a template that users can adapt to start generating their own weather forecasts leveraging the "pure AI" NWP systems recently developed and open-sourced by Huawei, Nvidia, and Google DeepMind. We boot-strap on top of the fantastic [`ai-models`](https://github.com/ecmwf-lab/ai-models) repository published by ECMWF Labs, but implement our own wrappers to help end-users quickly get started generating their own forecasts.

This is a **preview release** of this tool; it has a few limitations:

- We only provide one storage adapter, for Google Cloud Storage (we can expand this
  to S3, Azure, or other providers as there is interest).
- We only enable access to the CDS-based archive of ERA-5 data to initialize the
  models (access via MARS will be forthcoming, but since most users will not have
  these credentials, it wasn't a high priority).
- Only PanguWeather is currently supported; once the basic kinks of this tool are worked
  out with test users, we can quickly add FourCastNet and GraphCast.
- A single initialization date and forecast lead time is hard-coded for testing purposes.
  These will ultimately be configurable via the command line so you can run an arbitrary
  forecast.
- The current application only runs on [Modal](https://www.modal.com); in the future, it
  would be great to port this to other serverless platforms.

Furthermore, we significantly rely on the fantastic [`ecmwf-labs/ai-models`](https://github.com/ecmwf-lab/ai-models)
package to automate a lot of the traditional MLOps that are necessary to run this type
of product in a semi-production context. `ai-models` handles acquiring data to use as
inputs for model inference (read: generate a forecast from initial conditions) by
providing an as-needed interface with the Copernicus Data Store and MARS API, it
provides pre-trained model weights shipped via ONNX, it implements a simple interface
for performing model inference, and it outputs a well-formed (albeit in GRIB) output
file that can be fed into downstream workflows (e.g. model visualization). We don't
anticipate replacing this package, but we may contribute improvements and features
upstream (e.g. a high priority is writing a NetCDF output adapter that writes timesliced
files per model step, with metadata following CF conventions) as they mature here.

**Your feedback to <daniel@danielrothenberg.com> or [@danrothenberg](https://twitter.com/danrothenberg) would be greatly appreciated!**

## Usage / Restrictions

If you use this package, please give credit to [Daniel Rothenberg](https://github.com/darothen)
(<daniel@danielrothenberg.com> or [@danrothenberg](https://twitter.com/danrothenberg)),
as well as the incredible team at [ECMWF Lab](https://github.com/ecmwf-lab) and the
publishers of any forecast model you use.

**NOTE THAT EACH MODEL PROVIDED BY AI-MODELS HAS ITS OWN LICENSE AND RESTRICTION**.

This package may *only* be used in a manner compliant with the licenses and terms of all
the libraries, model weights, and services upon which it is built.

## Quick Start

1. Set up accounts (if you don't already have them) for:
   1. [Google Cloud](https://cloud.google.com)
   2. [Modal](https://www.modal.com)
   3. [Copernicus Data Store](https://cds.climate.copernicus.eu/)
2. Complete the `.env` file with your CDS API credentials, GCS service account keys, and
   a bucket name where model outputs will be uploaded. **You should create this bucket
   before running the application!**
3. From a terminal, login with the `modal-client`
4. Navigate to the repository on-disk and execute the command,
   ```shell
   $ modal run ai-models-modal.main
   ```
   The first time you run this, it will take a few minutes to build an image and set up
   assets on Modal. Then, the model will run remotely on Modal infrastructure, and you
   can monitor its progress via the logs streamed to your terminal.
5. Download the model output from Google Cloud Storage at **gs://{GCS_BUCKET_NAME}** as
   provided via the `.env` file.

## Getting Started

To use this demo, you'll need accounts set up on [Google Cloud](https://cloud.google.com),
[Modal](https://www.modal.com), and the [Copernicus Data Store](<https://cds.climate.copernicus.eu/>).
Don't worry - even though you do need to supply them ith credit card information, this
demo should cost virtually nothing to run; we'll use very limited storage on Google
Cloud Storage for forecast model outputs that we generate (a few cents per month if you
become a prolific user), and Modal offers new individual users a [startup package](https://modal.com/signup)
which includes $30/month of free compute - so you could run this application for about 8
hours straight before you'd incur any fees (A100's cost about $3.73 on Modal at the time
of writing).

If you're very new to cloud computing, the following sections will help walk you through
the handful of steps necessary to get started with this demo. Experienced users can
quickly skim through to see how they need to modify the provided `.env` to set up the
necessary credentials for the application to work.

### Setting up Google Cloud

The current version of this library ships with a single storage handler - a tool
to upload to Google Cloud Storage. Without an external storage mechanism, there
isn't a simple way to access the forecasts you generate using this tool.
Thankfully, a basic cloud storage setup is easy to create, extremely cheap, and
will serve you many purposes!

There are two main steps that you need to take here. We assume you already have
an account on Google Cloud Platform (it should be trivial to setup from
http://console.cloud.google.com).

#### 1) Create a bucket on Google Cloud Storage

Navigate to your project's [Cloud Storage](https://console.cloud.google.com/storage/browser)
control panel. From here, you should see a listing of buckets that you own.

Find the button labeled **Create** near the top of the page. On the form that
loads, you should only need to provide a name for your bucket. Your name needs
to be globally unique - buckets across different projects must have different
names. We recommend the simple naming scheme `<username>-ai-models-for-all`.

Keep all the default settings after inputting your bucket name and submit the
form with the blue **Create** button at the bottom of the page.

Finally, navigate to the `.env` file in this repo; set the **GCS_BUCKET_NAME**
variable to the bucket name you chose previously. You do not need quotes around
the bucket name.

#### 2) Create a Service Account

We need a mechanism so that your app running on Modal can authenticate with
Google Cloud in order to use its resources and APIs. To do this, we're going to
create a *service account* and set the limited permissions needed to run this
application.

From your [Google Cloud Console](http://console.cloud.google.com), navigate to
the **IAM & Admin** panel and then select **Service Accounts** from the menu. On
the resulting page, click the button near the top that says
**Create Service Account**.

On the form that pops up, use the following information:

- *Service account name*: modal-ai-models-for-all
- *Service account description*: Access from Modal for ai-models-for-all application
  
The *Service account ID* field should automatically fill out for you. Click
**Create and Continue** and you should move to Step 2, "Grant this service
account access to project (optional)". Here, we will give permissions for the
service account to access Cloud Storage resources (so that it can be used to
upload and list objects in the bucket you previously created).

From the drop-down labeled "Select a role", search for "Storage Object Admin"
(you may want to use the filter). Add this role then click **Continue**. You
shouldn't need to grant any specific user permissions, assuming you're the owner
of the Google Cloud account in which you're setting this up. Click **Done**.

Finally, we need to access the credentials for this new account. Navigate back
to the page **IAM & Admin** > **Service Accounts**, and click the name in the
table with the "model-ai-models-for-all" service account you just created. At
the top of the page, navigate to the **Keys** tab and click the **ADD KEY**
button on the middle of hte page. This will generate and download a new private
key that you'll use. Select the "JSON" option from the pop-up and download the 
file by clicking **Create**.

The credentials you created will be downloaded to disk automatically. Open that
JSON file in your favorite text editor; you'll see a mapping of many different
keys to values, including "private_key". We need to pass this entire JSON object
whenever we want to authenticate with Google Cloud Storage. To do this, we'll
save it in the same `.env` file under the name **GCS_SERVICE_ACCOUNT_INFO**.

Unfortunately we can't just copy/paste - we need to "stringify" the data. You
should probably do this in Python or your preferred programming language by
reading in the JSON file you saved, serializing to a string, and outputting. In
a pinch, you can copy/paste the full JSON data into a site like [this one](https://jsonformatter.org/json-stringify-online)
and use the resulting string. Copy that output string into your `.env`.

### Configuring `cdsapi`

We need access to the [Copernicus Data Store](https://cds.climate.copernicus.eu/)
to retrieve historical ERA-5 data to use when initializing our forecasts. The
easiest way to set this up would be to have the user retrieve their credentials
from [here](https://cds.climate.copernicus.eu/api-how-to) and save them to a
local file, `~/.cdsapirc`. But that's a tad inconvenient to build into our
application image. Instead, we can just set the environment variables
**CDSAPI_URL** and **CDSAPI_KEY**.