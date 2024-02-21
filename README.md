# `ai-models` For All

This package boot-straps on top of the fantastic [`ai-models`](https://github.com/ecmwf-lab/ai-models) library to build a serverless application to generate "pure AI
NWP" weather forecasts on [Modal](https://www.modal.com). Users can run their own
historical re-forecasts using either [PanguWeather](https://www.nature.com/articles/s41586-023-06185-3),
[FourCastNet](https://arxiv.org/abs/2202.11214), or [GraphCast](https://www.science.org/doi/10.1126/science.adi2336),
and save the outputs to their own cloud storage provider for further use.

The initial release of this application is fully-featured, with some limitations:

- We only provide one storage adapter, for Google Cloud Storage. This can be generalized
  to support S3, Azure, or any other provider in the future.
- By default, users may initialize a forecast from the CDS-based ERA-5 archive; we also
  the option to initialize from a GFS forecast, retrieved from NOAA's archive of these
  products on Google Cloud Storage. We do not provide a mechanism to initialize with IFS
  operational forecasts from MARS.
- The current application only runs on [Modal](https://www.modal.com); in the future, it
  would be great to port this to other serverless platforms, re-using as much of the
  core implementation as possible.

This application relies on the fantastic [`ecmwf-labs/ai-models`](https://github.com/ecmwf-lab/ai-models)
package to automate a lot of the traditional MLOps that are necessary to run this type
of product in a semi-production context. `ai-models` handles acquiring data to use as
inputs for model inference (read: generate a forecast from initial conditions) by
providing an as-needed interface with the Copernicus Data Store and MARS API, it
provides pre-trained model weights, it implements a simple ONNX-based interface
for performing model inference, and it outputs a well-formed (albeit in GRIB) output
file that can be fed into downstream workflows (e.g. model visualization). We don't
anticipate replacing this package, but we may contribute improvements and features
upstream (e.g. a high priority is writing a NetCDF output adapter that writes timesliced
files per model step, with metadata following CF conventions) as they mature here.

**Your feedback to <daniel@danielrothenberg.com> or [@danrothenberg](https://twitter.com/danrothenberg) would be greatly appreciated!**

## Usage / Restrictions

If you use this application, please give credit to [Daniel Rothenberg](https://github.com/darothen)
(<daniel@danielrothenberg.com> or [@danrothenberg](https://twitter.com/danrothenberg)),
as well as the incredible team at [ECMWF Lab](https://github.com/ecmwf-lab) and the
publishers of any forecast model you use.

**NOTE THAT EACH FORECAST MODEL PROVIDED BY AI-MODELS HAS ITS OWN LICENSE AND RESTRICTIONS**.

This package may *only* be used in a manner compliant with the licenses and terms of all
the libraries, model weights, and application platforms/services upon which it is built.
The forecasts generated by the AI models and the software which power them are *experimental in nature*
and may break or fail unexpectedly during normal use.

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
   $ modal run ai-models-modal.main \
         --model-name {panguweather,fourcastnetv2-small,graphcast} \
         --model-init 2023-07-01T00:00:00 \
         --lead-time 12 \
         [--use-gfs]
   ```

   The first time you run this, it will take a few minutes to build an image and set up
   assets on Modal. Then, the model will run remotely on Modal infrastructure, and you
   can monitor its progress via the logs streamed to your terminal. The bracketed CLI
   args are the defaults that will be used if you don't provide any.
5. Download the model output from Google Cloud Storage at **gs://{GCS_BUCKET_NAME}** as
   provided via the `.env` file.
6. Install required dependencies onto your machine using the requirements.txt file (pip install -r requirements.txt)

## Using GFS/GDAS Initial Conditions

We've implemented the ability for users to fetch initial conditions from an
archived GFS forecast cycle. In the current implementation, we make some assumptions
about how to process and map the GFS data to the ERA-5 data that the `ai-models`
package typically tries to fetch:

1. Some models require surface geopotential or orography fields as an input; we use the
   GFS/GDAS version of this data instead of copying over from ERA-5. Similarly, when
   needed we use the land-sea mask from GFS/GDAS instead of copying over ERA-5's.
2. GraphCast is initialized with accumulated precipitation data that is not readily
   available in the GFS/GDAS outputs; we currently approximate this very crudely by
   looking at the 6-hr lagged precipitation rate fields from subsequent GFS/GDAS
   analyses.
3. The AI models are not fine-tuned (yet) on GFS data, so underlying differences in the
   core distribution of atmospheric data between ERA-5 and GFS/GDAS could degrade
   forecast quality in unexpected ways. Additionally, we apply the ERA-5 derived
   Z-score or uniform distribution scaling from the parent AI models instead of providing
   new ones for GFS/GDAS data.

We use the `gfs.tHHz.pgrb2.0p25.f000` output files to pull the initial conditions. These
are available in near-real-time (unlike the final GDAS analyses, which are lagged by
about one model cycle). We may provide the option to use the `.anl` analysis files, too
or hot-start initial conditions, based on feedback from the community/users. Converting
to these files simply requires building a new set of mappers from the corresponding
ERA-5 fields.

### Running a Forecast from GFS/GDAS

To tell `ai-models-for-all` to use GFS initial conditions, simply pass the command line
flag "`--use-gfs`" and initialize a model run as usual.

```shell
$ modal run ai-models-modal.main \
      --model-name {panguweather,fourcastnetv2-small,graphcast} \
      --model-init 2023-07-01T00:00:00 \
      --lead-time 12 \
      --use-gfs \
```

The package will automatically download and process the GFS data to use for you, as well
as archive it for future reference. Please note that the model setup process (where
assets are downloaded and cached for future runs) may take much longer than usual as we
also take the liberty of generating independent copies of the ERA-5 template files used
to process the GFS data. Given the current quota restrictions on the CDS-API, this may
take a very long time (luckily, the stub functions which perform this process are super
cheap to run and will cost pennies even if they get stuck for several hours).

For your convenience, we've saved pre-computed data templates for you to use; for a
typical `.env` setup described below, you can locally run the following Google Cloud
SDK command to copy over the input templates so that `ai-models-for-all` will
automatically discover them:

```shell
$ gcloud storage cp \
    gs://ai-models-for-all-input-templates/era5-to-gfs-f000 \
    gs://${GCS_BUCKET_NAME}
```

## More Detailed Setup Instructions

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
**CDSAPI_URL** and **CDSAPI_KEY**. Note that we still create a stub RC file during
image generation, but this is a shortcut so that users only need to modify a single
file with their credentials.

## Other Notes

- The code here has liberal comments detailing development notes, caveats, gotchas,
  and opportunities.
- You may see some diagnostic outputs indicating that libraries including libexpat,
  libglib, and libtpu are missing. These should not impact the functionality of the
  current application (we've tested that all three AI models do in fact run
  and produce expected outputs).
- You still need to install some required libraries locally. These are provided for
  you in the requirements.txt file. Use pip install -r requirements.txt to install.
- It should be *very* cheap to run this application; even accounting for the time it
  takes to download model assets the first time a given AI model is run, most of the
  models can produce a 10-day forecast in about 10-15 minutes. So end-to-end, for a
  long forecast, the GPU container should really only be running for < 20 minutes,
  which means that at today's (11-25-2023) market rates of $3.73/hr per A100 GPU, it
  should cost about a bit more than a dollar to generate a forecast, all-in.

## Roadmap

The following major projects are slated for Q1'24:

**Operational AI Model Forecasts** - We will begin running pseudo-operational
forecasts for all included models in early Q1 using GFS initial conditions in
near-real-time, and disseminating the outputs in a publicly available Google
Cloud Storage bucket (per model licensing restrictions).

**Post-processing / Visualization** - We will implement some simple (optional)
routines to post-process the default GRIB outputs into more standard ARCO formats,
and generate a complete set of visualizations that users can review as a stand-alone
gallery. Pending collaborations, we will try to make these available on popular
model visualization websites (contact @darothen if you're interested in hosting).

**Porting to `earth2mip`** - Although we've used ecmwf-labs/ai-models for the initial
development, this package's extremely tight coupling with ECMWF infrastructure and
the climetlab library pose considerable development challenges. Therefore, we aim
to re-write this library using the NVIDIA/earth2mip framework. This is a far more
comprehensive and extensible framework for preparing a variety of modeling and learning
tasks related to AI-NWP, and provides access to the very large library of AI models
NVIDIA is collecting for their model zoo. This will likely be built in a stand-alone
package, e.g. `earth2mip-for-all`, but the goal is to provide the same accessibility
and ease-of-use for users who simply want to run these models to create their own
forecasts with limited engineering/infrastructure investment.
