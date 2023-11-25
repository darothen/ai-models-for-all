# `ai-models` For All

This repository is intended to provide a template that users can adapt to start generating their own weather forecasts leveraging the "pure AI" NWP systems recently developed and open-sourced by Huawei, Nvidia, and Google DeepMind. We boot-strap on top of the fantastic [`ai-models`](https://github.com/ecmwf-lab/ai-models) repository published by ECMWF Labs, but implement our own wrappers to help end-users quickly get started generating their own forecasts.

## Setting up Google Cloud

The current version of this library ships with a single storage handler - a tool
to upload to Google Cloud Storage. Without an external storage mechanism, there
isn't a simple way to access the forecasts you generate using this tool.
Thankfully, a basic cloud storage setup is easy to create, extremely cheap, and
will serve you many purposes!

There are two main steps that you need to take here. We assume you already have
an account on Google Cloud Platform (it should be trivial to setup from
http://console.cloud.google.com).

### 1) Create a bucket on Google Cloud Storage

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

### 2) Create a Service Account for Modal

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
