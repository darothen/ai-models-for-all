"""Utilities for working with Google Cloud Storage.

This library is lifted as-is from darothen@'s `plotflow` library, which is
originally part of a serverless application for processing NWP outputs on the
cloud and generating visualizations from them. See
Rothenberg, Daniel: "Enabling Scalable, Serverless Weather Model Analyses by
"Kerchunking" Data in the Cloud." AMS Annual Meeting, Baltimore, MD. 2024
for more details.
"""

import os
from pathlib import Path
from typing import Any

import ujson
from google.cloud import storage

from . import config

logger = config.get_logger(__name__)


def get_service_account_json(env_var: str = "GCS_SERVICE_ACCOUNT_INFO") -> dict:
    """Try to generate service account JSON from an env var.

    Parameters:
    -----------
    env_var: str
        Name of an environment variable containing stringified JSON service account credentials.
    """
    service_account_info = os.environ.get(env_var, "")
    if not service_account_info:
        return {}
    return ujson.loads(service_account_info)


class GoogleCloudStorageHandler(object):
    def __init__(self, client: storage.Client = None):
        if client is None:
            self._client = storage.Client()
        else:
            self._client = client

    @property
    def client(self):
        return self._client

    @staticmethod
    def with_anonymous_client() -> "GoogleCloudStorageHandler":
        return GoogleCloudStorageHandler(
            client=storage.Client.create_anonymous_client()
        )

    @staticmethod
    def with_service_account_info(
        service_account_info: Any,
    ) -> "GoogleCloudStorageHandler":
        return GoogleCloudStorageHandler(
            client=storage.Client.from_service_account_info(service_account_info)
        )

    # TODO: Add retry and timeout logic to all of these functions, following the docs at
    #       https://cloud.google.com/storage/docs/retry-strategy#customize-retries
    def download_blob(
        self,
        bucket_name: str,
        source_blob_name: str,
        destination_pth: Path,
    ) -> None:
        """Download a blob from GCS to a local path.

        Parameters
        ----------
        bucket_name : str
            Bucket on GCS containing the blob to download.
        source_blob_name : str
            Name of the blob to download.
        destination_pth : Path
            Local path to download the blob to.
        """

        bucket = self.client.bucket(bucket_name)

        # NOTE: `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
        # any content from Google Cloud Storage. As we don't need additional data,
        # using `Bucket.blob` is preferred here.
        blob = bucket.blob(source_blob_name)
        logger.info(
            f"Downloading gs://{bucket_name}/{source_blob_name} to {destination_pth}"
        )
        blob.download_to_filename(destination_pth)

    def upload_blob(
        self, bucket_name: str, source_file_pth: Path, destination_blob_name: str
    ):
        """Uploads a blob to GCS from a local path.

        Parameters
        ----------
        bucket_name : str
            Bucket on GCS where the blob should be uploaded.
        source_file_pth : Path
            Local path to the file to upload.
        destination_blob_name : str
            Blob name to use when writing to `bucket_name` on GCS.
        """

        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        logger.info(
            f"Uploading {source_file_pth} to gs://{bucket_name}/{destination_blob_name}."
        )
        blob.upload_from_filename(source_file_pth)

    def upload_json_to_blob(
        self, bucket_name: str, json_str: str, destination_blob_name: str
    ):
        """Uploads JSON string to a GCS blob.

        Parameters
        ----------
        bucket_name : str
            Bucket on GCS where the blob should be uploaded.
        json_str : str
            Encoded JSON data string to upload.
        destination_blob_name : str
            Blob name to use when writing to `bucket_name` on GCS.
        """

        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        logger.info(f"Uploading JSON to gs://{bucket_name}/{destination_blob_name}.")
        blob.upload_from_string(data=json_str, content_type="application/json")
