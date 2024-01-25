"""Utilities for acquiring, fetching, and working with GFS/GDAS data for
use in the AI models application."""

import datetime
import pathlib
from collections import namedtuple
from typing import Any, Sequence, Type

import pygrib
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from . import config

logger = config.get_logger(__name__)


def identity(x: Any) -> Any:
    """Identity pass-through function."""
    return x


# A `grib_mapper` is a simple wrapper for information we use to succintly identify
# and coerce GRIB messages from one source to another.
grib_mapper = namedtuple(
    "grib_mapper", ["source_field", "target_field", "fn", "source_matcher_override"]
)

# ERA5 field name -> Mapper from GDAS to ERA5
mappers = {
    "z": grib_mapper("gh", "z", lambda x: x * 9.81, {}),  # Geopotential height
    "msl": grib_mapper(
        "prmsl", "msl", identity, {"typeOfLevel": "meanSea"}
    ),  # Mean sea level pressure
    "10u": grib_mapper(
        "10u", "10u", identity, {"typeOfLevel": "heightAboveGround", "level": 10}
    ),  # 10 meter U wind component
    "10v": grib_mapper(
        "10v", "10v", identity, {"typeOfLevel": "heightAboveGround", "level": 10}
    ),  # 10 meter V wind component
    "2t": grib_mapper(
        "2t", "2t", identity, {"typeOfLevel": "heightAboveGround", "level": 2}
    ),  # 2 meter temperature
}

# NOTE: Would prefer this to be a TypeAlias (https://peps.python.org/pep-0613/)
# but it's not available until Python 3.12.
PyGribHandle = Type[pygrib._pygrib.open]
PyGribMessage = Type[pygrib._pygrib.gribmessage]


GFS_BUCKET = "global-forecast-system"


def make_gfs_ics_blob_name(model_epoch: datetime.datetime) -> str:
    """Generate the blob name for a GFS initial conditions file.

    We specifically target the GDAS 0-hour atmosphere analysis, as this has a
    comprehensive set of outputs which we can use to cull the data we need to
    initialize an AI NWP forecast.

    Parameters
    ----------
    model_epoch : datetime.datetime
        The model initialization time.
    """
    return "/".join(
        [
            f"gdas.{model_epoch:%Y%m%d}",
            f"{model_epoch:%H}",
            "atmos",
            f"gdas.t{model_epoch:%H}z.pgrb2.0p25.f000",
        ]
    )


def make_gfs_base_pth(model_epoch: datetime.datetime) -> pathlib.Path:
    """Generate the local path for a GFS initial conditions file.

    Parameters
    ----------
    model_epoch : datetime.datetime
        The model initialization time.
    """
    return config.INIT_CONDITIONS_DIR / f"{model_epoch:%Y%m%d%H%M}"


def select_grb(grbs: PyGribHandle, **matchers) -> PyGribMessage:
    """
    Select a single GRIB message from a PyGribHandle using the supplied matchers.
    """
    matching_grbs = grbs.select(**matchers)
    if not matching_grbs:
        raise ValueError(f"Could not match GRIB message with {matchers}")
    elif len(matching_grbs) > 1:
        raise ValueError(f"Multiple matches for {matchers}")
    return matching_grbs[0]


def process_gdas_grib(
    template_pth: pathlib.Path, gdas_pth: pathlib.Path
) -> Sequence[PyGribMessage]:
    """Process a GDAS GRIB file to prepare an input for an AI NWP forecast.

    Parameters
    ----------
    template_pth : pathlib.Path
        The local path to the ERA-5 template GRIB file for a given model.
    gdas_pth : pathlib.Path
        The local path to the GDAS GRIB file, most likely downloaded from GCS.

    Returns
    -------
    Sequence[GrbMessage]
        A sequence of GRIB messages which can be written to a binary output file.
    """

    logger.info("Reading template GRIB file %s...", template_pth)
    template_grbs = []
    with pygrib.open(str(template_pth)) as grbs:
        for grb in grbs:
            template_grbs.append(grb)
    logger.info("... found %d GRIB messages", len(template_grbs))

    logger.info("Copying and processing GRIB messages from %s...", gdas_pth)
    with pygrib.open(str(gdas_pth)) as source_grbs, logging_redirect_tqdm(
        loggers=[
            logger,
        ]
    ):
        for grb in tqdm(
            template_grbs,
            unit="msg",
            total=len(template_grbs),
            desc="GRIB messages",
        ):
            if grb.shortName in mappers:
                mapper = mappers[grb.shortName]
                source_matchers = mapper.source_matcher_override
                source_grb = select_grb(
                    source_grbs,
                    shortName=mapper.source_field,
                    typeOfLevel=source_matchers.get("typeOfLevel", grb.typeOfLevel),
                    level=source_matchers.get("level", grb.level),
                )
                old_mean = grb.values.mean()
                grb.values = mapper.fn(source_grb.values)
                new_mean = grb.values.mean()
                grb.shortName = mapper.target_field
                logger.debug(
                    "Old: %g | New: %g | Copied: %g",
                    old_mean,
                    mapper.fn(source_grb.values).mean(),
                    new_mean,
                )
            else:
                source_grb = select_grb(
                    source_grbs,
                    shortName=grb.shortName,
                    typeOfLevel=grb.typeOfLevel,
                    level=grb.level,
                )
                old_mean = grb.values.mean()
                grb.values = source_grb.values
                new_mean = grb.values.mean()
                logger.debug(
                    "Old: %g | New: %g | Copied: %g",
                    old_mean,
                    mapper.fn(source_grb.values).mean(),
                    new_mean,
                )

    return template_grbs
