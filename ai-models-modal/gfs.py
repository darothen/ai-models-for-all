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


# Density of water
RHO_WATER = 1000.0  # kg m^-3

# A `grib_mapper` is a simple wrapper for information we use to succintly identify
# and coerce GRIB messages from one source to another.
grib_mapper = namedtuple(
    "grib_mapper", ["source_field", "target_field", "fn", "source_matcher_override"]
)

# ERA5 field name -> Mapper from GDAS to ERA5
# We break these down hierarchically by the type_of_level in order to help disambiguate
# certain fields (such as "z") which be present in both single- and multi-level field
# sets. Note that these are the canonical level types, not the "pl"(->isobaricInhPa) and
# "sfc"(->surface) level types that we use for querying the CDS API.
mappers_by_type_of_level = {
    "isobaricInhPa": {
        "z": grib_mapper("gh", "z", lambda x: x * 9.81, {}),  # Geopotential height
    },
    "surface": {
        # NOTE: In GraphCast, we also consume surface geopotential height, which according
        # to the param_db (https://codes.ecmwf.int/grib/param-db/129) should just be the
        # surface orography.
        "z": grib_mapper("orog", "z", lambda x: x * 9.81, {}),  # Geopotential height
        # NOTE: We might want to copy the _original_ ERA-5 lsm field instead of using
        # the GDAS one.
        "lsm": grib_mapper("lsm", "lsm", identity, {}),  # Land-sea binary mask,
        # NOTE: This is a gross approximation to estimating 1-hr precip accumulation from
        # the available instantaneous precip rate. We should develop a more complex
        # way involving reading the hourly precip accumulations from the GFS forecasts.
        "tp": grib_mapper(
            "prate", "tp", lambda x: (x / RHO_WATER) * 3600 * 1, {}
        ),  # Total precipitation
        "msl": grib_mapper(
            "prmsl", "msl", identity, {"typeOfLevel": "meanSea"}
        ),  # Mean sea level pressure
        "10u": grib_mapper(
            "10u", "10u", identity, {"typeOfLevel": "heightAboveGround", "level": 10}
        ),  # 10 meter U wind component
        "10v": grib_mapper(
            "10v", "10v", identity, {"typeOfLevel": "heightAboveGround", "level": 10}
        ),  # 10 meter V wind component
        "100u": grib_mapper(
            "100u", "100u", identity, {"typeOfLevel": "heightAboveGround", "level": 100}
        ),  # 100 meter U wind component
        "100v": grib_mapper(
            "100v", "100v", identity, {"typeOfLevel": "heightAboveGround", "level": 100}
        ),  # 100 meter V wind component
        "2t": grib_mapper(
            "2t", "2t", identity, {"typeOfLevel": "heightAboveGround", "level": 2}
        ),  # 2 meter temperature
        "tcwv": grib_mapper(
            "pwat",
            "tcwv",
            identity,
            {"typeOfLevel": "atmosphereSingleLayer", "level": 0},
        ),  # Total column water vapor, taken from GFS precipitable water
    },
}


# NOTE: Would prefer this to be a TypeAlias (https://peps.python.org/pep-0613/)
# but it's not available until Python 3.12.
PyGribHandle = Type[pygrib._pygrib.open]
PyGribMessage = Type[pygrib._pygrib.gribmessage]


GFS_BUCKET = "global-forecast-system"


def make_gfs_ics_blob_name(model_epoch: datetime.datetime) -> str:
    """Generate the blob name for a GFS initial conditions file.

    We specifically target the GFS 0-hour forecast; in practice this shouldn't
    very much from the GDAS analysis (or GFS ANL file), but it has field names
    highly consistent with the ERA-5 metadata, for the most part.

    Parameters
    ----------
    model_epoch : datetime.datetime
        The model initialization time.
    """
    return "/".join(
        [
            f"gfs.{model_epoch:%Y%m%d}",
            f"{model_epoch:%H}",
            "atmos",
            f"gfs.t{model_epoch:%H}z.pgrb2.0p25.f000",
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


def select_grb_from_list(grbs: Sequence[PyGribMessage], **matchers) -> PyGribMessage:
    """
    Select a single GRIB message from a list of PyGribMessages using the supplied matchers.
    """
    matching_grbs = [grb for grb in grbs if grb_matches(grb, **matchers)]
    if not matching_grbs:
        for i, grb in enumerate(grbs):
            # if grb.shortName == matchers["shortName"]:
            print(i, *[(k, v, grb[k], grb[k] == v) for k, v in matchers.items()])
        raise ValueError(f"Could not match GRIB message with {matchers}")
    elif len(matching_grbs) > 1:
        raise ValueError(f"Multiple matches for {matchers}")
    return matching_grbs[0]


def grb_matches(grb: PyGribMessage, **matchers) -> PyGribMessage:
    """
    Return "true" if a GRIB message matches all the specified key-value attributes.
    """
    return all(grb[k] == v for k, v in matchers.items())


def process_gdas_grib(
    template_pth: pathlib.Path,
    gdas_pth: pathlib.Path,
    extra_template_matchers: dict = {},
) -> Sequence[PyGribMessage]:
    """Process a GDAS GRIB file to prepare an input for an AI NWP forecast.

    Parameters
    ----------
    template_pth : pathlib.Path
        The local path to the ERA-5 template GRIB file for a given model.
    gdas_pth : pathlib.Path
        The local path to the GDAS GRIB file, most likely downloaded from GCS.
    extra_template_matchers : dict, optional
        Additional key-value pairs to hard-code when selecting GRB messages from
        the template; this is useful when we need to downselect some of the
        template messages.

    Returns
    -------
    Sequence[GrbMessage]
        A sequence of GRIB messages which can be written to a binary output file.
    """
    logger.info("Reading template GRIB file %s...", template_pth)
    template_grbs = []
    with pygrib.open(str(template_pth)) as grbs:
        if extra_template_matchers:
            grbs = grbs.select(**extra_template_matchers)
        for grb in grbs:
            template_grbs.append(grb)
    logger.info("... found %d GRIB messages", len(template_grbs))

    logger.info("Copying and processing GRIB messages from %s...", gdas_pth)
    with pygrib.open(str(gdas_pth)) as source_grbs, logging_redirect_tqdm(
        loggers=[
            logger,
        ]
    ):
        # Pre-emptively subset all the source_grbs by matching against short names in
        # the template collection we previously opened. This greatly reduces the time it
        # takes to seek through the source GRIB file, which involves repeatedly reading
        # through the entire file from start to finish (~30x improvement when reading from
        # an SSD, so much faster on a cloud VM).
        all_short_names = [grb.shortName for grb in template_grbs]
        for mappers in mappers_by_type_of_level.values():
            all_short_names.extend(m.source_field for m in mappers.values())
        all_short_names = set(all_short_names)
        source_grb_list = source_grbs.select(shortName=all_short_names)

        for grb in tqdm(
            template_grbs,
            unit="msg",
            total=len(template_grbs),
            desc="GRIB messages",
        ):
            # Get the type of level so that we can match to the right mapper set.
            mappers = mappers_by_type_of_level[grb.typeOfLevel]
            if grb.shortName in mappers:
                mapper = mappers[grb.shortName]
                source_matchers = mapper.source_matcher_override
                source_grb = select_grb_from_list(
                    source_grb_list,
                    shortName=mapper.source_field,
                    typeOfLevel=source_matchers.get("typeOfLevel", grb.typeOfLevel),
                    level=source_matchers.get("level", grb.level),
                )
                old_mean = grb.values.mean()
                grb.values = mapper.fn(source_grb.values)
                new_mean = grb.values.mean()
                grb.shortName = mapper.target_field
                logger.debug(
                    "mapped: [x] | %10s | Old: %g | New: %g | Copied: %g",
                    grb.shortName,
                    old_mean,
                    mapper.fn(source_grb.values).mean(),
                    new_mean,
                )
            else:
                source_grb = select_grb_from_list(
                    source_grb_list,
                    shortName=grb.shortName,
                    typeOfLevel=grb.typeOfLevel,
                    level=grb.level,
                )
                old_mean = grb.values.mean()
                grb.values = source_grb.values
                new_mean = grb.values.mean()
                logger.debug(
                    "mapped: [ ] | %10s | Old: %g | Copied: %g",
                    grb.shortName,
                    old_mean,
                    new_mean,
                )

    return template_grbs
    return template_grbs
