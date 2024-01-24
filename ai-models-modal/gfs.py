"""Utilities for acquiring, fetching, and working with GFS/GDAS data for
use in the AI models application."""

import datetime
import pathlib
from collections import Sequence, namedtuple

import pygrib

from . import config, gcs

logger = config.get_logger(__name__)

# NOTE: Would prefer this to be a TypeAlias (https://peps.python.org/pep-0613/)
# but it's not available until Python 3.12.
GrbMessage = pygrib._pygrib.gribmessage

# Basic fields that we always want to copy over from the raw GFS/GDAS output we
# download.
PARAM_LEVEL_PL = (
    ["z", "q", "t", "u", "v"],
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
)
PARAM_SFC = ["10u", "10v", "2t"]

# Mapping of target parameters which we need to derive from the raw GFS/GDAS
# outpus that we download. Mappings can include renaming the "shortName"
# parameter, applying a a function to the data, or both.
target_mapper = namedtuple("target_mapper", ["target_field", "fn"])
TARGET_PL = {
    "gh": target_mapper("z", lambda x: x / 9.81),  # Geopotential height
}
TARGET_SFC = {
    "prmsl": target_mapper("msl", lambda x: x),  # Mean sea level pressure
}


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


def process_gdas_grib(gdas_pth: pathlib.Path) -> Sequence[GrbMessage]:
    """Process a GDAS GRIB file to prepare an input for an AI NWP forecast.

    Parameters
    ----------
    gdas_pth : pathlib.Path
        The local path to the GDAS GRIB file, most likely downloaded from GCS.

    Returns
    -------
    Sequence[GrbMessage]
        A sequence of GRIB messages which can be written to a binary output file.
    """

    output_grbs = []
    with pygrib.open(str(gdas_pth)) as grbs:
        for grb in grbs:
            short_name = grb.shortName
            # First two cases: we're just copy/pasting the GRIB message as long as
            # it's in the list of messages that we want to save.
            if short_name in PARAM_SFC:
                output_grbs.append(grb)
            elif (short_name in PARAM_LEVEL_PL[0]) and (grb.level in PARAM_LEVEL_PL[1]):
                output_grbs.append(grb)
            # Final two cases: we need to perform a manipulation on the data, either
            # changing it's shortname (common) or applying an arithmetic function,
            # or both.
            elif short_name in TARGET_SFC:
                tm = TARGET_SFC[short_name]
                grb.values[:] = tm.fn(grb.values)
                grb.shortName = tm.target_field
                output_grbs.append(grb)
            elif (short_name in TARGET_PL) and (grb.level in PARAM_LEVEL_PL[1]):
                tm = TARGET_PL[short_name]
                grb.values[:] = tm.fn(grb.values)
                grb.shortName = tm.target_field
                output_grbs.append(grb)
            else:
                continue

    return output_grbs
