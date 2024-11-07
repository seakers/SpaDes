"""
Utility data types.
"""

from enum import Enum
from typing import Annotated, List, Union

from pydantic import Field, StrictInt, StrictStr

Identifier = Union[StrictInt, StrictStr]

Vector = Annotated[
    List[float],
    Field(min_length=3, ma_length=3, description="Cartesian vector (x,y,z)."),
]

Quaternion = Annotated[
    List[float],
    Field(min_length=4, max_length=4, description="Quaternion (x,y,z,w)."),
]


class PlanetaryCoordinateReferenceSystem(str, Enum):
    EPSG_4326 = "EPSG:4326"  # Earth WGS 1984
    ESRI_104902 = "ESRI:104902"  # Venus 2000
    ESRI_104903 = "ESRI:104903"  # Moon 2000
    ESRI_104905 = "ESRI:104905"  # Mars 2000
    ESRI_104975 = "ESRI:104975"  # Sun IAU 2015


class CartesianReferenceFrame(str, Enum):
    ITRS = "ITRS"  # International Terrestrial Reference System (ITRS)
    ICRF = "ICRF"  # International Celestial Reference Frame


class FixedOrientation(str, Enum):
    NADIR_GEOCENTRIC = "NADIR_GEOCENTRIC"  # nadir pointing through geocenter
    NADIR_GEODETIC = "NADIR_GEODETIC"  # nadir normal to ellipsoid surface
