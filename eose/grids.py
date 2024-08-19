"""
Models to represent target grids.
"""

import itertools
import math
from typing import List, Optional, Union

from geopandas import GeoDataFrame
from pydantic import BaseModel, Field
from shapely.geometry import shape, Point as SPoint

from .geometry import Altitude, FeatureCollection, MultiPolygon, Polygon
from .targets import TargetPoint
from .utils import PlanetaryCoordinateReferenceSystem


class UniformAngularGrid(BaseModel):
    """
    Specifies a grid with uniform angular spacing in longitude and latitude.

    The grid enumerates target points from West-to-East (column index j)
    followed by South-to-North (row index i) with properties assigned by:
     * `id = i + j * floor(360/delta_longitude)`
     * `longitude = -180 + (i + 0.5) * delta_longitude`
     * `latitude = -90 + (j + 0.5) * delta_latitude`

    An optional region serves as a mask to constrain targets.
    """

    delta_longitude: float = Field(
        ..., gt=0, description="Longitude separation (decimal degrees) between targets."
    )
    delta_latitude: float = Field(
        ..., gt=0, description="Latitude separation (decimal degrees) between targets."
    )
    altitude: Optional[Altitude] = Field(
        None, description="Target altitude (optional)."
    )
    region: Optional[Union[MultiPolygon, Polygon]] = Field(
        None, description="Spatial region in which to generate targets."
    )
    crs: Optional[PlanetaryCoordinateReferenceSystem] = Field(
        None, description="Coordinate reference system in which targets are defined."
    )

    def as_dataframe(self) -> GeoDataFrame:
        """
        Converts this uniform angular grid to a `geopandas.GeoDataFrame` object.
        """
        return GeoDataFrame.from_features(self.as_features())

    def as_features(self) -> FeatureCollection:
        """
        Converts this uniform angular grid to a GeoJSON `FeatureCollection`.
        """
        return FeatureCollection(
            type="FeatureCollection",
            features=[target.as_feature() for target in self.as_targets()],
        )

    def as_targets(self) -> List[TargetPoint]:
        """
        Converts this uniform angular grid into a list of `TargetPoint` objects.
        """
        if self.region is None:
            min_i = 0
            max_i = math.floor(360 / self.delta_longitude)
            min_j = 0
            max_j = math.floor(180 / self.delta_latitude)
        else:
            min_lon, min_lat, max_lon, max_lat = shape(self.region).bounds
            min_i = math.floor((min_lon + 180) / self.delta_longitude)
            max_i = math.ceil((max_lon + 180) / self.delta_longitude)
            min_j = math.floor((min_lat + 90) / self.delta_latitude)
            max_j = math.ceil((max_lat + 90) / self.delta_latitude)

        return [
            TargetPoint(
                id=id,
                crs=self.crs,
                position=(
                    (longitude, latitude)
                    if self.altitude is None
                    else (longitude, latitude, self.altitude)
                ),
            )
            for j, i in itertools.product(range(min_j, max_j), range(min_i, max_i))
            for id in [i + j * math.floor(360 / self.delta_longitude)]
            for longitude in [-180 + (i + 0.5) * self.delta_longitude]
            for latitude in [-90 + (j + 0.5) * self.delta_latitude]
            if (
                self.region is None
                or shape(self.region).covers(SPoint(longitude, latitude))
            )
        ]
