"""
Geometry data types and object schema.

Redefines GeoJSON objects from `geojson_pydantic` to add constraints on:
 * longitude (-180 to 180, inclusive)
 * latitude (-90 to 90, inclusive)
"""

from typing import Annotated, ForwardRef, List, Literal, Optional, Tuple, Union

from pydantic import Field
from geojson_pydantic import (
    Feature as _Feature,
    FeatureCollection as _FeatureCollection,
    GeometryCollection as _GeometryCollection,
    LineString as _LineString,
    MultiPoint as _MultiPoint,
    MultiLineString as _MultiLineString,
    Point as _Point,
    Polygon as _Polygon,
    MultiPolygon as _MultiPolygon,
)
from skyfield.toposlib import GeographicPosition

Longitude = Annotated[
    float, Field(ge=-180, le=180, description="Decimal degrees longitude.")
]

Latitude = Annotated[
    float, Field(ge=-90, le=90, description="Decimal degrees latitude.")
]

Altitude = Annotated[float, Field(description="Meters above reference ellipsoid.")]

BoundingBox = Union[
    Tuple[Longitude, Latitude, Longitude, Latitude],
    Tuple[Longitude, Latitude, Altitude, Longitude, Latitude, Altitude],
]

Position = Union[Tuple[Longitude, Latitude], Tuple[Longitude, Latitude, Altitude]]

MultiPointCoords = List[Position]

LineStringCoords = Annotated[List[Position], Field(min_length=2)]

LinearRing = Annotated[List[Position], Field(min_length=4)]

PolygonCoords = List[LinearRing]

MultiPolygonCoords = List[PolygonCoords]


class Point(_Point):
    type: Literal["Point"] = Field("Point")
    bbox: Optional[BoundingBox] = None
    coordinates: Position

    @classmethod
    def from_skyfield(cls, position: GeographicPosition) -> "Point":
        """
        Creates a point from a Skyfield `GeographicPosition` object.
        """
        return Point(
            coordinates=(
                position.longitude.degrees,
                position.latitude.degrees,
                position.elevation.m,
            )
        )


class LineString(_LineString):
    type: Literal["LineString"] = Field("LineString")
    bbox: Optional[BoundingBox] = None
    coordinates: LineStringCoords


class MultiPoint(_MultiPoint):
    type: Literal["MultiPoint"] = Field("MultiPoint")
    bbox: Optional[BoundingBox] = None
    coordinates: MultiPointCoords


class MultiLineString(_MultiLineString):
    type: Literal["MultiLineString"] = Field("MultiLineString")
    bbox: Optional[BoundingBox] = None
    coordinates: List[LineStringCoords]


class Polygon(_Polygon):
    type: Literal["Polygon"] = Field("Polygon")
    bbox: Optional[BoundingBox] = None
    coordinates: PolygonCoords


class MultiPolygon(_MultiPolygon):
    type: Literal["MultiPolygon"] = Field("MultiPolygon")
    bbox: Optional[BoundingBox] = None
    coordinates: MultiPolygonCoords


GeometryCollection = ForwardRef("GeometryCollection")

Geometry = Annotated[
    Union[
        Point,
        MultiPoint,
        LineString,
        MultiLineString,
        Polygon,
        MultiPolygon,
        GeometryCollection,
    ],
    Field(discriminator="type"),
]


class GeometryCollection(_GeometryCollection):
    type: Literal["GeometryCollection"] = Field("GeometryCollection")
    bbox: Optional[BoundingBox] = None
    geometries: List[Geometry]


class Feature(_Feature):
    type: Literal["Feature"] = Field("Feature")
    geometry: Union[Geometry, None] = Field(...)


class FeatureCollection(_FeatureCollection):
    type: Literal["FeatureCollection"] = Field("FeatureCollection")
    bbox: Optional[BoundingBox] = None
    features: List[Feature] = Field(...)
