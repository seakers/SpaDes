"""
Models to represent observation targets.
"""

from typing import Optional

from pydantic import BaseModel, Field

from .utils import PlanetaryCoordinateReferenceSystem, Identifier
from .geometry import Position, Point, Feature


class TargetPoint(BaseModel):
    """
    Target point on or above the surface of a planetary body.
    """

    id: Optional[Identifier] = Field(None, description="Target identifier.")
    crs: Optional[PlanetaryCoordinateReferenceSystem] = Field(
        None, description="Coordinate reference system in which this target is defined."
    )
    position: Position = Field(..., description="Position of this target.")

    def as_geometry(self) -> Point:
        """
        Convert this target point to a GeoJSON `Point`.
        """
        return Point(type="Point", coordinates=self.position)

    def as_feature(self) -> Feature:
        """
        Convert this target point to a GeoJSON `Feature`.
        """
        return Feature(
            type="Feature",
            geometry=self.as_geometry(),
            properties={"id": self.id, "crs": self.crs},
        )

    @classmethod
    def from_geometry(cls, geometry: Point) -> "TargetPoint":
        """
        Creates a target point from a GeoJSON `Point` object.
        """
        return TargetPoint(position=geometry.coordinates)

    @classmethod
    def from_feature(cls, feature: Feature) -> "TargetPoint":
        """
        Creates a target point from a GeoJSON `Feature` object.
        """
        if not isinstance(feature.geometry, Point):
            raise ValueError("Geometry must be a Point.")
        return TargetPoint(
            id=feature.properties.get("id"),
            crs=feature.properties.get("crs"),
            position=feature.geometry.coordinates,
        )
