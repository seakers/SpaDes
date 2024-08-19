from typing import List, Optional
from datetime import timedelta

from geopandas import GeoDataFrame
from pandas import to_timedelta
from pydantic import AwareDatetime, BaseModel, Field

from .geometry import Point, Feature, FeatureCollection
from .targets import TargetPoint
from .grids import UniformAngularGrid
from .satellites import Satellite

class CoverageRequest(BaseModel):
    start: AwareDatetime = Field(..., description="Coverage analysis start time.")
    duration: timedelta = Field(..., ge=0, description="Coverage analysis duration.")
    targets: List[TargetPoint] = Field(..., description="Target points.")
    satellites: List[Satellite] = Field(..., description="Member satellites.")

class CoverageRecord(BaseModel):
    target: TargetPoint = Field(..., description="Target point.")
    mean_revisit: Optional[timedelta] = Field(
        None,
        ge=0, description="Mean revisit time computed as the mean time between observations."
    )
    number_samples: int = Field(
        0,
        ge=0, description="Number of observation samples."
    )

    def as_feature(self) -> Feature:
        """
        Convert this coverage record to a GeoJSON `Feature`.
        """
        return Feature(
            type="Feature",
            geometry=self.as_geometry(),
            properties=self.model_dump(),
        )

    def as_geometry(self) -> Point:
        """
        Convert this coverage record to a GeoJSON `Point` geometry.
        """
        return self.target.as_geometry()

class CoverageResponse(BaseModel):
    records: List[CoverageRecord] = Field([], description="Coverage results")
    harmonic_mean_revisit: Optional[timedelta] = Field(
        None,
        ge=0, description="Harmonic mean revisit time over all samples."
    )
    coverage_fraction: float = Field(
        0, ge=0, le=1, description="Fraction of samples that observed as least once."
    )

    def as_features(self) -> FeatureCollection:
        """
        Converts this coverage response to a GeoJSON `FeatureCollection`.
        """
        return FeatureCollection(
            type="FeatureCollection",
            features=[record.as_feature() for record in self.records],
        )

    def as_dataframe(self) -> GeoDataFrame:
        """
        Converts this coverage response to a `geopandas.GeoDataFrame`.
        """
        gdf = GeoDataFrame.from_features(self.as_features())
        gdf["mean_revisit"] = to_timedelta(gdf["mean_revisit"]) # helper for type coersion
        return gdf
