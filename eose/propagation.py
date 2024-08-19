from datetime import timedelta
from typing import List, Optional, Union

from pandas import to_datetime
from pydantic import AwareDatetime, BaseModel, Field
from geopandas import GeoDataFrame
from skyfield.api import load, Distance, Velocity, wgs84
from skyfield.framelib import itrs
from skyfield.positionlib import ICRF

from .geometry import Point, Feature, FeatureCollection
from .orbits import GeneralPerturbationsOrbitState
from .utils import Vector, Quaternion, CartesianReferenceFrame, FixedOrientation


class PropagationRequest(BaseModel):
    orbit: GeneralPerturbationsOrbitState = Field(
        ..., description="Orbit to be propagated."
    )
    start: AwareDatetime = Field(..., description="Propagation start time.")
    duration: timedelta = Field(..., ge=0, description="Propagation duration.")
    time_step: timedelta = Field(..., gt=0, description="Propagation time step duration.")
    frame: Union[CartesianReferenceFrame, str] = Field(
        CartesianReferenceFrame.ICRF,
        description="Reference frame in which propagation results defined.",
    )


class PropagationRecord(BaseModel):
    frame: Union[CartesianReferenceFrame, str] = Field(
        ...,
        description="Reference frame in which position/velocity are defined.",
    )
    time: AwareDatetime = Field(..., description="Time")
    position: Vector = Field(
        ...,
        description="Position (m)",
    )
    velocity: Vector = Field(
        ...,
        description="Velocity (m/s)",
    )
    body_orientation: Optional[Union[FixedOrientation, Quaternion]] = Field(
        None,
        description="Orientation of the spacecraft body-fixed frame, relative to requested frame.",
    )
    view_orientation: Optional[Quaternion] = Field(
        [0, 0, 0, 1],
        description="Orientation of the instrument view, relative to the body-fixed frame.",
    )

    def as_feature(self) -> Feature:
        """
        Convert this propagation record to a GeoJSON `Feature`.
        """
        return Feature(
            type="Feature",
            geometry=self.as_geometry(),
            properties=self.model_dump(),
        )

    def as_geometry(self) -> Point:
        """
        Convert this propagation record to a GeoJSON `Point` geometry.
        """
        ts = load.timescale()
        if self.frame == CartesianReferenceFrame.ICRF:
            icrf_position = ICRF(
                Distance(m=self.position).au,
                Velocity(km_per_s=[i / 1000 for i in self.velocity]).au_per_d,
                ts.from_datetime(self.time),
                399,
            )
        elif self.frame == CartesianReferenceFrame.ITRS:
            icrf_position = ICRF.from_time_and_frame_vectors(
                ts.from_datetime(self.time),
                itrs,
                Distance(m=self.position),
                Velocity(km_per_s=[i / 1000 for i in self.velocity]),
            )
            icrf_position.center = 399

        return Point.from_skyfield(wgs84.geographic_position_of(icrf_position))


class PropagationResponse(BaseModel):
    records: List[PropagationRecord] = Field([], description="Propagation results")

    def as_features(self) -> FeatureCollection:
        """
        Converts this propagation response to a GeoJSON `FeatureCollection`.
        """
        return FeatureCollection(
            type="FeatureCollection",
            features=[record.as_feature() for record in self.records],
        )

    def as_dataframe(self) -> GeoDataFrame:
        """
        Converts this propagation response to a `geopandas.GeoDataFrame`.
        """
        gdf = GeoDataFrame.from_features(self.as_features())
        gdf["time"] = to_datetime(gdf["time"]) # helper for type coersion
        return gdf
