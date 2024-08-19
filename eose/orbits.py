from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field
from sgp4 import exporter, omm
from sgp4.api import Satrec


class GeneralPerturbationsOrbitState(BaseModel):
    object_name: Optional[str] = Field(None, description="Object name.")
    object_id: Optional[str] = Field(None, description="Object identifier.")
    epoch: datetime = Field(
        ..., description="Epoch (time of orbit element specification)."
    )
    mean_motion: float = Field(
        ..., gt=0, description="Mean motion (revolutions per day)."
    )
    eccentricity: float = Field(..., ge=0, lt=1, description="Eccentricity.")
    inclination: float = Field(..., ge=0, le=180, description="Inclination (degrees).")
    ra_of_asc_node: float = Field(
        ..., ge=0, lt=360, description="Right ascension of ascending node (degrees)."
    )
    arg_of_pericenter: float = Field(
        ..., ge=0, lt=360, description="Argument of pericenter (degrees)."
    )
    mean_anomaly: float = Field(
        ..., ge=0, lt=360, description="Mean anomaly (degrees)."
    )
    ephemeris_type: Optional[int] = Field(
        0,
        ge=0,
        description="Ephemeris type (0: Default (Kozai), 1: SGP, 2: SGP4 (Brouwer), 3: SDP4, 4: SGP8, 5: SDP8).",
    )
    classification_type: Optional[str] = Field(
        "U",
        pattern="[UCS]",
        description="Classification (U: unclassified, C: classified, S: secret).",
    )
    norad_cat_id: Optional[int] = Field(
        None, ge=0, description="NORAD catalog identifier."
    )
    element_set_no: Optional[int] = Field(
        None, ge=0, le=999, description="Element set number."
    )
    rev_at_epoch: Optional[int] = Field(
        None, ge=0, le=99999, description="Revolution number at epoch."
    )
    bstar: float = Field(
        0.0,
        ge=0,
        lt=1,
        description="B-star drag term (radiation pressure coefficient).",
    )
    mean_motion_dot: float = Field(
        0.0,
        gt=-1,
        lt=1,
        description="First derivative of mean motion (ballistic coefficient).",
    )
    mean_motion_ddot: float = Field(
        0.0, gt=-1, lt=1, description="Second derivative of mean motion."
    )

    @classmethod
    def from_omm(cls, omm: dict) -> "GeneralPerturbationsOrbitState":
        """
        Creates a general perturbations orbit state from an Orbit Mean-Elements Message (OMM) dictionary.
        """
        return GeneralPerturbationsOrbitState.model_validate(
            dict([(key.lower(), value) for key, value in omm.items()])
        )

    def to_omm(self) -> dict:
        """
        Converts this general perturbations orbit state to Orbit Mean-Elements Message (OMM) dictionary.
        """
        return dict(
            [
                (
                    key.upper(),
                    (
                        ""
                        if value is None
                        else value.isoformat() if isinstance(value, datetime) else value
                    ),
                )
                for key, value in self.model_dump().items()
            ]
        )

    def to_satrec(self) -> Satrec:
        """
        Converts this general perturbations orbit state to an `sgp4.api.Satrec` object.
        """
        sat = Satrec()
        omm.initialize(sat, self.to_omm())
        return sat

    def to_tle(self) -> List[str]:
        """
        Converts this general perturbations orbit state to Two Line Element (TLE) list of strings.
        """
        return exporter.export_tle(self.to_satrec())
