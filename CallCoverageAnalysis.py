import numpy as np
from datetime import timedelta
from joblib import Parallel, delayed
from scipy.stats import hmean
import pandas as pd

from tatc.schemas import Instrument, Satellite as TATC_Satellite, TwoLineElements, Point
from tatc.analysis import collect_multi_observations, aggregate_observations, reduce_observations
from tatc.utils import swath_width_to_field_of_regard

from eose.coverage import CoverageRecord, CoverageRequest, CoverageResponse
from eose.grids import UniformAngularGrid
from eose.orbits import GeneralPerturbationsOrbitState
from eose.satellites import Satellite
from eose.targets import TargetPoint

import requests
import json
from datetime import datetime, timedelta, timezone

from shapely.geometry import box, mapping

import matplotlib.pyplot as plt
import geopandas as gpd


def coverage_tatc(request: CoverageRequest) -> CoverageResponse:
    unique_ids = len(set(target.id for target in request.targets)) == len(request.targets)
    points = [
        Point(
            id = target.id if unique_ids and isinstance(target.id, int) else i,
            longitude = target.position[0],
            latitude = target.position[1],
            altitude = target.position[2] if len(target.position) > 2 else 0
        )
        for i, target in enumerate(request.targets)
    ]
    satellites = [
        TATC_Satellite(
            name=satellite.orbit.object_name,
            orbit=TwoLineElements(tle=satellite.orbit.to_tle()),
            instruments=[
                Instrument(
                    name="Default",
                    field_of_regard=satellite.field_of_view
                )
            ]
        )
        for satellite in request.satellites
    ]

    observations = reduce_observations(
        aggregate_observations(
            pd.concat(
                Parallel(-1)(
                    delayed(collect_multi_observations)(
                        point, 
                        satellites, 
                        request.start, 
                        request.start + request.duration
                    )
                    for point in points
                )
            )
        )
    )

    records = list(
        observations.apply(
            lambda r: CoverageRecord(
                target=request.targets[points.index(next(p for p in points if p.id == r["point_id"]))],
                mean_revisit=None if pd.isnull(r["revisit"]) else timedelta(seconds=r["revisit"].total_seconds()),
                number_samples=r["samples"]
            ),
            axis=1,
        )
    ) + [
        CoverageRecord(
            target=request.targets[i],
            mean_revisit=None,
            number_samples=0
        )
        for i, point in enumerate(points)
        if not any(observations["point_id"] == point.id)
    ]
    records.sort(key=lambda r: r.target.id)

    return CoverageResponse(
        records=records,
        harmonic_mean_revisit=None 
            if observations.dropna(subset="revisit").empty 
            else timedelta(seconds=hmean(observations.dropna(subset="revisit")["revisit"].dt.total_seconds())),
        coverage_fraction=len(observations.index) / len(points)
    )

def tatcCovReqTransformer(jsonPath):
    requestJSON = open(jsonPath)
    requestObject = json.load(requestJSON)

    satelliteDicts = requestObject["satellites"]

    analysisType = requestObject["analysisType"]
    startString = requestObject["start"]
    start = datetime(int(startString[:4]), int(startString[4:6]), int(startString[6:8]), tzinfo=timezone.utc)
    duration = timedelta(seconds=requestObject["duration"])
    samplePointsDict = requestObject["samplePoints"]
    if samplePointsDict["type"] == "grid":
        deltaLatitude = samplePointsDict["deltaLatitude"]
        deltaLongitude = samplePointsDict["deltaLongitude"]
        region = samplePointsDict["region"]
        samplePoints = UniformAngularGrid(
            delta_latitude=deltaLatitude, delta_longitude=deltaLongitude, region=mapping(box(region[0], region[1], region[2], region[3]))
        ).as_targets()
    elif samplePointsDict["type"] == "points":
        points = samplePointsDict["points"]
        samplePoints = [TargetPoint(
                            id=i,
                            position=(
                                (point[0], point[1])
                            ),
                        )
                        for i, point in enumerate(points)]

    satellites = []
    for sat in satelliteDicts:
        orbit = sat["orbit"]
        a = orbit["semiMajorAxis"] # km
        e = orbit["eccentricity"]
        i = orbit["inclination"]%180 # to make between 0 and 180 bc thats what tatc/eose expects
        lan = orbit["longAscendingNode"]
        argp = orbit["argPeriapsis"]
        trueAnomaly = orbit["trueAnomaly"]

        FOR = sat["FOR"]
        # swathWidth = sat["swathWidth"]
        # FOR = swath_width_to_field_of_regard(swathWidth*1000, (a-6371)*1000) # m
        # FOR = 100 # km

        mu = 3.986e14
        P = 2*np.pi*np.sqrt((a*1000)**3/mu) # seconds
        meanMotion = 1/P * 60 * 60 * 24 # revs per day
        eccentricAnomaly = np.arctan2(np.sqrt(1-e**2)*np.sin(np.deg2rad(trueAnomaly)),1+e*np.cos(np.deg2rad(trueAnomaly)))
        meanAnomaly = np.deg2rad(eccentricAnomaly - e*np.sin(eccentricAnomaly))
        adjSat = { # format comes from celestrack omm format --- Assume sgp4 
            "OBJECT_NAME":"ISS (ZARYA)",
            "OBJECT_ID":"1998-067A",
            "EPOCH":'2024-06-07T09:53:34.728000', # make same as start of analysis period
            "MEAN_MOTION":meanMotion, # revs per day
            "ECCENTRICITY":e,
            "INCLINATION":i,
            "RA_OF_ASC_NODE":lan,
            "ARG_OF_PERICENTER":argp,
            "MEAN_ANOMALY":meanAnomaly,
            "EPHEMERIS_TYPE":0,
            "CLASSIFICATION_TYPE":analysisType,
            "NORAD_CAT_ID":25544,
            "ELEMENT_SET_NO":999,
            "REV_AT_EPOCH":45703,
            "BSTAR":0, # often 0 for artificial orbits
            "MEAN_MOTION_DOT":0, # often 0 for art
            "MEAN_MOTION_DDOT":0 # always 0 for sgp4
        }

        print("\nSemi Major Axis: ", a)
        print("Eccentricity: ", e)
        print("Inclination: ", i)
        print("LAN: ", lan)
        print("Arg Periapsis: ", argp)
        print("True Anomaly: ", trueAnomaly)
        
        
        satObj = Satellite(
            orbit=GeneralPerturbationsOrbitState.from_omm(adjSat),
            field_of_view=FOR # really field of regard
        )

        satellites.append(satObj)

        # return satellites
    request = CoverageRequest(
        satellites=satellites,
        targets=samplePoints,
        start=start,
        duration=duration,
    )
    # print(request.model_dump_json())

    response = coverage_tatc(request)
    # plotCoverage(response)

    harmonicMeanRevisit = response.harmonic_mean_revisit/timedelta(hours=1)

    return harmonicMeanRevisit
        
def plotCoverage(response):
    data = response.as_dataframe()

    # load shapefile
    world = gpd.read_file(
        "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
    )

    # example composite plot using GeoDataFrames
    fig, ax = plt.subplots()
    ax.set_title(f"Number Samples (Coverage={response.coverage_fraction:.1%})")
    data.plot(ax=ax, column="number_samples", legend=True)
    world.boundary.plot(ax=ax, lw=0.5, color="k")
    ax.set_aspect("equal")
    plt.show()

    # example composite plot using GeoDataFrames
    fig, ax = plt.subplots()
    ax.set_title(f"Mean Revisit (Harmonic Mean={response.harmonic_mean_revisit/timedelta(hours=1):.1f} hr)")
    data["mean_revisit_hr"] = data.apply(lambda r: r["mean_revisit"]/timedelta(hours=1), axis=1)
    data.plot(ax=ax, column="mean_revisit_hr", legend=True)
    world.boundary.plot(ax=ax, lw=0.5, color="k")
    ax.set_aspect("equal")
    plt.show()


# jsonPath = 'coverageAnalysisCallObject0.json'
# satellites = tatcCovReqTransformer(jsonPath)

# request = CoverageRequest(
#     satellites=satellites,
#     targets=UniformAngularGrid(
#         delta_latitude=20, delta_longitude=20, region=mapping(box(-180, -50, 180, 50))
#     ).as_targets(),
#     start=datetime(2024, 1, 1, tzinfo=timezone.utc),
#     duration=timedelta(days=7),
# )
# # print(request.model_dump_json())

# response = coverage_tatc(request)
# # print(response.model_dump_json())

# data = response.as_dataframe()
# print(data)

# import matplotlib.pyplot as plt
# import geopandas as gpd

# # load shapefile
# world = gpd.read_file(
#     "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
# )

# # example composite plot using GeoDataFrames
# fig, ax = plt.subplots()
# ax.set_title(f"Number Samples (Coverage={response.coverage_fraction:.1%})")
# data.plot(ax=ax, column="number_samples", legend=True)
# world.boundary.plot(ax=ax, lw=0.5, color="k")
# ax.set_aspect("equal")
# plt.show()

# # example composite plot using GeoDataFrames
# fig, ax = plt.subplots()
# ax.set_title(f"Mean Revisit (Harmonic Mean={response.harmonic_mean_revisit/timedelta(hours=1):.1f} hr)")
# data["mean_revisit_hr"] = data.apply(lambda r: r["mean_revisit"]/timedelta(hours=1), axis=1)
# data.plot(ax=ax, column="mean_revisit_hr", legend=True)
# world.boundary.plot(ax=ax, lw=0.5, color="k")
# ax.set_aspect("equal")
# plt.show()