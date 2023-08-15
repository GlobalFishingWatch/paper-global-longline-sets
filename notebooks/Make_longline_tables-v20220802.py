# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Make Longline Tables
#
# This notebook creates the table `global-fishing-watch.paper_global_longline_sets.longline_sets_categorised_v20220801`, each predicted longline set is assigned to one of 6 categories: `entirely a night`, `entirely during the day`, `Sets Over Dusk <2h`, `Sets Over Dusk >2h`, `Sets Over Dawn <2h`, `Sets Over Dawn >2h`. Based on the starting location of a set, each set is assigned to one of the regions of interest; a region in CCSBT (or "other" if not in CCSBT), and whether or not it is in WCPFC.
#
# This notebook also creates the table `global-fishing-watch.paper_global_longline_sets.longline_sets_categorised60-60_v20220801` The same as above with 60 minutes cut off each end of a set.

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import numpy as np

np.random.seed(0)
import time
import datetime
from jinja2 import Template
from jinja2 import Environment, BaseLoader
from jinja2 import Environment, FileSystemLoader
from google.cloud import bigquery
import geopandas
from geopandas import GeoSeries
from shapely.geometry import Polygon


def gbq(q):
    return pd.read_gbq(q)


# -

# %load_ext autoreload
# %autoreload 2

# +
import sys

sys.path.append("../queries/")
sys.path.append("../data/")
import Regions_json

from config import (
    NAUTICAL_TIME_RASTER_TABLE,
    LONGLINE_FEATURES_TABLE,
    LONGLINE_EVENTS_TABLE,
    LONGLINE_POSITION_SCORE_TABLE,
    CATEGORISED_SETS_TABLE,
)
# -

params = {
    "LONGLINE_EVENTS_TABLE": LONGLINE_EVENTS_TABLE,
    "LONGLINE_POSITION_SCORE_TABLE": LONGLINE_POSITION_SCORE_TABLE,
    "CATEGORISED_SETS_TABLE": CATEGORISED_SETS_TABLE,
    "SETS_TIME_ADJUST": [0, 0],
}


def run_script(script, path, params, save_to, check_query=False):
    with open(os.path.join(path, script)) as f:
        sql_template = Environment(loader=FileSystemLoader(path)).from_string(f.read())

    # Format the query according to the desired features
    q = sql_template.render(
        NAUTICAL_TIME_RASTER_TABLE=NAUTICAL_TIME_RASTER_TABLE,
        LONGLINE_FEATURES_TABLE=LONGLINE_FEATURES_TABLE,
        LONGLINE_EVENTS_TABLE=params["LONGLINE_EVENTS_TABLE"],
        LONGLINE_POSITION_SCORE_TABLE=params["LONGLINE_POSITION_SCORE_TABLE"],
        NORTH_PACIFIC=Regions_json.North_Pacific,
        SOUTH_PACIFIC=Regions_json.South_Pacific,
        SOUTH_ATLANTC=Regions_json.South_Atlantic,
        SOUTH_INDIAN=Regions_json.South_Indian,
        SETS_TIME_ADJUST_START=params["SETS_TIME_ADJUST"][0],
        SETS_TIME_ADJUST_END=params["SETS_TIME_ADJUST"][1],
    )

    if check_query is True:
        print(q)
    else:
        client = bigquery.Client()
        job_config = bigquery.QueryJobConfig(
            destination=f"{params[save_to]}", write_disposition="WRITE_TRUNCATE"
        )

        # Start the query, passing in the extra configuration.
        query_job = client.query(q, job_config=job_config)
        query_job.result()  # Wait for the job to complete.

    print("Query results loaded to the table{}".format(params[save_to]))


# # Make table with sets categorised as night, day etc

run_script("categorise_sets_day_night.sql.j2", "../queries", params,
           'CATEGORISED_SETS_TABLE',check_query=False)

# # Make events table with 1 hour cut off each end of a set

params["SETS_TIME_ADJUST"] = [60, 60]
params["LONGLINE_EVENTS_TABLE"] = LONGLINE_EVENTS_TABLE
params["LONGLINE_EVENTS_TABLE_NEW"] = (
    LONGLINE_EVENTS_TABLE[:-1]
    + str(params["SETS_TIME_ADJUST"][0])
    + "-"
    + str(params["SETS_TIME_ADJUST"][1])
    + "_"
)
run_script(
    "adjust_start_end_times.sql.j2",
    "../queries",
    params,
    "LONGLINE_EVENTS_TABLE_NEW",
    check_query=False,
)

# # Make table with sets categorised as night, day etc from events table with 1 hour cut-off

# +
params["LONGLINE_EVENTS_TABLE"] = params["LONGLINE_EVENTS_TABLE_NEW"]

params["CATEGORISED_SETS_TABLE"] = (
    CATEGORISED_SETS_TABLE[0:-10]
    + str(params["SETS_TIME_ADJUST"][0])
    + "-"
    + str(params["SETS_TIME_ADJUST"][1])
    + "_"
    + CATEGORISED_SETS_TABLE[-9:]
)
run_script(
    "categorise_sets_day_night.sql.j2",
    "../queries",
    params,
    "CATEGORISED_SETS_TABLE",
    check_query=False,
)
# -
CATEGORISED_SETS_TABLE[0:-10]


