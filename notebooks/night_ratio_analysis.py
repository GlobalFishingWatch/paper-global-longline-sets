# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Create night setting ratio table
# This notebook produces the `night setting ratios for Table 1`. The ratios of sets happening at night are computed for different time frames, fleets, and regions. The ratio can be either based on number of sets, or set duration

import pandas as pd


def perc_night(
    df, flag=None, year=None, lat_range=None, region=None, inWCPFC=False, mtype="sets"
):
    if region is not None:
        df = df[(df.region_CCSBT == region)].reset_index(drop=True).copy()
    if inWCPFC:
        df = df[(df.in_WCPFC == 1)].reset_index(drop=True).copy()
    if year is not None:
        df = df[(df.start_time.dt.year == year)].reset_index(drop=True).copy()
    if flag is not None:
        df = df[(df.best_flag == flag)].reset_index(drop=True).copy()
    if lat_range is not None:
        df = (
            df[(df.start_lat > lat_range[0]) & (df.start_lat <= lat_range[1])]
            .reset_index(drop=True)
            .copy()
        )
    print("number of sets: ", len(df))
    if len(df):
        if mtype == "sets":
            return (
                sum(df.mostly_night) / len(df) * 100,
                sum(df.cat2 == 2) / len(df) * 100,
                len(df),
            )

        elif mtype == "duration":
            return (
                sum(df[df.mostly_night].set_duration) / sum(df.set_duration) * 100,
                sum(df[df.cat2 == 2].set_duration) / sum(df.set_duration) * 100,
                len(df),
            )
    else:

        print("NA")


# %load_ext autoreload
# %autoreload 2

# # Get predicted sets

q = """
   
SELECT * FROM `paper_global_longline_sets.longline_sets_categorised_v20220801` """
df_0 = pd.read_gbq(q, project_id="global-fishing-watch")

# # Get predicted sets with 1 hour cut off either side

q = """
   
SELECT * FROM `paper_global_longline_sets.longline_sets_categorised60-60v20220801_` """
df_60 = pd.read_gbq(q, project_id="global-fishing-watch")

night_cats = [2, 5, 7]
day_cats = [1, 6, 8]

len(df_0)

len(df_60)

df_0.best_flag = df_0.best_flag.fillna("unknown")
df_60.best_flag = df_60.best_flag.fillna("unknown")

# ## Filter sets to be within 2017-2020 and between 2 and 15 hours

df_0 = (
    df_0[(df_0.start_time.dt.year >= 2017) & (df_0.start_time.dt.year <= 2020)]
    .copy()
    .reset_index(drop=True)
)

min_dur = 2
max_dur = 15
df_0 = df_0[(df_0.set_duration >= min_dur) & (df_0.set_duration <= max_dur)].copy()

# ## Remove sets from 1-hour-cut-off list to match the list of sets in standard list

joint_set_ids = set(df_0.set_id) & set(df_60.set_id)
set_id_mask_0 = [(x in joint_set_ids) for x in df_0.set_id]
df_0 = df_0[set_id_mask_0]
set_id_mask_60 = [(x in joint_set_ids) for x in df_60.set_id]
df_60 = df_60[set_id_mask_60]

len(df_0)

len(df_60)

df_0["mostly_day"] = df_0.cat2.isin(day_cats)
df_0["mostly_night"] = df_0.cat2.isin(night_cats)
df_60["mostly_day"] = df_60.cat2.isin(day_cats)
df_60["mostly_night"] = df_60.cat2.isin(night_cats)

print("mean duration: ", round(df_0.set_duration.mean(), 1))
print("std: ", round(df_0.set_duration.std(), 1))

print("mean duration: ", round(df_60.set_duration.mean(), 1))
print("std: ", round(df_60.set_duration.std(), 1))

# # Compute night setting ratios for different regions and flags

# rounding to decimal places
rnd = 1
df_columns = [
    "flag",
    "year",
    "CCSBT_Area",
    "in_WCPFC",
    "lat_range",
    "only_night",
    "only_night_60",
    "mostly_night",
    "mostly_night_60",
    "num_sets",
    "num_sets_60",
]

areas = {
    "AUS_2017_CCSBT4" : {"flag": "AUS","year": 2017,"area": "4","inWCPFC": None,"lat_range": None},
    "AUS_2017_CCSBT7" : {"flag": "AUS","year": 2017,"area": "7","inWCPFC": None,"lat_range": None},
    "JPN_2017_CCSBT4" : {"flag": "JPN","year": 2017,"area": "4","inWCPFC": None,"lat_range": None},
    "JPN_2017_CCSBT7" : {"flag": "JPN","year": 2017,"area": "7","inWCPFC": None,"lat_range": None},
    "JPN_2017_CCSBT8" : {"flag": "JPN","year": 2017,"area": "8","inWCPFC": None,"lat_range": None},
    "TWN_2017_CCSBT8" : {"flag": "TWN","year": 2017,"area": "8","inWCPFC": None,"lat_range": None},
    "NZL_2017_CCSBT5" : {"flag": "NZL","year": 2017,"area": "5","inWCPFC": None,"lat_range": None},
    "NZL_2017_CCSBT6" : {"flag": "NZL","year": 2017,"area": "6","inWCPFC": None,"lat_range": None},
    "NZL_2018_CCSBT5" : {"flag": "NZL","year": 2018,"area": "5","inWCPFC": None,"lat_range": None},
    "NZL_2018_CCSBT6" : {"flag": "NZL","year": 2018,"area": "6","inWCPFC": None,"lat_range": None},
    "KOR_2017_CCSBT9" : {"flag": "KOR","year": 2017,"area": "9","inWCPFC": None,"lat_range": None},
    "TWN_2019_WCPFC30": {"flag": "TWN","year": 2019,"area": None,"inWCPFC": True,"lat_range": [-90,-30]},
    "TWN_2020_WCPFC30": {"flag": "TWN","year": 2020,"area": None,"inWCPFC": True,"lat_range": [-90,-30]},
    "JPN_2019_WCPFC30": {"flag": "JPN","year": 2019,"area": None,"inWCPFC": True,"lat_range": [-90,-30]},
    "JPN_2020_WCPFC30": {"flag": "JPN","year": 2020,"area": None,"inWCPFC": True,"lat_range": [-90,-30]},
    "NZL_2019_WCPFC30": {"flag": "NZL","year": 2019,"area": None,"inWCPFC": True,"lat_range": [-90,-30]},
    "NZL_2020_WCPFC30": {"flag": "NZL","year": 2020,"area": None,"inWCPFC": True,"lat_range": [-90,-30]}    
}

df_night_ratio = []
measure_ratio = "sets"  # Either "sets" or "duration"
for key in list(areas.keys()):
    print(areas[key]["flag"])
    mostly_night, only_night, num_sets = perc_night(
        df_0,
        flag=areas[key]["flag"],
        year=areas[key]["year"],
        region=areas[key]["area"],
        lat_range=areas[key]["lat_range"],
        inWCPFC=areas[key]["inWCPFC"],
        mtype=measure_ratio,
    )
    mostly_night1, only_night1, num_sets1 = perc_night(
        df_60,
        flag=areas[key]["flag"],
        year=areas[key]["year"],
        region=areas[key]["area"],
        lat_range=areas[key]["lat_range"],
        inWCPFC=areas[key]["inWCPFC"],
        mtype=measure_ratio,
    )

    print(areas[key]["flag"], areas[key]["year"])
    print("mostly_night: ", round(mostly_night, rnd))
    print("only_night: ", round(only_night, rnd))
    df_night_ratio.append(
        [
            areas[key]["flag"],
            areas[key]["year"],
            areas[key]["area"],
            areas[key]["inWCPFC"],
            areas[key]["lat_range"],
            only_night,
            only_night1,
            mostly_night,
            mostly_night1,
            num_sets,
            num_sets1,
        ]
    )

df_results = pd.DataFrame(df_night_ratio, columns=df_columns)

print("ratio", measure_ratio)
df_results.round(0)

print("ratio", measure_ratio)
df_results.round(1)


