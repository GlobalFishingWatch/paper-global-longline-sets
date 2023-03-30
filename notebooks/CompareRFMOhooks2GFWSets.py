# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Compare RFMO Hooks with GFW Hooks
#
# This notebook produces the ratio of hooks to sets in `Assessing model accuracy`. RFMO data on hooks was downloaded from each RFMO’s website, except for ICCAT, which was obtained from direct correspondence. 
#
# `Assessing model accuracy`: "To determine if our dataset on long- line sets was representative of all longline activity, we compared our longline set data from AIS with hooks reported to the tRFMOs. Fishing effort using longlines is typically reported by Flag States to tRFMOs as the aggregate number of hooks deployed in an area. Dividing the num- ber of reported hooks reported between 2017 and 2019 by the number of detected longline sets yielded a ratio of ≈3300 hooks per set. This ra- tio is higher than the actual number of hooks per set, on average, largely because we detect longlines set by vessels that transmit AIS, and an unknown number of vessels are not broadcasting AIS. Nonetheless, although the number of hooks per set varies by vessel and set, it typi- cally ranges between 1000 and 4000 hooks per set (Bigelow et al., 2006; Dunn et al., 2008; Nieblas et al., 2019). Given our ratio is in this range, it suggests that our model has likely captured a large proportion, if not the majority, of longline activity within our regions of interest. "

# +
import numpy as np
import pandas as pd



# +

import sys

sys.path.append("../data/")

# -

# %load_ext autoreload
# %autoreload 2

# +
q = '''

CREATE TEMPORARY FUNCTION
  find_ROI(lat float64,lon float64) AS (
    CASE
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[113,23],[260.6,23],[260.6,66],[113,66],[113,23]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "North_Pacific"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[150.0, -60.0], [150.0, -55.0], [141.0, -55.0], [141.0, -30.0], [150.0, -30.0], [160.0, -30.0], [240.0, -30.0], [240.0, -60.0], [150.0, -60.0]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Pacific"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[-69.7, -23.0], [25, -23.0], [25, -62.3], [-69.7, -62.3], [-69.7, -23.0]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Atlantic"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[25, -23.0], [118.0, -23.0], [118.0, -58.5], [25, -58.5], [25, -23.0]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Indian"                 
    ELSE
    "Other_Region"
  END
    ); 
        
CREATE TEMPORARY FUNCTION
  find_region_CCSBT(lat float64,lon float64) AS (
    CASE
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[100,-10],[130,-10],[130,-20],[100,-20],[100,-10]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "1"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[80,-20],[120,-20],[120,-35],[80,-35],[80,-20]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "2"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[120,-20],[140,-20],[140,-35],[120,-35],[120,-20]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "3"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[140,-30],[170,-30],[170,-40],[140,-40],[140,-30]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "4"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[170,-30],[190,-30],[190,-40],[170,-40],[170,-30]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "5"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[160,-40],[190,-40],[190,-60],[160,-60],[160,-40]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "6"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[120,-35],[140,-35],[140,-40],[160,-40],[160,-60],[120,-60],[120,-35]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "7"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[60,-35],[120,-35],[120,-60],[60,-60],[60,-35]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "8"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[-20,-35],[60,-35],[60,-60],[-20,-60],[-20,-35]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "9"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[-70,-35],[-20,-35],[-20,-60],[-70,-60],[-70,-35]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "10"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[20,-20],[80,-20],[80,-35],[20,-35],[20,-20]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "14"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON("{'type': 'Polygon','coordinates': [[[-10,-20],[20,-20],[20,-35],[-10,-35],[-10,-20]]]}",make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "15"
    ELSE
    "Other_Region"
  END
    ); 
with 

sets_table as (
  select 
    start_time,
    ssvid, 
    lat,
    lon,
    ABS(TIMESTAMP_DIFF(end_time, start_time, minute))/60 AS set_duration, 
  from 
   `birdlife.sets_5min_cat_v20220701` 
  where 
    extract(year from _partitiontime) in(2017,2018, 2019)
),

sets_table_g as (
  select 
    start_time,
    st_centroid(st_union_agg(st_geogpoint(lon,lat ))) c,
    ssvid, 
  from 
   sets_table
  where 
   (set_duration >= 2 AND set_duration <= 15)
  group by 
    ssvid, start_time
),

vessel_info as (
  select 
    ssvid, 
    best.best_flag flag 
  from 
    `gfw_research.vi_ssvid_v20210913` 
),


sets_grouped as (
  select 
    floor(st_x(c)/5) as lon_index,
    floor(st_y(c)/5) as lat_index,
    flag as iso3,
    extract(year from start_time) year,
    extract(month from start_time) month,
    count(*) sets
  from 
    sets_table_g
  left join 
    vessel_info
  using(ssvid)
  group by 
    lon_index,lat_index, flag,year, month),
  
hooks_table as (select * except(iso3), CASE iso3 WHEN 'TAI' THEN 'TWN' ELSE iso3 END As iso3 
From `birdlife.rfmo_hooks_v20211206` )
select 
  source,
  year,
  month,
  iso3,
  ifnull(hooks,0) hooks,
  ifnull(sets, 0) sets,
  lat_index,
  lon_index,
  find_ROI(lat_index*5, lon_index*5) as region,
  find_region_CCSBT(lat_index*5, lon_index*5) as region_CCSBT,  
  
from 
  sets_grouped
full outer join
  hooks_table
using
  (year, month, lat_index, lon_index, iso3)'''

# This query accesses a private BigQuery table. The results, though,
# are saved in the csv file in the folder saved_dataframes
# uncomment the below to run
# df = pd.read_gbq(q)
# df.to_csv("saved_dataframes/gridded_sets_rfmo.csv", index=False)
# -

df = pd.read_csv('saved_dataframes/gridded_sets_rfmo.csv')

# ## Add ESP hooks for ICCAT

csv_file = "../data/EffDis.csv"
df_csv = pd.read_csv(csv_file, index_col=None, header=0)

# +
esp_flags = df_csv[df_csv.FleetCode.str.slice(stop=6)=="EU.ESP"].FleetCode.unique()
years = df.year.unique()
# years = [2017,2018,2019]

# map quarter to start month
q_map = {1:1,2:4,3:7,4:10}
df_esp = df_csv[(df_csv.FleetCode.isin(esp_flags)) & (df_csv.YearC.isin(years))].copy()

effort_type = "EstEffort"
df_esp["hooks"] = df_esp[effort_type]
df_esp["lon_index"] = [np.floor(x/5) for x in df_esp.xLon5.values]
df_esp["lat_index"] = [np.floor(x/5) for x in df_esp.yLat5.values]
df_esp["month"] = [q_map[q] for q in df_esp.ByQuarter.values]
df_esp["iso3"] = "ESP"
df_esp["source"] = "iccat"
df_esp["year"] = df_esp["YearC"] 
# -

# ## hooks by month 
# ### Hooks are by quarter, so assign average for each month in quarter

df_esp2 = df_esp.copy()
for idx, row in df_esp.iterrows():
    for i in [1,1]:
        row["month"] = row["month"]+i
        df_esp2 = df_esp2.append(row, ignore_index=True)
df_esp2.hooks=df_esp2.hooks/3

print(df_esp2.hooks.sum())
print(df_esp.hooks.sum())

# ## Add ESP sets and regions to main dataframe

for idx, row in df_esp2.iterrows():
    df_sets = df[
        (df.lat_index == row.lat_index)
        & (df.lon_index == row.lon_index)
        & (df.iso3 == "ESP")
        & (df.month == row.month)
        & (df.year == row.year)
    ].copy()
    sets = 0
    if len(df_sets):
        assert len(df_sets["sets"].unique())==1
        sets = df_sets["sets"].values[0]
    df_esp2.loc[idx, "sets"] = sets

    df_reg = df[(df.lat_index == row.lat_index) & (df.lon_index == row.lon_index)]
    df_esp2.loc[idx, "region"] = df_reg["region"].values[0]
    df_esp2.loc[idx, "region_CCSBT"] = df_reg["region_CCSBT"].values[0]

df = df.append(df_esp2, ignore_index=True)

# ## Clean data: remove null locations and keep max hooks number where there are duplicate entries

# +
df.iso3 = df.iso3.fillna("unknown")
df = df[df.year > 2016].copy()
# Remove null lat / lons
df = (
    df[(~df.lat_index.isnull()) | (~df.lon_index.isnull())]
    .copy()
    .reset_index(drop=True)
)

# Handle hook data overlaps
# Sort by hooks descending for lat, lon, year, month, iso3
# to keep the max hooks for that grid location
df2 = (
    df.sort_values("hooks", ascending=False)
    .drop_duplicates(["lat_index", "lon_index", "year", "month", "iso3"])
    .copy()
    .reset_index()
)
# -

df2.year.unique()


# ## Print number of hooks to sets by flag and region, and globally 

def round_hooks(x, base=100):
    return int(base * round(float(x)/base))


# +
d1 = df2.groupby(["lon_index", "lat_index"]).sum().reset_index()
hooks_by_region = {}
print("Global hooks to sets: ", round(d1.hooks.sum() / d1.sets.sum()))
print("  ")
hooks_by_region["Global"] = round_hooks(d1.hooks.sum() / d1.sets.sum(),100)

top_flags = df2.iso3.value_counts()[0:15].index
for iso3 in top_flags:

    d2 = (
        df2[(df2.iso3 == iso3)]
        .groupby(["lon_index", "lat_index"])
        .sum()
        .reset_index()
    )
    hooks_to_sets = 0
    if (d2.hooks.sum() > 0) & (d2.sets.sum() > 0):
        hooks_to_sets = round(d2.hooks.sum() / d2.sets.sum())
#     print(
#         iso3,
#         "Hooks",
#         d2["hooks"].sum(),
#         "Sets",
#         d2["sets"].sum(),
#         "hooks to sets: ",
#         hooks_to_sets,
#     )
# print(" ")
for region in df.region.unique():
    print(region)
    d1 = (
        df2[(df2.region == region)]
        .groupby(["lon_index", "lat_index"])
        .sum()
        .reset_index()
    )
    top_flags = df2[df2.region == region].iso3.value_counts()[0:10].index
    print("Total hooks to sets: ", round(d1.hooks.sum() / d1.sets.sum()))
    hooks_by_region[region] = round_hooks(d1.hooks.sum() / d1.sets.sum(),100)
    for iso3 in top_flags:

        d2 = (
            df2[(df2.region == region) & (df2.iso3 == iso3)]
            .groupby(["lon_index", "lat_index"])
            .sum()
            .reset_index()
        )
        hooks_to_sets = 0
        if (d2.hooks.sum() > 0) & (d2.sets.sum() > 0):
            hooks_to_sets = round(d2.hooks.sum() / d2.sets.sum())
#         print(
#             iso3,
#             "Hooks",
#             d2["hooks"].sum(),
#             "Sets",
#             d2["sets"].sum(),
#             "hooks to sets: ",
#             hooks_to_sets,
#         )
#     print("  ")
# -

hooks_by_region


