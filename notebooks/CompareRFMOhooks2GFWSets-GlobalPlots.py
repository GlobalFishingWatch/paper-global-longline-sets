# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Compare RFMO Hooks with GFW Hooks

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# %matplotlib inline

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rc("legend", frameon=False)

# +
import seaborn as sns

sns.set_theme()
import pyseas.maps as psm
import pyseas.contrib as psc
import pyseas.cm
import matplotlib.colors as mpcolors
from pyseas.maps import bivariate
import sys

sys.path.append("../data/")

import Regions_json
from shapely.geometry import shape, Point, Polygon

regions_json = {}
regions_json["South_Pacific"] = shape(
    json.loads(Regions_json.South_Pacific.replace("'", '"'))
)
regions_json["North_Pacific"] = shape(
    json.loads(Regions_json.North_Pacific.replace("'", '"'))
)
regions_json["South_Atlantic"] = shape(
    json.loads(Regions_json.South_Atlantic.replace("'", '"'))
)
regions_json["South_Indian"] = shape(
    json.loads(Regions_json.South_Indian.replace("'", '"'))
)

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
   `global-fishing-watch.paper_global_longline_sets.sets_5min_cat_v20220701` 
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
    `world-fishing-827.gfw_research.vi_ssvid_v20210913` 
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
From `world-fishing-827.birdlife.rfmo_hooks_v20211206` )
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

df = pd.read_gbq(q)


# + str(df_temp.start_time.dt.year.max())
def plot_hooks_sets(
    df_temp,
    scale=5,
    region=None,
    regions_json=regions_json,
    proj="global.pacific_centered",
    flag="All",
    y_axis="sets",
    max_y=5000,
    min_y=500,
    max_x=5.0,
):
    if region is not None:
        df_temp = df_temp[(df_temp.region == region) | (df_temp.region_CCSBT == region)].reset_index(drop=True).copy()


    grid_sets = psm.rasters.df2raster(
        df_temp,
        "lon_index",
        "lat_index",
        y_axis,
        origin="upper",
        xyscale=scale,
        #         per_km2=True,
    )
    grid_hooks = psm.rasters.df2raster(
        df_temp,
        "lon_index",
        "lat_index",
        "hooks_diff",
        origin="upper",
        xyscale=scale,
        #         per_km2=True,
    )

    cmap = bivariate.TransparencyBivariateColormap(pyseas.cm.misc.blue_orange)
    with psm.context(psm.styles.dark):
        fig, (ax0) = psm.create_maps(
            1, 1, figsize=(10, 10), dpi=300, facecolor="white", projection=proj
        )

        norm1 = mpcolors.Normalize(vmin=0.0, vmax=max_x, clip=True)
        norm2 = mpcolors.LogNorm(vmin=min_y, vmax=max_y, clip=True)
        grid_sets[grid_sets < 0.001] = np.nan

        bivariate.add_bivariate_raster(
            grid_hooks, grid_sets, cmap, norm1, norm2, ax=ax0
        )
        psm.add_land(ax0)
        cb_ax = bivariate.add_bivariate_colorbox(
            cmap,
            norm1,
            norm2,
            ylabel= y_axis,
            xlabel="hooks/sets (1000)",
            fontsize=8,
            loc=(0.6, -0.17),
            aspect_ratio=3.0,
            xformat="{x:.2f}",
            yformat="{x:.2f}",
            fig=fig,
            ax=ax0,
        )
        if region is None:
            for key in regions_json.keys():
                lons, lats = np.array(regions_json[key].exterior.coords.xy)
                psm.add_plot(lons, lats, ax=ax0)
        else:
            if region in regions_json:
                lons, lats = np.array(regions_json[region].exterior.coords.xy)
                psm.add_plot(lons, lats, ax=ax0)
        gl = pyseas.maps.add_gridlines()
        title_string = str(region) if region else ""
        title_string += "Longline sets and hooks: "
        title_string += " "
        title_string += flag if flag else ""
        title_string += " "

        ax0.set_title(title_string, pad=10, fontsize=20)

        if (region is None) or (region == "Other_Region") or (region == "Other_Region_CCSBT"):
            for key in regions_json.keys():
                lons, lats = np.array(regions_json[key].exterior.coords.xy)
                psm.add_plot(lons, lats)
        else:
            if region in regions_json:
                lons, lats = np.array(regions_json[region].exterior.coords.xy)
                psm.add_plot(lons, lats)


# -
def plot_hooks_sets_month(df_temp, iso3, hooks_per_set, by_flag=1):
    years = np.sort(df_temp.year.unique())
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    months = np.sort(df_temp.month.unique())
    regions = np.sort(df_temp.region.unique())
    fig, axs = plt.subplots(len(regions), 1, sharex=False, figsize=(25, 25))
    fig.suptitle("Hooks vs Sets by Month: " + iso3, fontsize=22, y=0.92)

    for r, region in enumerate(regions):
        num_sets = []
        num_hooks = []
        x_months = []
        for y, year in enumerate(years):

            for month in months:
                if (
                    len(
                        df_temp[
                            (df_temp.region == region)
                            & (df_temp.year == year)
                            & (df_temp.month == month)
                        ]
                    )
                    > 0
                ):
                    if month_names[month - 1] == month_names[0]:
                        x_months.append(str(year) + " " + str(month_names[month - 1]))
                    else:
                        x_months.append(str(month_names[month - 1]))
                    if(by_flag==1):
                        iso3_cond = (df_temp.iso3 == iso3)

                    else:
                        iso3_cond = 1                                
                    d2 = (
                        df_temp[
                            (df_temp.region == region)
                            & iso3_cond
                            & (df_temp.year == year)
                            & (df_temp.month == month)
                        ]
                        .groupby(["lon_index", "lat_index"])
                        .sum()
                        .reset_index()
                    )

                    num_sets.append(d2.sets.sum())
                    num_hooks.append(d2.hooks.sum() / hooks_per_set[region])

        df = pd.DataFrame(
            {"sets": num_sets, "hooks / " + str(hooks_per_set[region]): num_hooks},
            index=x_months,
        )
        df.plot.line(rot=0, ax=axs[r], color=["red", "green"])
        axs[r].set_title(region, fontsize=20)
        axs[r].tick_params(labelrotation=90, axis="x")
        axs[r].set_xticks(np.arange(0, len(x_months)))
        axs[r].set_xticklabels(x_months, rotation=90, fontsize=16)
        plt.ylabel("Number of sets", fontsize=16)
        axs[r].legend()

    fig.subplots_adjust(hspace=0.7)

    plt.show()


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
    print(
        iso3,
        "Hooks",
        d2["hooks"].sum(),
        "Sets",
        d2["sets"].sum(),
        "hooks to sets: ",
        hooks_to_sets,
    )
print(" ")
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
        print(
            iso3,
            "Hooks",
            d2["hooks"].sum(),
            "Sets",
            d2["sets"].sum(),
            "hooks to sets: ",
            hooks_to_sets,
        )
    print("  ")
# -

hooks_by_region

d2 = df2.groupby(["lon_index", "lat_index"]).sum().reset_index()
d2["hooks_diff"] = (d2.hooks / d2.sets) / 1000
d2["hooks|sets"] = d2.sets
d2.loc[d2.sets == 0, "hooks|sets"] = (
    d2[d2.sets == 0].hooks / hooks_by_region["Global"]
)
plot_hooks_sets(
    d2,
    scale=0.2,
    y_axis="hooks|sets",
    max_y=5000,
    min_y=50,
    proj="global.atlantic_centered",
    max_x=10,
)
# plt.savefig("Global_hooks_sets.png",dpi=300, bbox_inches='tight')

top_flags = df2.iso3.value_counts()[0:5].index
for iso3 in top_flags:
    d2 = df2[(df2.iso3 == iso3)].groupby(["lon_index", "lat_index"]).sum().reset_index()
    d2["hooks_diff"] = (d2.hooks / d2.sets) / 1000
    d2["hooks|sets"] = d2.sets
    d2.loc[d2.sets == 0, "hooks|sets"] = (
        d2[d2.sets == 0].hooks / hooks_by_region["Global"]
    )

    plot_hooks_sets(
        d2,
        scale=0.2,
        y_axis="hooks|sets",
        max_y=5000,
        min_y=10,        
        proj="global.atlantic_centered",
        flag=iso3,
        max_x=10,
    )

plot_hooks_sets_month(df2, "Global", hooks_by_region, by_flag=False)

top_flags = df2.iso3.value_counts()[0:5].index
for iso3 in top_flags:
    plot_hooks_sets_month(df2, iso3, hooks_by_region)




