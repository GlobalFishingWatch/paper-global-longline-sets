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

# # Analyse the longline sets data
#
# This notebook produces Figure 2, number of vessels, number of sets, and set durations. Figure 2 shows the fraction of day and night positions on a global plot.
#
# Figure Caption: "Day setting dominates almost everywhere in the ocean. Blue areas indicate that sets happen mostly at night, and orange indicates sets occur mostly during the day. Bounding boxes represent regions with tRFMO regulations in the South Indian Ocean, North Pacific, South Pacific, and South Atlantic. "
#
# "For the period between January 2017 and December 2020, we classified 1,451,159 sets globally from 4923 vessels. "
#
# "The average duration of a set in our data (6.5 ± 1.5 hours)"

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import cartopy

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

figures_folder = '../outputs/figures'


def plot_positions_biv(
    df_n,
    df_d,
    scale=10,
    region=None,
    proj=cartopy.crs.EqualEarth(central_longitude=23-180),
    flag=None,
    title_string="fishing positions",
    max_x=1.0,
    max_y=5.0,    
):

    grid_total = psm.rasters.df2raster(
        df_d,
        "lon_index",
        "lat_index",
        "positions",
        origin="upper",
        xyscale=scale,
        per_km2=True,
    )
    grid_day = psm.rasters.df2raster(
        df_n,
        "lon_index",
        "lat_index",
        "positions",
        origin="upper",
        xyscale=scale,
        per_km2=True,
    )

    grid_day_ratio = np.empty_like(grid_day)
    grid_day_ratio.fill(np.nan)
    np.divide(grid_day, grid_total, out=grid_day_ratio, where=grid_total != 0)

    cmap = bivariate.TransparencyBivariateColormap(pyseas.cm.misc.blue_orange)
    with psm.context(psm.styles.light):
        fig, (ax0) = psm.create_maps(
            1, 1, figsize=(10, 10), dpi=300, facecolor="white", projection=proj
        )

        norm1 = mpcolors.Normalize(vmin=0.0, vmax=max_x, clip=True)
        norm2 = mpcolors.LogNorm(vmin=0.05, vmax=max_y, clip=True)
        grid_total[grid_total < 0.001] = np.nan

        bivariate.add_bivariate_raster(
            grid_day_ratio, grid_total, cmap, norm1, norm2, ax=ax0
        )
        psm.add_land(ax0)
        cb_ax = bivariate.add_bivariate_colorbox(
            cmap,
            norm1,
            norm2,
            xlabel="fraction of daytime fishing positions",
            ylabel="total fishing positions",
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

        ax0.set_title(title_string, fontsize=16)

        plt.subplots_adjust(hspace=0.05)
        plt.tight_layout()
        plt.savefig(figures_folder + '/positions_bivariate_light.png', dpi=300, bbox_inches='tight')

        plt.show()


# %load_ext autoreload
# %autoreload 2

# # Get sets

# +
q = """
     

with sets_table as (
  select 
    start_time,
    end_time,
    ABS(TIMESTAMP_DIFF(end_time, start_time, minute))/60 AS set_duration,    
    --st_centroid(st_union_agg(st_geogpoint(lon,lat ))) c,
    st_x(st_centroid(st_union_agg(st_geogpoint(lon,lat )))) as lon,
    st_y(st_centroid(st_union_agg(st_geogpoint(lon,lat )))) as lat,
    ssvid, 
    cat2
  from 
   `birdlife.sets_5min_cat_v20220701`
  group by 
    ssvid, start_time, end_time, cat2
)

Select * from sets_table

"""

df = pd.read_gbq(q)
# -
# ## Limit years to 2017-2020 

df = df[(df.start_time.dt.year>=2017) & (df.end_time.dt.year<=2020)].copy().reset_index(drop=True)

# +

night_cats = [2, 5, 7]
day_cats = [1, 6, 8]
df["mostly_day"] = df.cat2.isin(day_cats)
df["mostly_night"] = df.cat2.isin(night_cats)
# -

min_dur = 2
max_dur = 15
df = (
    df[(df.set_duration >= min_dur) & (df.set_duration <= max_dur)]
    .reset_index()
    .copy()
)

ax = df.set_duration.hist(bins=20)
plt.suptitle('Sets Duration Histogram', x=0.5, y=1.0, ha='center', fontsize='xx-large')
ax.set_ylabel("number of sets")
ax.set_xlabel("hours")
plt.savefig(figures_folder + '/Sets_Duration_Histogram.png', dpi=300, bbox_inches='tight')

print("mean duration: ", round(df.set_duration.mean(),1))
print("std: ", round(df.set_duration.std(),1))

# ## Number of sets by year

df.start_time.dt.year.value_counts().sort_index()

# ## Total number of sets 

len(df)

# ## Total number of vessels 

len(df.ssvid.unique())

print("Number night sets",len(df[df.cat2==2]))
print("Number mostly night sets",len(df[df.mostly_night]))
print("Percentage night sets",round(len(df[df.cat2==2])/len(df)*100,1))
print("Percentage mostly night sets",round(len(df[df.mostly_night])/len(df)*100,1))

# # Plot global longline fishing positions (Fig 2)

# +
q = """
select count(*) positions, floor(lon*10) lon_index, floor(lat*10) lat_index, day_category
from `birdlife.sets_5min_cat` 
where EXTRACT(YEAR from timestamp) >= 2017
AND EXTRACT(YEAR from timestamp) <= 2020
group by day_category, lon_index, lat_index 
"""

df_grid = pd.read_gbq(q)
# -

df_daylight = df_grid[df_grid.day_category.isin(["day","dusk","dawn"])].copy().reset_index() 

plot_positions_biv(df_daylight,df_grid, title_string="Longline fraction of daytime setting positions : 2017-2020")

# # Average pings per hour for predicted sets


# +
#. This query references gfw_research.pipe_v20201001_fishing which is not a public table

q = f"""
WITH
  ----------------------------------------
  -- Get events, compute set duration, and
  -- add a unique id for each set
  ----------------------------------------

  events AS (
  SELECT
    * EXCEPT (id),
    id AS mmsi,
    ABS(TIMESTAMP_DIFF(end_time, start_time, minute))/60 AS set_duration,
    ROW_NUMBER() OVER (ORDER BY id, start_time) AS set_id,
  FROM
    `global-fishing-watch.paper_global_longline_sets.longline_events_smoothv20220801_*`
  WHERE
     label = "setting"),
  
  fishing_data as (
    SELECT
    ssvid, 
    seg_id,
    timestamp,
    DATE(timestamp) AS date_ymd,
    hours
  FROM
    `gfw_research.pipe_v20201001_fishing`
  WHERE
    (_partitiontime BETWEEN TIMESTAMP("2017-01-01")
      AND TIMESTAMP("2021-01-01")) 

  ),

 joined_pipe AS (
  SELECT
    distinct
    b.seg_id,
    date_ymd,
    COUNT(timestamp) AS pings_in_seg,
    SUM(hours) AS sum_hours,
  FROM
    events a
   JOIN
    fishing_data b
  ON
    (mmsi = ssvid
    AND timestamp BETWEEN start_time
    AND end_time)
    group by seg_id, date_ymd)

SELECT
  distinct
  seg_id,
  date_ymd,
  pings_in_seg/sum_hours AS avg_pings_per_h
FROM
  joined_pipe
WHERE
  sum_hours > 0
  AND pings_in_seg > 0"""

df_seg = pd.read_gbq(q)
# -

print("average pings per hour: ",round(df_seg.avg_pings_per_h.mean()))
print("median pings per hour: ",round(df_seg.avg_pings_per_h.median()))      




