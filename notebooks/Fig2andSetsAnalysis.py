# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Analyse the longline sets data

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

figures_folder = '../outputs/figures'


def plot_day_biv(
    df_temp, scale=10, region=None, proj="global.pacific_centered", flag=None
):
    if region is not None:
        df_temp = df_temp[df_temp.region == region].reset_index(drop=True).copy()
    if flag is not None:
        df_temp = df_temp[df_temp["best_flag"] == flag].reset_index(drop=True).copy()
    #     scale = 10
    #     if('lon_index' in df_temp):
    #         df_temp.drop('lon_index', inplace=True, axis=1)
    #         df_temp.drop('lat_index', inplace=True, axis=1)
    df_temp["lon_index"] = np.floor(df_temp["lon"] * scale)
    df_temp["lat_index"] = np.floor(df_temp["lat"] * scale)
    df_temp_gridded = (
        df_temp[(df_temp.set_duration >= min_dur) & (df_temp.set_duration <= max_dur)][
            ["lon_index", "lat_index"]
        ]
        .groupby(["lon_index", "lat_index"])
        .size()
        .reset_index(name="set_counts")
    )

    df_temp_gridded_day = (
        df_temp[
            (df_temp.set_duration >= min_dur)
            & (df_temp.set_duration <= max_dur)
            & (df_temp.cat2.isin([1, 6, 8]))
        ][["lon_index", "lat_index"]]
        .groupby(["lon_index", "lat_index"])
        .size()
        .reset_index(name="set_counts")
    )

    grid_total = psm.rasters.df2raster(
        df_temp_gridded,
        "lon_index",
        "lat_index",
        "set_counts",
        origin="upper",
        xyscale=scale,
        per_km2=True,
    )
    grid_day = psm.rasters.df2raster(
        df_temp_gridded_day,
        "lon_index",
        "lat_index",
        "set_counts",
        origin="upper",
        xyscale=scale,
        per_km2=True,
    )

    grid_day_ratio = np.empty_like(grid_day)
    grid_day_ratio.fill(np.nan)
    np.divide(grid_day, grid_total, out=grid_day_ratio, where=grid_total != 0)

    cmap = bivariate.TransparencyBivariateColormap(pyseas.cm.misc.blue_orange)
    with psm.context(psm.styles.light):
        fig, (ax0) = psm.create_maps(1, 1, figsize=(10, 10), dpi=300, facecolor="white",projection=proj)

        norm1 = mpcolors.Normalize(vmin=0.0, vmax=1, clip=True)
        norm2 = mpcolors.LogNorm(vmin=0.001, vmax=0.1, clip=True)
        grid_total[grid_total < 0.001] = np.nan

        bivariate.add_bivariate_raster(
            grid_day_ratio, grid_total, cmap, norm1, norm2, ax=ax0
        )
        psm.add_land(ax0)
        cb_ax = bivariate.add_bivariate_colorbox(
            cmap,
            norm1,
            norm2,
            xlabel="day sets\n(as fraction of total sets)",
            ylabel="total sets\n(per $\mathregular{km^2}$)",
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
        title_string = region if region else ""
        title_string += "Longline sets day to night ratio: "
        title_string += region if region else ""
        title_string += " "
        title_string += flag if flag else ""
        title_string += " "
        title_string += (
            str(df_temp.start_time.dt.year.min())
            + "-"
            + str(df_temp.start_time.dt.year.max())
        )
        ax0.set_title(title_string, fontsize=16)

        plt.subplots_adjust(hspace=0.05)
        plt.tight_layout()
        plt.savefig(figures_folder + '/sets_bivariate_day_vs_night_light.png', dpi=300, bbox_inches='tight')

        plt.show()


# %load_ext autoreload
# %autoreload 2

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

df.isna().sum()

# ## Limit years to 2017-2020 

df = df[(df.start_time.dt.year>=2017) & (df.end_time.dt.year<=2020)].copy().reset_index(drop=True)

# +

night_cats = [2, 5, 7]
day_cats = [1, 6, 8]
df["mostly_day"] = df.cat2.isin(day_cats)
df["mostly_night"] = df.cat2.isin(night_cats)
# -

df.set_duration.hist(bins=20)

min_dur = 2
max_dur = 15
df = (
    df[(df.set_duration >= min_dur) & (df.set_duration <= max_dur)]
    .reset_index()
    .copy()
)

# ##

ax = df.set_duration.hist(bins=20)
plt.suptitle('Sets Duration Histogram', x=0.5, y=1.0, ha='center', fontsize='xx-large')
ax.set_ylabel("number of sets")
ax.set_xlabel("hours")
plt.savefig(figures_folder + '/Sets_Duration_Histogram.png', dpi=300, bbox_inches='tight')

round(sum(df.set_duration>8)/len(df),2)

round(sum(df.set_duration>10)/len(df),2)

print("mean duration: ", round(df.set_duration.mean(),1))
print("std: ", round(df.set_duration.std(),1))

# ## Number of sets by year

df.start_time.dt.year.value_counts().sort_index()

# ## Total number of sets 

len(df[df.start_time.dt.year<=2020])

# ## Total number of vessels 

len(df.ssvid.unique())

plot_day_biv(df, scale=5)

plot_day_biv(df, scale=5)

len(df)



print("Number night sets",len(df[df.cat2==2]))
print("Number mostly night sets",len(df[df.mostly_night]))
print("Percentage night sets",round(len(df[df.cat2==2])/len(df)*100,1))
print("Percentage mostly night sets",round(len(df[df.mostly_night])/len(df)*100,1))







