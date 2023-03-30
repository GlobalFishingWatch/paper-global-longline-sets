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

# # Figure 1 and Estimates of Total Night Sets
#
# This notebook produces `Figure 1: Longline Sets that started on 2020-05-01`, estimates for a given day and a year how much of the ocean is within 30km of a set, and fraction of sets entirely at night
#
# Results: "On any given day (one day shown in Fig. 1a), there were around 1000 (997 ± 125) sets in the global ocean by vessels broadcasting AIS. Albatross with radio tags have redirected towards fishing vessels up to 30 km away, suggesting that they can detect fishing vessels at this distance, which is also the limit of their visual range (Collet et al., 2015, 2017). Considering this range, we measured the area of the ocean within 30 km of a setting longline during the night, dawn, day, and dusk. We find that on an aver- age day, about 5.3 million km2, or about 1.5 % of the ocean, is within 30 km of a set, and this number varied between 3.1 and 6.5 million km2 for different days in our four year time period. Over the course of a year, about 146 million km2, or over 40 % of the ocean, is within this distance of a set, and 38 %, or 137 million km2 is within this distance to a vessel setting during the day. 
# "
#
# Figure caption: " (a) One day of longline sets in the global ocean. Bounding boxes represent regions with tRFMO regulations: South Indian Ocean, North Pacific, South Pacific, and South Atlantic. Shown are all longline sets (1166 sets) that started on 1 May 2020. (b) A zoomed in region (red box on a) shows 75 sets and the time of day of the different parts of the set (night, dawn, day — for these sets there was no overlap with dusk). In this region, virtually all sets started before dawn and continued into the day. The area within 30 km of the set, the distance an albatross can detect a vessel, is shown in gray."
#
# "because vessels often set in groups, as can be seen in `figure 1b`, where all the longlines appear to be set at the same time of day; setting in groups may reduce the chance of interference with one another. "

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import matplotlib.gridspec as gridspec
import skimage.io
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
from datetime import datetime

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import pyseas.maps as psm
import pyseas.contrib as psc
import pyseas.cm

import matplotlib.cm as cm
import json

from shapely.geometry import shape

import cartopy

# %matplotlib inline

import matplotlib as mpl

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rc('legend',fontsize='12') 
plt.rc('legend',frameon=False)

# +
# # geom = shape(data)
# with open("../queries/Regions_json_longlines.txt", 'r') as f:
#     Regions_json = f.read()

# print(Regions_json )

# -

North_Pacific = '{"type": "Polygon","coordinates": [[[113,23],[260.6,23],[260.6,66],[113,66],[113,23]]]}'
South_Pacific = '{"type": "Polygon","coordinates": [[[150.0, -60.0], [150.0, -55.0], [141.0, -55.0], [141.0, -30.0], [150.0, -30.0], [160.0, -30.0], [240.0, -30.0], [240.0, -60.0], [150.0, -60.0]]]}'
South_Atlantic = '{"type": "Polygon","coordinates": [[[-69.7, -23.0], [25, -23.0], [25, -62.3], [-69.7, -62.3], [-69.7, -23.0]]]}'
South_Indian = '{"type": "Polygon","coordinates": [[[25, -23.0], [118.0, -23.0], [118.0, -58.5], [25, -58.5], [25, -23.0]]]}'

regions_json = {}
regions_json["North_Pacific"] = shape(json.loads(North_Pacific.replace("\'", "\"")))
regions_json["South_Atlantic"] = shape(json.loads(South_Atlantic.replace("\'", "\"")))
regions_json["South_Indian"] = shape(json.loads(South_Indian.replace("\'", "\"")))
regions_json["South_Pacific"] = shape(json.loads(South_Pacific.replace("\'", "\"")))

# # Plot One Day Of Sets

# +
# how many sets started on 2020-05-01?

q = '''select count(*) as num_sets from (
      (select distinct CONCAT(ssvid, CAST(start_time AS string)) as set_id 
    from 
      `birdlife.sets_5min_cat_v20220701` 
    where date(start_time) = "2020-05-01" )
)'''
pd.read_gbq(q)
# -


q = '''
with sets_starting_onday as (
      (select CONCAT(ssvid, CAST(start_time AS string)) as set_id 
    from 
      `birdlife.sets_5min_cat_v20220701` 
    where date(start_time) = "2020-05-01" )
)

select lat_index, lon_index,day_category, count(distinct set_id) sets 
from 
  `global-fishing-watch.paper_global_longline_sets.sets_grid_40_v20220701`
join 
  sets_starting_onday
using(set_id)
where 
  date(_partitiontime) between "2020-05-01" and "2020-05-02" 
  group by  lat_index, lon_index,day_category

'''
df_s1d = pd.read_gbq(q)





def map_raster2(raster, norm, minvalue = 1e-2,
                  colorbar_label = r"hours of gaps  per 1000 $\mathregular{km^2}$ ",
                  title = "Sets",figsize=(20, 20),
               title_font_size = 20, filename='temp.png' ):

    fig = plt.figure(figsize=figsize)
    with pyseas.context(psm.styles.dark):
        with pyseas.context({'text.color' : '#FFFFFF'}):
            ax = psm.create_map(projection='global.pacific_centered')
            im = psm.add_raster(raster, ax=ax,
                            cmap='fishing',
                            norm=norm,
                            origin='lower'
                            )
            psm.add_land()
#             cb = psm.colorbar.add_colorbar(im, label=colorbar_label, 
#                                        loc='bottom', format='%.1f')
#             ax.spines['geo'].set_visible(False)
            ax.set_title(title, pad=10, fontsize=title_font_size, color = 'black' )
#             psm.add_figure_background(color='black')
        plt.savefig(filename, dpi=400, bbox_inches = 'tight')
    plt.show()


# +
bboxes = []

min_lon, min_lat, max_lon, max_lat = 27.6+26,-39.5,59.4+18,-33.0
# bboxes.append([min_lon, min_lat, max_lon, max_lat])
# min_lon, min_lat, max_lon, max_lat = 142.7,31.7,166.5,40.4
# bboxes.append([min_lon, min_lat, max_lon, max_lat])


# min_lon, min_lat, max_lon, max_lat = 2.7,-35.2,23.1,-25.6
# bboxes.append([min_lon, min_lat, max_lon, max_lat])


# +
scale = 40
sets_area = psm.rasters.df2raster(df_s1d,
                               'lon_index', 'lat_index',
                               'sets', xyscale=scale, 
                                per_km2=False, origin = 'lower')

sets_area[sets_area == 0] = np.nan
raster = sets_area
norm = mpcolors.Normalize(vmin=0, vmax=1)
minvalue =0
colorbar_label = "num sets within 30km"
title = f"Longline sets that started on 2020-05-01"
filename=f'sets_all_2021-05-01.png'
figsize=(20, 20)
title_font_size = 20


fig = plt.figure(figsize=figsize)
with pyseas.context(psm.styles.light):
    with pyseas.context({'text.color' : '#FFFFFF'}):
        ax = psm.create_map(projection=cartopy.crs.EqualEarth(central_longitude=23-180))#(central_longitude=0, globe=None))#'global.pacific_centered')
        im = psm.add_raster(raster, ax=ax,
                        cmap='fishing',
                        norm=norm,
                        origin='lower'
                        )
        psm.add_land()
#             cb = psm.colorbar.add_colorbar(im, label=colorbar_label, 
#                                        loc='bottom', format='%.1f')
#             ax.spines['geo'].set_visible(False)
        ax.set_title(title, pad=10, fontsize=title_font_size, color = 'black' )
#             psm.add_figure_background(color='black')
        ax.add_patch(mpatches.Rectangle(xy=[min_lon, min_lat], width=max_lon-min_lon,
                                        height=max_lat-min_lat,
                                        facecolor='none',
                                        alpha=1,
                                        color = 'red',
                                        fill = None,
                                        transform=ccrs.PlateCarree() ))

    
        for key in regions_json.keys():
            lons, lats = np.array(regions_json[key].exterior.coords.xy)
            psm.add_plot(lons, lats)
#     plt.savefig(filename, dpi=300, bbox_inches = 'tight')
plt.show()

# -

start_date = "2020-05-01"
end_date = "2020-05-02"

# +
q = f'''


select 
  ssvid, timestamp, lon, lat, day_category
from 
 `birdlife.sets_5min_cat_v20220701`
where date(_partitiontime) between "{start_date}" and "{end_date}"
    and date(start_time) = "{start_date}"
    and start_time > timestamp("{start_date}") 
    and end_time < ("{end_date} 12:00:00")
    and lon between {min_lon} and {max_lon}
    and lat between {min_lat} and {max_lat}
    order by ssvid, timestamp
    '''
df = pd.read_gbq(q)


extent = min_lon, max_lon, min_lat, max_lat

the_center = [min_lon / 2 + max_lon / 2, min_lat / 2 + max_lat / 2]

projection = cartopy.crs.LambertAzimuthalEqualArea(
    central_latitude=the_center[1], central_longitude=the_center[0]
)


    
# -
sets_area2 = np.copy(sets_area)
sets_area2[sets_area2>1]=1

colors = ["#FBE2BA", "#FFBD52", "#DB8901", "#0573B3", "#39394A", "#81B9D9"]


# +
norm = mpcolors.Normalize(vmin=0, vmax=1.5)

with psm.context(psm.styles.light):
    fig = plt.figure(figsize=(15, 6))
    ax = psm.create_map(projection=projection)
    
    im = psm.add_raster(sets_area2/2, ax=ax,
                cmap=cm.binary, #'fishing',
                norm=norm,
                origin='lower'
                )
    
    psm.add_land()
    d = df[df.day_category == "night"]
    ax.scatter(
        d.lon, d.lat, transform=ccrs.PlateCarree(), s=3, color="#39394A", label="night setting"
    )
    d = df[df.day_category == "dawn"]
    ax.scatter(
        d.lon, d.lat, transform=ccrs.PlateCarree(), s=3, color = "#ff4545", label="dawn setting"
    ) #  color="#FFBD52"
    d = df[df.day_category == "day"]
    ax.scatter(
        d.lon, d.lat, transform=ccrs.PlateCarree(), s=3, color="#FBE2BA" , label="day setting"
    )
#     d = df[df.day_category == "dusk"]
#     ax.scatter(d.lon, d.lat, transform=ccrs.PlateCarree(), s=3, label="dusk setting", color="#DB8901")
    
#     ax.scatter(0,0,s=3,color='#555555',label = "area within 30km of set")
    ax.scatter(-1e10,-1e10,marker="s",s=3,color='grey',label = "area within 30km of set")
    
    ax.set_extent(extent,crs=ccrs.PlateCarree() )
    
    lgnd = plt.legend(loc="lower right", numpoints=1, fontsize=15, frameon= False)

    
    for handle in lgnd.legendHandles:
        handle.set_sizes([30])
        
    psm.add_scalebar(ax=ax,skip_when_extent_large=True)

# +
# count number of unique sets in this region


q = f'''

select 
  count(*) as num_sets 
from (
    select 
      distinct CONCAT(ssvid, CAST(start_time AS string)) as set_id 
    from 
      `birdlife.sets_5min_cat_v20220701`
    where 
      date(_partitiontime) between "{start_date}" and "{end_date}"
      and date(start_time) = "{start_date}"
      and start_time > timestamp("{start_date}") 
      and end_time < ("{end_date} 12:00:00")
      and lon between {min_lon} and {max_lon}
      and lat between {min_lat} and {max_lat}
     )
    '''
pd.read_gbq(q)
# -
# # Area of the ocean within 30km of a set in 2020

q = '''with grid as (
select
  distinct lat_index, lon_index 
from 
  `global-fishing-watch.paper_global_longline_sets.sets_grid_10_v20220701` 
where _partitiontime between "2020-01-01" and "2020-12-31"
) 

select 
  sum(1/10*1/10*111*111*cos(3.1416/90*lat_index/20))/1e6 km2,
  count(*) cells
from 
  grid
join
  `global-fishing-watch.paper_global_longline_sets.distance_from_shore` 
on 
  floor(lon*100) = floor( (lon_index/10 +1/20 )*100)
  and 
  floor(lat*100) = floor( (lat_index/10 +1/20 )*100)
where 
  distance_from_shore_m > 0
'''
pd.read_gbq(q)


# 146 million km2 in 2020 were within 30km of a set 

# # Distance in a year from day sets

q = '''with grid as (

select
  distinct lat_index, lon_index # day_category 
from 
  `global-fishing-watch.paper_global_longline_sets.sets_grid_10_v20220701` 
where 
  _partitiontime between "2020-01-01" and "2020-12-31"
  and day_category = 'day'
)

select 
  sum(1/10*1/10*111*111*cos(3.1416/90*lat_index/20))/1e6 km2,
  count(*) cells
from 
  grid
join
  `global-fishing-watch.paper_global_longline_sets.distance_from_shore` 
on 
  floor(lon*100) = floor( (lon_index/10 +1/20 )*100)
  and 
  floor(lat*100) = floor( (lat_index/10 +1/20 )*100)
where 
  distance_from_shore_m > 0

'''
pd.read_gbq(q)

# 137 million km2 are within 30km of a daytime set.
#
# Given the ocean is ~360 million km2, what fraction of it is within 30km of a day set or a set?

# 
136.8/360

146/360

# # What fraction of sets are entirely at night?

q = '''
select 
  sum(if(frac_night=1,1,0))/count(*) 
  frac_night 
from (
  select 
    ssvid,
    start_time,
    sum(if(day_category='night',1,0))/count(*) frac_night
  from   
    `birdlife.sets_5min_cat_v20220701` 
  where 
    _partitiontime between "2017-01-01"  and "2020-12-31"
    and timestamp_diff(timestamp, start_time, minute) > 60*0
    and timestamp_diff(end_time, start_time, minute) > 60*0
  group by ssvid, start_time)
'''
pd.read_gbq(q)

# # What fraction of sets are entirely at night if we subtract off the first and last hour of the set?

# subtract one hour off the end and start of sets
q = '''
select 
  sum(if(frac_night=1,1,0))/count(*) 
  frac_night 
from 
    (select 
    ssvid,
    start_time,
    sum(if(day_category='night',1,0))/count(*) frac_night
    from 
      `birdlife.sets_5min_cat_v20220701` 
    where 
      _partitiontime between "2017-01-01"  and "2020-12-31"
      and timestamp_diff(timestamp, start_time, minute) > 60*1
      and timestamp_diff(end_time, start_time, minute) > 60*1
    group by 
      ssvid, start_time
    )
'''
pd.read_gbq(q)

# # What fraction of sets are entirely at night if we subtract off two hours from the start and end of a set?

# subtract twho hours off the start and end of sets
q = '''
select 
  sum(if(frac_night=1,1,0))/count(*) 
  frac_night 
from 
    (select 
      ssvid,
      start_time,
      sum(if(day_category='night',1,0))/count(*) frac_night
    from     
      `birdlife.sets_5min_cat_v20220701` 
    where 
      _partitiontime between "2017-01-01"  and "2020-12-31"
      and timestamp_diff(timestamp, start_time, minute) > 60*2
      and timestamp_diff(end_time, start_time, minute) > 60*2
    group by ssvid, start_time)'''
pd.read_gbq(q)


