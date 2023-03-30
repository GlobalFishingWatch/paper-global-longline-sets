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

# # Figure 3
# This notebook produces `Figure 3` in `Results`. The figure shows that in regions with night setting recommendations, sets most commonly start a few hours before sunrise, and most overlap with dawn. The figure shows the seasonal variation in longline fishing, and the changes across years.
# The figure is made with: 
#  - time series of day and night sets
#  - start and end time of sets
#  - categories of sets (over dawn, over dusk, entirely night, entirely day)
#  
#  
# Figure Caption: "Day setting and setting during dawn are common both globally and in each region. The number of sets by region that are mostly during the day, mostly at night, and entirely at night show seasonal patterns in each region (a, d, g, j, m). Globally (b), and in all regions except the North Pacific (e, h, k, n), the majority of the sets overlap with nautical dawn (hatched marks), with the most common sets being those that overlap with the dawn but are mostly during daytime hours. The most common times to start in every region (red bars in c, f, i, l, o) are the hours before sunrise, with most sets ending a few hours after sunrise (green bars).
# "

# +


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import skimage.io
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
from datetime import datetime
import proplot as pplt
from highlight_text import HighlightText, ax_text, fig_text


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
# plt.rc('legend',fontsize='12') 
# plt.rc('legend',frameon=False)

import warnings
warnings.filterwarnings('ignore')

colors = ["#FBE2BA", "#FFBD52", "#DB8901", "#0573B3", "#39394A", "#81B9D9"]



# -

# following should be saved somwhere, like Joanna saved it in the prvious version
North_Pacific = '{"type": "Polygon","coordinates": [[[113,23],[260.6,23],[260.6,66],[113,66],[113,23]]]}'
South_Pacific = '{"type": "Polygon","coordinates": [[[150.0, -60.0], [150.0, -55.0], [141.0, -55.0], [141.0, -30.0], [150.0, -30.0], [160.0, -30.0], [240.0, -30.0], [240.0, -60.0], [150.0, -60.0]]]}'
South_Atlantic = '{"type": "Polygon","coordinates": [[[-69.7, -23.0], [25, -23.0], [25, -62.3], [-69.7, -62.3], [-69.7, -23.0]]]}'
South_Indian = '{"type": "Polygon","coordinates": [[[25, -23.0], [118.0, -23.0], [118.0, -58.5], [25, -58.5], [25, -23.0]]]}'

regions_json = {}
regions_json["North_Pacific"] = shape(json.loads(North_Pacific.replace("\'", "\"")))
regions_json["South_Atlantic"] = shape(json.loads(South_Atlantic.replace("\'", "\"")))
regions_json["South_Indian"] = shape(json.loads(South_Indian.replace("\'", "\"")))
regions_json["South_Pacific"] = shape(json.loads(South_Pacific.replace("\'", "\"")))

# +
# Time series of number of majority day, majority night, and all sets 

q = '''CREATE TEMPORARY FUNCTION
   
North_Pacific() as ( "{'type': 'Polygon','coordinates': [[[113,23],[260.6,23],[260.6,66],[113,66],[113,23]]]}");

CREATE TEMPORARY FUNCTION

South_Pacific() as ("{'type': 'Polygon','coordinates': [[[150.0, -60.0], [150.0, -55.0], [141.0, -55.0], [141.0, -30.0], [150.0, -30.0], [160.0, -30.0], [240.0, -30.0], [240.0, -60.0], [150.0, -60.0]]]}");

CREATE TEMPORARY FUNCTION
South_Atlantic() as ("{'type': 'Polygon','coordinates': [[[-69.7, -23.0], [25, -23.0], [25, -62.3], [-69.7, -62.3], [-69.7, -23.0]]]}");

CREATE TEMPORARY FUNCTION
South_Indian() as ("{'type': 'Polygon','coordinates': [[[25, -23.0], [118.0, -23.0], [118.0, -58.5], [25, -58.5], [25, -23.0]]]}");


CREATE TEMPORARY FUNCTION
  find_region(lat float64,lon float64) AS (
    CASE
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(NORTH_PACIFIC(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "North_Pacific"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(SOUTH_PACIFIC(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Pacific"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(SOUTH_ATLANTIC(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Atlantic"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(SOUTH_INDIAN(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Indian"                 
    ELSE
    "Other_Region"
  END
    ); 

with fractions as (

  select 
    date(start_time)date,
    concat(ssvid, cast(start_time as string)) set_id,
    sum(if(day_category = 'night', 1,0))/count(*) frac_night,
    sum(if(day_category = 'day', 1,0))/count(*) frac_day,
    sum(if(day_category = 'dawn', 1,0))/count(*) frac_dawn,
    sum(if(day_category = 'dusk', 1,0))/count(*) frac_dusk,
    st_centroid(st_union_agg(st_geogpoint(lon,lat))) loc,
    sum(1/12) hours
  from       
    `birdlife.sets_5min_cat_v20220701` 
  where 
    date(_partitiontime) between "2017-01-01" and "2020-12-31"
    and abs(lat)<90
    and abs(lon)<180
  group by 
    set_id,date),


categorized as 
( select
  if(frac_day + frac_dawn + frac_dusk > .5,1,0) as majority_day,
  if(frac_day + frac_dawn + frac_dusk < .5,1,0) as majority_night,
  if(frac_night=1,1,0) as entirely_night,
  if(frac_dawn > 0,1,0) as overlapping_dawn,
  *
  from fractions
)

select 
date,
'all' as region,
count(*) sets,
sum(majority_day) majority_day,
sum(majority_night) majority_night,
sum(overlapping_dawn) overlapping_dawn,
sum(entirely_night) entirely_night
from categorized
group by  date
union all
select date,
find_region( st_y(loc),st_x(loc)) as region,
count(*) sets,
sum(majority_day) majority_day,
sum(majority_night) majority_night,
sum(overlapping_dawn) overlapping_dawn,
sum(entirely_night) entirely_night
from categorized
group by region, date
order by date'''

dfcd = pd.read_gbq(q)

# +
# Time of day of sets starting

q = '''CREATE TEMPORARY FUNCTION
   
North_Pacific() as ( "{'type': 'Polygon','coordinates': [[[113,23],[260.6,23],[260.6,66],[113,66],[113,23]]]}");

CREATE TEMPORARY FUNCTION

South_Pacific() as ("{'type': 'Polygon','coordinates': [[[150.0, -60.0], [150.0, -55.0], [141.0, -55.0], [141.0, -30.0], [150.0, -30.0], [160.0, -30.0], [240.0, -30.0], [240.0, -60.0], [150.0, -60.0]]]}");

CREATE TEMPORARY FUNCTION
South_Atlantic() as ("{'type': 'Polygon','coordinates': [[[-69.7, -23.0], [25, -23.0], [25, -62.3], [-69.7, -62.3], [-69.7, -23.0]]]}");

CREATE TEMPORARY FUNCTION
South_Indian() as ("{'type': 'Polygon','coordinates': [[[25, -23.0], [118.0, -23.0], [118.0, -58.5], [25, -58.5], [25, -23.0]]]}");


CREATE TEMPORARY FUNCTION
  find_region(lat float64,lon float64) AS (
    CASE
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(NORTH_PACIFIC(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "North_Pacific"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(SOUTH_PACIFIC(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Pacific"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(SOUTH_ATLANTIC(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Atlantic"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(SOUTH_INDIAN(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Indian"                 
    ELSE
    "Other_Region"
  END
    );   
    

with sets_table as (

select 
row_number() over (partition by concat(ssvid, cast(start_time as string)) order by timestamp, rand()) row,
lat, 
lon,
start_time,
local_hour - sunrise hours_after_sunrise 
from        
  `birdlife.sets_5min_cat_v20220701` 
  --  where date(_partitiontime) between "2020-01-01" and "2020-12-31"
    )
    
 select 
 floor(hours_after_sunrise*6)/6 hours_after_sunrise,
 find_region(lat, lon) region,
 count(*) sets
 from sets_table
 where row = 1
 group by hours_after_sunrise, region
 
 union all
 
 select 
 floor(hours_after_sunrise*6)/6 hours_after_sunrise,
 "all" region,
 count(*) sets
 from sets_table
 where row = 1
 group by hours_after_sunrise, region
 
 
 order by hours_after_sunrise'''

dfs = pd.read_gbq(q)


# -



# +
q = '''
CREATE TEMPORARY FUNCTION
   
North_Pacific() as ( "{'type': 'Polygon','coordinates': [[[113,23],[260.6,23],[260.6,66],[113,66],[113,23]]]}");

CREATE TEMPORARY FUNCTION

South_Pacific() as ("{'type': 'Polygon','coordinates': [[[150.0, -60.0], [150.0, -55.0], [141.0, -55.0], [141.0, -30.0], [150.0, -30.0], [160.0, -30.0], [240.0, -30.0], [240.0, -60.0], [150.0, -60.0]]]}");

CREATE TEMPORARY FUNCTION
South_Atlantic() as ("{'type': 'Polygon','coordinates': [[[-69.7, -23.0], [25, -23.0], [25, -62.3], [-69.7, -62.3], [-69.7, -23.0]]]}");

CREATE TEMPORARY FUNCTION
South_Indian() as ("{'type': 'Polygon','coordinates': [[[25, -23.0], [118.0, -23.0], [118.0, -58.5], [25, -58.5], [25, -23.0]]]}");


CREATE TEMPORARY FUNCTION
  find_region(lat float64,lon float64) AS (
    CASE
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(NORTH_PACIFIC(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "North_Pacific"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(SOUTH_PACIFIC(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Pacific"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(SOUTH_ATLANTIC(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Atlantic"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(SOUTH_INDIAN(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Indian"                 
    ELSE
    "Other_Region"
  END
    );   
    

with sets_table as (

select 
  row_number() over (partition by concat(ssvid, cast(start_time as string)) order by timestamp, rand()) row,
  row_number() over (partition by concat(ssvid, cast(start_time as string)) order by timestamp desc, rand()) row_end,
  lat, 
  lon,
  start_time,
  case when (local_hour - sunrise) < - 8 then (local_hour - sunrise) + 24
   when (local_hour - sunrise) > 20 then (local_hour - sunrise) - 24
   else (local_hour - sunrise) end hours_after_sunrise 
from  
  `birdlife.sets_5min_cat_v20220701`
--  where date(_partitiontime) between "2020-01-01" and "2020-12-31"
    ),
    
 start_table as (
    
 select 
 floor(hours_after_sunrise*6)/6 hours_after_sunrise,
 find_region(lat, lon) region,
 count(*) sets
 from sets_table
 where row = 1
 group by hours_after_sunrise, region
 
 union all
 
 select 
 floor(hours_after_sunrise*6)/6 hours_after_sunrise,
 "all" region,
 count(*) sets
 from sets_table
 where row = 1
 group by hours_after_sunrise, region
 
 order by hours_after_sunrise),
 
 
 end_table as 
 (
    
 select 
 floor(hours_after_sunrise*6)/6 hours_after_sunrise,
 find_region(lat, lon) region,
 count(*) sets_end
 from sets_table
 where row_end = 1
 group by hours_after_sunrise, region
 
 union all
 
 select 
 floor(hours_after_sunrise*6)/6 hours_after_sunrise,
 "all" region,
 count(*) sets_end
 from sets_table
 where row_end = 1
 group by hours_after_sunrise, region
 order by hours_after_sunrise)
 
 
 select * from start_table join end_table
 using(hours_after_sunrise, region)
 order by hours_after_sunrise
 
 
 
 '''

dfs2 = pd.read_gbq(q)
# -



# +
# categorys of sets -- overlapping dawn, dusk, day, and night

q = """CREATE TEMPORARY FUNCTION
   
North_Pacific() as ( "{'type': 'Polygon','coordinates': [[[113,23],[260.6,23],[260.6,66],[113,66],[113,23]]]}");

CREATE TEMPORARY FUNCTION

South_Pacific() as ("{'type': 'Polygon','coordinates': [[[150.0, -60.0], [150.0, -55.0], [141.0, -55.0], [141.0, -30.0], [150.0, -30.0], [160.0, -30.0], [240.0, -30.0], [240.0, -60.0], [150.0, -60.0]]]}");

CREATE TEMPORARY FUNCTION
South_Atlantic() as ("{'type': 'Polygon','coordinates': [[[-69.7, -23.0], [25, -23.0], [25, -62.3], [-69.7, -62.3], [-69.7, -23.0]]]}");

CREATE TEMPORARY FUNCTION
South_Indian() as ("{'type': 'Polygon','coordinates': [[[25, -23.0], [118.0, -23.0], [118.0, -58.5], [25, -58.5], [25, -23.0]]]}");


CREATE TEMPORARY FUNCTION
  find_region(lat float64,lon float64) AS (
    CASE
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(NORTH_PACIFIC(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "North_Pacific"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(SOUTH_PACIFIC(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Pacific"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(SOUTH_ATLANTIC(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Atlantic"
      WHEN ST_CONTAINS(ST_GeogFromGeoJSON(SOUTH_INDIAN(),make_valid=>True), ST_GEOGPOINT(lon, lat)) THEN "South_Indian"                 
    ELSE
    "Other_Region"
  END
    ); 

with fractions as (

  select 
    date(start_time)date,
    concat(ssvid, cast(start_time as string)) set_id,
    sum(if(day_category = 'night', 1,0))/count(*) frac_night,
    sum(if(day_category = 'day', 1,0))/count(*) frac_day,
    sum(if(day_category = 'dawn', 1,0))/count(*) frac_dawn,
    sum(if(day_category = 'dusk', 1,0))/count(*) frac_dusk,
    st_centroid(st_union_agg(st_geogpoint(lon,lat))) loc,
    sum(1/12) hours
  from      
    `birdlife.sets_5min_cat_v20220701` 
  where 
    date(_partitiontime) between "2017-01-01" and "2020-12-31"
    and abs(lat)<90
    and abs(lon)<180
  group by set_id,date),


categorized as 
( select

   case 
  when frac_dawn > 0 and frac_dawn+ frac_day + frac_dusk  >= .5 and frac_dawn >= frac_dusk then "1overlap dawn (mostly day)"
  when frac_day = 1 then "2entirely day"
  when frac_dusk > 0 and frac_dawn+ frac_day + frac_dusk >= .5 and frac_dusk > frac_dawn then "3overlap with dusk (mostly day)"
  when frac_dusk > 0 and frac_night >= .5 and frac_dusk >= frac_dawn then "4overlap with dusk (mostly night)"
  when frac_night = 1 then "5entirely night"
  when frac_night >= .5 and frac_dawn > 0 and frac_dawn >= frac_dusk then "6overlap with dawn (mostly night)"
  else "other" end category,

  *
  from fractions
)

select 
'all' as region,
count(*) sets,
category,
from categorized
group by category
union all
select 
find_region( st_y(loc),st_x(loc)) as region,
count(*) sets,
category,
from categorized
group by region, category
order by category"""

dfc = pd.read_gbq(q)
dfc["category"] = dfc.category.apply(lambda x: x[1:])
# -



# +
fig, axs = pplt.subplots(ncols=3, 
                         nrows=6, 
                         wratios=(3.5, 2, 1.5),
                         hratios=(1.1,3,3,3,3,3),
                         figwidth=8,
                        figheight=7.5,
                        sharex=True,
                        sharey=False,
                        grid=False,
                        span=False,
                        hspace=2,
                        wspace=2,
                        proj=['cart','cart','cart',
                              'cart','cart','polar',
                            'cart','cart','polar',
                             'cart','cart','polar',
                             'cart','cart','polar',
                             'cart','cart','polar'])




pplt.rc['title.size'] = 12



for i, region in enumerate(
    ["all", "North_Pacific", "South_Atlantic", "South_Indian", "South_Pacific"]
):
    d = dfcd[dfcd.region == region]
    ax = axs[i * 3+3]
    #     ax.plot(d.date.values[7:-7],
    #                 d.sets.rolling(window=14).mean().values[7:-7],
    #                 label='all',
    #                color="#666666")
    ax.plot(
        d.date.values[7:-7],
        d.majority_day.rolling(window=14).mean().values[7:-7],
        label="majority day",
        color="#FFBD52",
#         color="#FED604",
    )  # DA7423")
    ax.plot(
        d.date.values[7:-7],
        d.majority_night.rolling(window=14).mean().values[7:-7],
        label="majority night",
        color="#0373B3",
    )
    ax.plot(
        d.date.values[7:-7],
        d.entirely_night.rolling(window=14).mean().values[7:-7],
        label="entirely night",
        color="#666666",
    )
    ax.set_ylim(0, d.majority_day.max())
#     if i == 4:
#         ax.legend(loc="b", ncols=4, frameon=False)
    if region == "all":
        title = "Global"
    else:
        title = region.replace("_", " ")
    ax.format(titleloc="ul", title=" " + title, abc=True)
    #     ax.format( abc=True)
    if i == 2:
        ax.set_ylabel("sets per day")
    ax.set_xlim(datetime(2017, 1, 1), datetime(2020, 12, 31))
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    ax.grid(axis = 'x')

    
    #########################
    ### Donut chart
    #########################
    ax = axs[i * 3 + 1 + 3]
    dd = dfc[dfc.region == region]
    sets = dd.sets.values
    labels = dd.category.values
    cycle = pplt.Cycle(colors)
    
    if i == 1:
        labels = ["\nOver dawn,\nmostly day",
              "Entirely Day",
              "Over dusk,\nmostly day",
             "Over dusk,\nmostly night",
             "Entirely night",
             "Over dawn,\nmostly night\n "]
        wedges, texts = ax.pie(sets[:6], counterclock=False, startangle=90, 
                              labels=labels)  
    
    else:
        wedges, texts = ax.pie(sets[:6], counterclock=False, startangle=90)

    hatches = ["///","","","","","///"]
    for j in range(len(wedges)):
        wedges[j].set( hatch = hatches[j],
                      facecolor = colors[j],
                     linewidth = 0,
                     edgecolor=(0,0,0,.7))

    circle = plt.Circle( ( 0, 0 ), .5, color='white' )
    ax.add_patch( circle )
           
    ax.format(abc=True)
   

    ########################
    ### Radial plot
    ########################


    ax = axs[i * 3 + 2 + 3]
        # remove grid
    ax.axis('off')
    pie_alpha = .7

    # Set the coordinates limits
    upperLimit = 90
    lowerLimit = 20

    d = dfs2[dfs2.region == region]
    d['hour'] = d.hours_after_sunrise.apply(lambda x: int(x)%24)
    d = d.groupby('hour').sum()
    d.sets = d.sets/d.sets.sum()
    d.sets_end = d.sets_end/d.sets_end.sum()
    y = d.sets.values / d.sets.values.sum() / (24 * 6)
    y2 = d.sets_end.values / d.sets_end.values.sum() / (24 * 6)
    x = d.hours_after_sunrise.values
    # Compute max and min in the dataset
    if i == 0:
        max_ = d['sets'].max()

        # Let's compute heights: they are a conversion of each item value in those new coordinates
        # In our example, 0 in the dataset will be converted to the lowerLimit (10)
        # The maximum will be converted to the upperLimit (100)
        slope = (upperLimit - lowerLimit) / max_


    heights = slope * d.sets #+ lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / 24 

    # Compute the angle each bar is centered on:
    indexes = list(range(len(d.index)))
    angles = [-(element-.5) * width + 3.1416/2 for element in indexes]
    angles

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width*1.8, 
        bottom=lowerLimit,
        linewidth=0, 
        color = '#A900AA',
        alpha = pie_alpha,
        edgecolor="white")

    ## now for the end times




    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
#     slope = (upperLimit - lowerLimit) / max_


    heights = slope * d.sets_end #+ lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°

    # Compute the angle each bar is centered on:
    indexes = list(range(len(d.index)))
    angles = [-(element-.5) * width + 3.1416/2 for element in indexes]
    angles

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width*1.8, 
        bottom=lowerLimit,
        linewidth=0, 
        alpha = pie_alpha*.6,
        edgecolor="white",
        color="#029A38")




    pi = np.pi

    
    if i == 0:
        
        for j in range(24):
            x = np.pi*2*j/24
            ax.plot([x,x],[80,85], color='grey',alpha=.3)
        
        fontsize = 10
        range_ = [80,90]

        ax.text(pi/2,95,"Sunrise",ha='center',fontsize=fontsize)
        ax.text(-pi/2,95,"+12h",ha='center',va='top',fontsize=fontsize)
        ax.text(0,95,"+6h",va='center',ha='left',fontsize=fontsize)
        ax.text(-pi,95,"+18h",va='center',ha='right',fontsize=fontsize)
        
    ax.plot([pi/2, pi/2],range_,color='grey')
    ax.plot([0, 0],range_,color='grey')
    ax.plot([-pi/2, -pi/2],range_,color='grey')
    ax.plot([-pi, -pi],range_,color='grey')
        
    ax.set_ylim(0,100)
    ax.format(abc=True)
    





# expand = [-5, -5, 5, 5]
# fig = legend.figure
# fig.canvas.draw()
# bbox = legend.get_window_extent()
# bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
# bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
# ax.set_axis_off()






######## Key to the bottom ############

fontsize = 10

ax = axs[0]

mostly_day_color = "#F5B752"
mostly_night_color = "#0373B3"
night_color = 'black'

highlight_textprops = [{"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'k', 
                        "bbox": {"facecolor": mostly_day_color,
                                  "linewidth": 0, "pad": 1.5}},
                       {"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'white', 
                        "bbox": {"facecolor": mostly_night_color,
                                  "linewidth": 0, "pad": 1.5}},
                       {"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'white', 
                        "bbox": {"facecolor": night_color,
                                  "linewidth": 0, "pad": 1.5}}
                      ]


ax_text(x=datetime(2016,10,1), y=1, va='top',
         s='<Sets per day happening ><mostly in the day><, ><mostly in the night>\n<or ><entirely in the night>',
         highlight_textprops=highlight_textprops,
         ax=ax)
ax.set_axis_off()



###

ax = axs[1]
highlight_textprops = [{"fontsize":fontsize, "color":'k'}]
ax_text(x=-.7, y=1, va='top',
         s='<Distribution of sets>',
                 highlight_textprops=highlight_textprops,
         ax=ax)
ax.set_axis_off()


# ###

ax = axs[2]

starting_color = "#A900AA"
ending_color = "#029A38"
highlight_textprops = [{"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'white', 
                        "bbox": {"facecolor": starting_color,
                                  "linewidth": 0, "pad": 1.5}},
                       {"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'white', 
                        "bbox": {"facecolor": ending_color,
                                  "linewidth": 0, "pad": 1.5}},
                       {"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'k'}
                      ]
ax_text(x=.1, y=1, va='top',
         s='<Relative number of sets>\n<starting>< or ><ending>< at a given>\n<time of day>',
                 highlight_textprops=highlight_textprops,
         ax=ax)
ax.set_axis_off()




axs.format(grid=False, abcloc="ur")



# plt.savefig("images/SetTimeRadialLegend2.png", dpi=300, bbox_inches="tight")



# +
# # Fix
#
# I couldn't add the top row and edit the letters, so I created two seperate figures and copied and pasted them together in image software
# +
# just the titles above the three columns
fig, axs = pplt.subplots(ncols=3, 
                         nrows=1, 
                         wratios=(3.5, 2, 1.5),
                         figwidth=8,
                        figheight=1.1/(1.1+3*5) * 7.5 *20,
                        sharex=True,
                        sharey=False,
                        grid=False,
                        span=False,
                        hspace=2,
                        wspace=2)




pplt.rc['title.size'] = 12







######## Key to the bottom ############

fontsize = 10

ax = axs[0]

mostly_day_color = "#F5B752"
mostly_night_color = "#0373B3"
night_color = 'black'

highlight_textprops = [{"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'k', 
                        "bbox": {"facecolor": mostly_day_color,
                                  "linewidth": 0, "pad": 1.5}},
                       {"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'white', 
                        "bbox": {"facecolor": mostly_night_color,
                                  "linewidth": 0, "pad": 1.5}},
                       {"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'white', 
                        "bbox": {"facecolor": night_color,
                                  "linewidth": 0, "pad": 1.5}}
                      ]


ax_text(x=0, y=1, va='top',
         s='<Sets per day happening ><mostly in the day><, ><mostly in the night>\n<or ><entirely in the night>',
         highlight_textprops=highlight_textprops,
         ax=ax)



ax.set_axis_off()



###

ax = axs[1]
highlight_textprops = [{"fontsize":fontsize, "color":'k'}]
ax.set_xlim(0,100)
ax.set_ylim(0,100)
ax_text(x=30, y=100, va='top',
         s='<Distribution of sets>',
                 highlight_textprops=highlight_textprops,
         ax=ax)

highlight_textprops = [{"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'k'}]
ax_text(x=47, y=98, va='top',
         s='<overlapping>\n<dawn>',
                 highlight_textprops=highlight_textprops,
         ax=ax)

rect = mpl.patches.Rectangle((30, 95), 15, 2.7, linewidth=1, hatch='///',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)


ax.set_axis_off()


# ###

ax = axs[2]

starting_color = "#A900AA"
ending_color = "#029A38"
highlight_textprops = [{"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'white', 
                        "bbox": {"facecolor": starting_color,
                                  "linewidth": 0, "pad": 1.5}},
                       {"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'white', 
                        "bbox": {"facecolor": ending_color,
                                  "linewidth": 0, "pad": 1.5}},
                       {"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'k'}
                      ]
ax_text(x=.2, y=1, va='top',
         s='<Relative number of sets>\n<starting>< or ><ending>< at a given>\n<time of day>',
                 highlight_textprops=highlight_textprops,
         ax=ax)
ax.set_axis_off()




axs.format(grid=False, abcloc="ur")



# plt.savefig("images/SetTimeRadialLegend2.png", dpi=300, bbox_inches="tight")


# +
# just figures
fig, axs = pplt.subplots(ncols=3, 
                         nrows=5, 
                         wratios=(3.5, 2, 1.5),
                         hratios=(3,3,3,3,3),
                         figwidth=8,
                        figheight=7,
                        sharex=True,
                        sharey=False,
                        grid=False,
                        span=False,
                        hspace=2,
                        wspace=2,
                        proj=['cart','cart','polar',
                            'cart','cart','polar',
                             'cart','cart','polar',
                             'cart','cart','polar',
                             'cart','cart','polar'])




pplt.rc['title.size'] = 12



for i, region in enumerate(
    ["all", "North_Pacific", "South_Atlantic", "South_Indian", "South_Pacific"]
):
    d = dfcd[dfcd.region == region]
    ax = axs[i * 3]
    #     ax.plot(d.date.values[7:-7],
    #                 d.sets.rolling(window=14).mean().values[7:-7],
    #                 label='all',
    #                color="#666666")
    ax.plot(
        d.date.values[7:-7],
        d.majority_day.rolling(window=14).mean().values[7:-7],
        label="majority day",
        color="#FFBD52",
#         color="#FED604",
    )  # DA7423")
    ax.plot(
        d.date.values[7:-7],
        d.majority_night.rolling(window=14).mean().values[7:-7],
        label="majority night",
        color="#0373B3",
    )
    ax.plot(
        d.date.values[7:-7],
        d.entirely_night.rolling(window=14).mean().values[7:-7],
        label="entirely night",
        color="#666666",
    )
    ax.set_ylim(0, d.majority_day.max())
#     if i == 4:
#         ax.legend(loc="b", ncols=4, frameon=False)
    if region == "all":
        title = "Global"
    else:
        title = region.replace("_", " ")
    ax.format(titleloc="ul", title=" " + title, abc=True)
    #     ax.format( abc=True)
    if i == 2:
        ax.set_ylabel("sets per day")
    ax.set_xlim(datetime(2017, 1, 1), datetime(2020, 12, 31))
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    ax.grid(axis = 'x')

    
    #########################
    ### Donut chart
    #########################
    ax = axs[i * 3 + 1]
    dd = dfc[dfc.region == region]
    sets = dd.sets.values
    labels = dd.category.values
    cycle = pplt.Cycle(colors)
    
    if i == 1:
        labels = ["\nOver dawn,\nmostly day",
              "Entirely Day",
              "Over dusk,\nmostly day",
             "Over dusk,\nmostly night",
             "Entirely night",
             "Over dawn,\nmostly night\n "]
        wedges, texts = ax.pie(sets[:6], counterclock=False, startangle=90, 
                              labels=labels)  
    
    else:
        wedges, texts = ax.pie(sets[:6], counterclock=False, startangle=90)

    hatches = ["///","","","","","///"]
    for j in range(len(wedges)):
        wedges[j].set( hatch = hatches[j],
                      facecolor = colors[j],
                     linewidth = 0,
                     edgecolor=(0,0,0,.7))

    circle = plt.Circle( ( 0, 0 ), .5, color='white' )
    ax.add_patch( circle )
           
    ax.format(abc=True)
   

    ########################
    ### Radial plot
    ########################


    ax = axs[i * 3 + 2]
        # remove grid
    ax.axis('off')
    pie_alpha = .7

    # Set the coordinates limits
    upperLimit = 90
    lowerLimit = 20

    d = dfs2[dfs2.region == region]
    d['hour'] = d.hours_after_sunrise.apply(lambda x: int(x)%24)
    d = d.groupby('hour').sum()
    d.sets = d.sets/d.sets.sum()
    d.sets_end = d.sets_end/d.sets_end.sum()
    y = d.sets.values / d.sets.values.sum() / (24 * 6)
    y2 = d.sets_end.values / d.sets_end.values.sum() / (24 * 6)
    x = d.hours_after_sunrise.values
    # Compute max and min in the dataset
    if i == 0:
        max_ = d['sets'].max()

        # Let's compute heights: they are a conversion of each item value in those new coordinates
        # In our example, 0 in the dataset will be converted to the lowerLimit (10)
        # The maximum will be converted to the upperLimit (100)
        slope = (upperLimit - lowerLimit) / max_


    heights = slope * d.sets #+ lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / 24 

    # Compute the angle each bar is centered on:
    indexes = list(range(len(d.index)))
    angles = [-(element-.5) * width + 3.1416/2 for element in indexes]
    angles

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width*1.8, 
        bottom=lowerLimit,
        linewidth=0, 
        color = '#A900AA',
        alpha = pie_alpha,
        edgecolor="white")

    ## now for the end times




    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
#     slope = (upperLimit - lowerLimit) / max_


    heights = slope * d.sets_end #+ lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°

    # Compute the angle each bar is centered on:
    indexes = list(range(len(d.index)))
    angles = [-(element-.5) * width + 3.1416/2 for element in indexes]
    angles

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width*1.8, 
        bottom=lowerLimit,
        linewidth=0, 
        alpha = pie_alpha*.6,
        edgecolor="white",
        color="#029A38")




    pi = np.pi

    
    if i == 0:
        
        for j in range(24):
            x = np.pi*2*j/24
            ax.plot([x,x],[80,85], color='grey',alpha=.3)
        
        fontsize = 10
        range_ = [80,90]

        ax.text(pi/2,95,"Sunrise",ha='center',fontsize=fontsize)
        ax.text(-pi/2,95,"+12h",ha='center',va='top',fontsize=fontsize)
        ax.text(0,95,"+6h",va='center',ha='left',fontsize=fontsize)
        ax.text(-pi,95,"+18h",va='center',ha='right',fontsize=fontsize)
        
    ax.plot([pi/2, pi/2],range_,color='grey')
    ax.plot([0, 0],range_,color='grey')
    ax.plot([-pi/2, -pi/2],range_,color='grey')
    ax.plot([-pi, -pi],range_,color='grey')
        
    ax.set_ylim(0,100)
    ax.format(abc=True)
    


axs.format(grid=False, abcloc="ur")



# expand = [-5, -5, 5, 5]
# fig = legend.figure
# fig.canvas.draw()
# bbox = legend.get_window_extent()
# bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
# bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
# ax.set_axis_off()






# plt.savefig("images/SetTimeRadialLegend2.png", dpi=300, bbox_inches="tight")

# -





