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

# # How much do albatross species' ranges overlap with sets? And are these sets during the day or night?
#
# This notebook produces `Figure 4` in `Results`. The figure shows the extent to which albatross ranges overlap with longline sets, and how this relates to time of day, and sunrise. The figure is made with: species areas, set times relative to dawn, set categories (over dawn, over dusk, entirely night, entirely day), and bird status from IUCN website. 
#
# Results: "These patterns of setting longlines are a threat to endangered and threatened albatrosses. In a given year, the fraction of an albatross' range within 30 km of a longline set varied from 7 % of the range for the Southern royal albatross (Diomedea epomophora), whose range is farther south than most longline activity, to 65 % for Amsterdam alba- tross (Diomedea amsterdamensis), whose range is in areas of intensive longlining in the southern Indian Ocean. In every species' range, there were a few tens of thousands of sets per year between 2017 and 2020. For all but one of 14 species that have a range of greater than 5 million km2 and that are listed as Vulnerable, Endangered, or Critically En- dangered by the IUCN, the majority of sets overlapped with dawn, and in none of the ranges was the fraction of night sets >7 %"
#
# Figure caption: "The ranges (column 2) of vulnerable, endangered, or critically endangered albatross species overlap extensively with longline sets. For all except one species (Phoebastria albatrus), the majority of sets overlap with dawn (hatched lines, column 3). The start and end time of sets in the species' range (column 4) reveals the strong preference to start before dawn and finish in the day, overlapping dawn when albatrosses are most vulnerable. Column 1 shows relative bird size. Only species that have a range of greater than 5 million km2 are shown."

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import matplotlib.gridspec as gridspec
import skimage.io
import pandas as pd
import cartopy
import cartopy.crs as ccrs
from shapely import wkt
import matplotlib.patches as mpatches
from datetime import datetime
import geopandas as gpd
from highlight_text import HighlightText, ax_text, fig_text
import proplot as pplt


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



import warnings
warnings.filterwarnings('ignore')

# +
# # !bq cp birdlife.albatross_ranges global-fishing-watch:paper_global_longline_sets.albatross_ranges

# +
# get outlines of species areas

q = '''
select 
  binomial as bird_species,
  wkt_simple wkt,
  st_area(st_geogfromtext(wkt_simple, make_valid => True))/1e6/1e6 million_km2_range
from 
  `global-fishing-watch.paper_global_longline_sets.albatross_ranges` 
where 
  seasonal =1 
-- and 
-- st_area(st_geogfromtext(wkt_simple, make_valid => True))/1e6 > 5e6
order by binomial'''

dfr = pd.read_gbq(q)

# +

dfr['range'] = dfr.wkt.apply(wkt.loads)
# -



# +
# get set times relative to dawn

q = '''

with 

 bird_range_table as 
(select
st_geogfromtext(wkt_simple, make_valid => True) as species_range,
binomial
from 
  `global-fishing-watch.paper_global_longline_sets.albatross_ranges` 
where 
  seasonal =1 
order by binomial),



sets_table as (

select 
  row_number() over (partition by concat(ssvid, cast(start_time as string)) order by timestamp, rand()) row,
  lat, 
  lon,
  start_time,
  local_hour - sunrise hours_after_sunrise 
from  
  `birdlife.sets_5min_cat_v20220701` 
where 
  date(_partitiontime) between "2017-01-01" and "2020-12-31"
    )
    
 select 
   floor(hours_after_sunrise*6)/6 hours_after_sunrise,
   binomial as bird_name,
   count(*) sets
 from 
   sets_table
 cross join
 bird_range_table
 where 
   row = 1
   -- and st_area(species_range)/1e6 > 5e6
   and st_contains( species_range, st_geogpoint(lon,lat))
 group by 
   hours_after_sunrise, bird_name
 
 order by hours_after_sunrise'''

dfh = pd.read_gbq(q)
# -



# +
q = '''
 
 with 

 bird_range_table as 

(
select
  st_geogfromtext(wkt_simple, make_valid => True) as species_range,
  binomial
from 
  `global-fishing-watch.paper_global_longline_sets.albatross_ranges` 
where 
  seasonal =1 
order by 
  binomial),



sets_table as (

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
     binomial as bird_name,
     count(*) sets
 from 
     sets_table
 cross join
     bird_range_table
 where 
     row = 1
     and st_area(species_range)/1e6 > 5e6
     and st_contains( species_range, st_geogpoint(lon,lat))
 group by 
   hours_after_sunrise, bird_name
 order by hours_after_sunrise
 ),
 
 
 end_table as (
   select 
   floor(hours_after_sunrise*6)/6 hours_after_sunrise,
   binomial as bird_name,
   count(*) sets_end
   from sets_table
   cross join
   bird_range_table
   where row_end = 1
   -- and st_area(species_range)/1e6 > 5e6
   and st_contains( species_range, st_geogpoint(lon,lat))
   group by hours_after_sunrise, bird_name
   order by hours_after_sunrise
 
 )
 
 
 select * from start_table join end_table
 using(hours_after_sunrise, bird_name)
 order by hours_after_sunrise
 '''

dfh = pd.read_gbq(q)
# -



# +
# get set categories

q = '''with  

bird_range_table as 

(select
  st_geogfromtext(wkt_simple, make_valid => True) as species_range,
  binomial
from 
  `global-fishing-watch.paper_global_longline_sets.albatross_ranges`
where 
  seasonal =1 
order by binomial),



fractions as (

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

    case 
    when frac_dawn > 0 and frac_dawn+ frac_day + frac_dusk  >= .5 and frac_dawn >= frac_dusk then "1overlap dawn (mostly day)"
    when frac_day = 1 then "2entirely day"
    when frac_dusk > 0 and frac_dawn+ frac_day + frac_dusk >= .5 and frac_dusk > frac_dawn then "3overlap with dusk (mostly day)"
    when frac_dusk > 0 and frac_night >= .5 and frac_dusk >= frac_dawn then "4overlap with dusk (mostly night)"
    when frac_night = 1 then "5entirely night"
    when frac_night >= .5 and frac_dawn > 0 and frac_dawn >= frac_dusk then "6overlap with dawn (mostly night)"
    else "other" end category,
    *
  from 
    fractions
)


select 
  binomial as bird_name,
  count(*) sets,
category,
from categorized
cross join
bird_range_table
where 
-- st_area(species_range)/1e6 > 5e6 and 
st_contains( species_range, loc)
group by bird_name, category
order by category'''

dfc = pd.read_gbq(q)
# -

# # Get Area Affected by Sets

# +
q = ''' WITH
  bird_range_table AS (
  SELECT
    st_geogfromtext(wkt_simple,
      make_valid => TRUE) AS species_range,
    st_area(st_geogfromtext(wkt_simple,
        make_valid => TRUE))/1e6/1e6 mil_km2_range,
    binomial AS bird_species
  FROM
    `global-fishing-watch.paper_global_longline_sets.albatross_ranges`
  WHERE
    seasonal =1
  ORDER BY
    binomial),
    
    
  sets_table AS (
  SELECT
    lat_index/10+.05 lat,
  IF
    (lon_index>1800,
      lon_index - 3600,
      lon_index)/10+.05 lon,
    day_category,
    COUNT(DISTINCT set_id) sets
  FROM
    birdlife.sets_grid_10
  WHERE
    DATE(_partitiontime) BETWEEN "2020-01-01"
    AND "2020-12-31"
  GROUP BY
    lat,
    lon,
    day_category ),
    
    
  sets_all AS (
  SELECT
    lat_index/10+.05 lat,
    lon_index/10+.05 lon,
    COUNT(DISTINCT set_id) sets
  FROM
    `global-fishing-watch.paper_global_longline_sets.sets_grid_10_v20220701`
  WHERE
    DATE(_partitiontime) BETWEEN "2020-01-01"
    AND "2020-12-31"
  GROUP BY
    lat,
    lon )
  -- day category and region
  
  
  
SELECT
  sets,
  day_category,
  bird_species,
  SUM(COS(lat*3.14/180)*111*111/10/10)/1e6 millionkm2,
  mil_km2_range
FROM
  sets_table
CROSS JOIN
  bird_range_table
WHERE
  -- mil_km2_range > 1 and
  st_contains( species_range,
    st_geogpoint(lon,
      lat))
GROUP BY
  sets,
  day_category,
  bird_species,
  mil_km2_range
UNION ALL
  -- all day categories by region
SELECT
  sets,
  "all" AS day_category,
  bird_species,
  SUM(COS(lat*3.14/180)*111*111/10/10)/1e6 millionkm2,
  mil_km2_range
FROM
  sets_all
CROSS JOIN
  bird_range_table
WHERE
  -- mil_km2_range > 5 and
  st_contains( species_range,
    st_geogpoint(lon,
      lat))
GROUP BY
  sets,
  day_category,
  bird_species,
  mil_km2_range
ORDER BY
  sets DESC'''

dfar = pd.read_gbq(q)
# -
bird_names = dfh.bird_name.unique()
bird_names.sort()
bird_names

# +
# research on the bird populations by Cian Luck, retreived from IUCN website

bird_status = '''Species	IUCN status	Population trend	Population size
Diomedea amsterdamensis	Endangered	Increasing	92	310
Diomedea antipodensis	Endangered	Decreasing	50000	310
Diomedea dabbenena	Critically endangered	Decreasing	3400	310
Diomedea epomophora	Vulnerable	Stable	27200	310
Diomedea exulans	Vulnerable	Decreasing	20100	310
Diomedea sanfordi	Endangered	Decreasing	17000	310
Phoebastria albatrus	Vulnerable	Increasing	1734	220
Phoebastria immutabilis	Near threatened	Stable	1600000	200
Phoebastria nigripes	Near threatened	Increasing	139800	200
Phoebetria fusca	Endangered	Decreasing	21234	200
Phoebetria palpebrata	Near threatened	Decreasing	58000	200
Thalassarche bulleri	Near threatened	Decreasing	50000	220
Thalassarche carteri	Endangered	Decreasing	82000	220
Thalassarche cauta	Near threatened	Unknown	30700	220
Thalassarche chrysostoma	Endangered	Decreasing	250000	220
Thalassarche eremita	Vulnerable	Stable	11000	220
Thalassarche impavida	Vulnerable	Increasing	43296	220
Thalassarche melanophris	Least concern	Increasing	1400000	220
Thalassarche salvini	Vulnerable	Unknown	79990	220
Thalassarche steadi	Near threatened	Decreasing	203600	220'''.split("\n")

s = []
c = []
pop_sizes = {}
pop_change = {}
bird_sizes = {}
for b in bird_status[1:]:
    s.append(b.split("\t")[1])
    c.append(b.split("\t")[2])
    bird = b.split("\t")[0]
    pop_change[bird] = b.split("\t")[2]
    pop_sizes[bird] = int(b.split("\t")[3])
    bird_sizes[bird] = int(b.split("\t")[4])

set(s)
# -

set(c)

change_y_mapping = {'Decreasing':-.8, 'Increasing':.8, 'Stable':0, 'Unknown':0}

# +
threatened_colors = {
    "Critically endangered": "#9e0d00",  #'#D54135' ,
    "Endangered": "#E97343",
    "Vulnerable": "#FFDE03",
    "Near threatened": "#CCE24F",
    "Least concern": "#72BE57",
}

# #7a85d6 -- old blue color

bird_colors = {}
iucn_statuses = {}

for b in bird_status[1:]:
    b = b.split("\t")
    bird_name = b[0]
    status = b[1]
    bird_colors[bird_name] = threatened_colors[status]
    iucn_statuses[bird_name] = status


# -
# # Only Threatened

# +
threatened_birds = []

for i, bird_name in enumerate(bird_names):
    
    if iucn_statuses[bird_name] in ( 'Critically endangered',
                                    'Endangered', 'Vulnerable'):
        threatened_birds.append(bird_name)

        
def get_threatened_status(bird):
    return iucn_statuses[bird]

threatened_birds.sort(key=get_threatened_status)

for t in (threatened_birds):
    print(t, iucn_statuses[t])
# -
# ls bird_images

# +
plt.rc("legend", fontsize="9")
plt.rc("legend", frameon=False)


# colors = ["#FFF7CF", "#FEE664", "#FED604", "#1241AA", "#232436", "#D2D9F0"]
colors = ["#FBE2BA", "#FFBD52", "#DB8901", "#0573B3", "#39394A", "#81B9D9"]
pplt.rc["land.color"] = "#CCCCCC"
arrow_color = "#828282"

# array = [
#     [i, i, i + 1, i + 1, i + 2, i + 2, i + 3]
#     for i in range(1, len(threatened_birds) * 4, 4)
# ]


a = ["cart", "robin", "polar", "cart"]


fig, axs = pplt.subplots(
    ncols=5,
    nrows=len(threatened_birds)+1,
    wratios=(.5,1.8,2,1.2,1.2),
    hratios=[.8]+[1.5]*len(threatened_birds),
    figwidth=7.1,
    figheight=11,
    sharex=True,
    sharey=False,
    grid=False,
    span=False,
    hspace=1,
    wspace=0,
    proj=["cart","cart", "cart", "cart", "cart"]
    + ["cart","cart", "robin", "cart", "polar"] * len(threatened_birds),
)


for i, bird_name in enumerate(threatened_birds):

    if iucn_statuses[bird_name] in (
        "Critically endangered",
        "Endangered",
        "Vulnerable",
    ):
        
        
        ######################
        ### Bird images
        ######################
        
        
        genus = bird_name.split(" ")[0]
        if genus == 'Phoebastria':
            genus = 'Phoebetria'

        bird_size = bird_sizes[bird_name]/310

        ax = axs[(i+1)*5]
        im = plt.imread(f'bird_images/{genus}_silhouette.png')
        ax.imshow(im, extent=[-170*bird_size, 170*bird_size, -120*bird_size, 120*bird_size])
        ax.set_xlim(-100,100)
        ax.set_ylim(-100,100)
        ax.set_axis_off()

        ####################
        ### Statistics about each bird
        #####################
        ax = axs[(i+1) * 5 + 1 ]
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 10)
        
        
        x_offset = 1.8
        ax.add_patch(mpl.patches.Rectangle((0,-1), 2.7, 11, color=bird_colors[bird_name]))
        
        if i == 0:
            ax.text(x=.2, y=5,
                s= f"critically\nendangered",
                fontsize=8, ha='left', va='center', rotation=90, color = 'white'
            )
            
        if i == 1:
            ax.text(x=.8, y=5,
                s= f"endangered",
                fontsize=8, ha='left', va='center', rotation=90, color = 'black'
            )            
    

        if i == 7:
            ax.text(x=.8, y=5,
                s= f"vunlernable",
                fontsize=8, ha='left', va='center', rotation=90, color = 'black'
            )  
            
        n = bird_name  # .replace(" ","\n")
        p = pop_sizes[bird_name]

        a = dfr[dfr.bird_species == bird_name].million_km2_range.values[0]
        d = dfar[(dfar.bird_species == bird_name) & (dfar.day_category == "all")]
        area_fished = d.millionkm2.sum()
        perc_fished = area_fished / a * 100

        ax.set_title(
            f"      {n}\n      population: ~{p:,}\n      {d.sets.sum():,} sets/yr in range\n      sets in {perc_fished:.0f}% of range",
            fontsize=9.5,
        )
#         change = pop_change[bird_name]
#         change_y = change_y_mapping[change]
#         y_adjust = -change_y
#         ymax = 10
#         ax.arrow(
#             x=15+ x_offset,
#             y=6 + y_adjust,
#             dx=2,
#             dy=change_y,
#             width=0.3,
#             color=arrow_color, #bird_colors[bird_name],
#         )
#         if change == "Unknown":
#             ax.arrow(x=17+ x_offset, y=6, dx=-2, dy=0, width=0.3, color=arrow_color)#bird_colors[bird_name])
        ax.set_axis_off()
        ax.format(titleloc="ul")
        


        ###################
        ## Range Map
        ###################
        ax = axs[(i+1) * 5 + 2]
        ax.format(land=True, zorder=-1)
        geom = dfr[dfr.bird_species == bird_name].range.values[0]
        ax.add_geometries(
            [geom], crs=ccrs.PlateCarree(), color="#444444"
        )  #'#7a85d6')#7a85d6')
        ax.set_axis_off()

        
        ###################
        ## Pie Chart or Bar Chart
        ###################
        ax = axs[(i+1) * 5 + 3]
        dd = dfc[(dfc.bird_name == bird_name) & (dfc.category != "other")]
        sets = dd.sets.values
        labels = dd.category.values
        cycle = pplt.Cycle(colors)
        if i == 0:
            labels = [
                "\n \n \n \nOver dawn,\nmostly day",
                "\n        Entirely Day",
                "Over dusk,\nmostly day",
                "Over dusk,\nmostly night",
                "Entirely night",
                "\tOver dawn,\n\tmostly night\n \n ",
            ]
            wedges, texts = ax.pie(sets[:6], counterclock=False, startangle=90, 
                              labels=labels,textprops={"fontsize": 6})
    
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

#             break
#         if i == 2: break
            
        ###################
        ## Radial chart
        ###################
        ax = axs[(i+1) * 5 + 4]
        d = dfh[dfh.bird_name == bird_name]

        ax.axis("off")
        pie_alpha = 0.7

        # Set the coordinates limits
        upperLimit = 90
        lowerLimit = 20

        #         d = dfs2[dfs2.region == region]
        d["hour"] = d.hours_after_sunrise.apply(lambda x: int(x) % 24)
        d = d.groupby("hour").sum()
        d.sets = d.sets / d.sets.sum()
        d.sets_end = d.sets_end / d.sets_end.sum()
        y = d.sets.values / d.sets.values.sum() / (24 * 6)
        y2 = d.sets_end.values / d.sets_end.values.sum() / (24 * 6)
        x = d.hours_after_sunrise.values
        # Compute max and min in the dataset
        if i == 0:
            max_ = d["sets"].max()

            # Let's compute heights: they are a conversion of each item value in those new coordinates
            # In our example, 0 in the dataset will be converted to the lowerLimit (10)
            # The maximum will be converted to the upperLimit (100)
            slope = (upperLimit - lowerLimit) / max_

        heights = slope * d.sets  # + lowerLimit

        # Compute the width of each bar. In total we have 2*Pi = 360°
        width = 2 * np.pi / 24

        # Compute the angle each bar is centered on:
        indexes = list(range(len(d.index)))
        angles = [-(element - 0.5) * width + 3.1416 / 2 for element in indexes]
        angles

        # Draw bars
        bars = ax.bar(
            x=angles,
            height=heights,
            width=width * 1.8,
            bottom=lowerLimit,
            linewidth=0,
            color="#A900AA",
            alpha=pie_alpha,
            edgecolor="white",
        )

        ## now for the end times

        # Let's compute heights: they are a conversion of each item value in those new coordinates
        # In our example, 0 in the dataset will be converted to the lowerLimit (10)
        # The maximum will be converted to the upperLimit (100)
        #     slope = (upperLimit - lowerLimit) / max_

        heights = slope * d.sets_end  # + lowerLimit

        # Compute the width of each bar. In total we have 2*Pi = 360°

        # Compute the angle each bar is centered on:
        indexes = list(range(len(d.index)))
        angles = [-(element - 0.5) * width + 3.1416 / 2 for element in indexes]
        angles

        # Draw bars
        bars = ax.bar(
            x=angles,
            height=heights,
            width=width * 1.8,
            bottom=lowerLimit,
            linewidth=0,
            alpha=pie_alpha * 0.6,
            edgecolor="white",
            color="#029A38",
        )

        pi = np.pi

        if i == 0:

            for j in range(24):
                x = np.pi * 2 * j / 24
                ax.plot([x, x], [80, 85], color="grey", alpha=0.3)

            fontsize = 8
            range_ = [80, 90]

            ax.text(pi / 2, 95, "Sunrise", ha="center", fontsize=fontsize)
            ax.text(-pi / 2, 95, "+12h", ha="center", va="top", fontsize=fontsize)
            ax.text(0, 95, "+6h", va="center", ha="left", fontsize=fontsize)
            ax.text(-pi, 95, "+18h", va="center", ha="right", fontsize=fontsize)

        ax.plot([pi / 2, pi / 2], range_, color="grey")
        ax.plot([0, 0], range_, color="grey")
        ax.plot([-pi / 2, -pi / 2], range_, color="grey")
        ax.plot([-pi, -pi], range_, color="grey")

        ax.set_ylim(0, 100)

#         ax.format(abc=True)


###

####################33
## add titles
##
######################


ax = axs[0]
ax.set_axis_off()
ax = axs[1]
ax.set_axis_off()

fontsize = 9.5


ax = axs[1]
# highlight_textprops = [{"fontsize":fontsize, "color":'k'},
#                        {"fontsize":fontsize, "color":'k'},
#                        {"fontsize":fontsize, "color":'k'}]
# ax_text(x=0, y=1, va='top',
#          s='<Population status - arrow>\n<indicates if population is increasing,>\n<decreasing, or staying the same>',
#                  highlight_textprops=highlight_textprops,
#          ax=ax)
# ax.set_axis_off()

ax.set_xlim(0, 20)
ymax = 10*1.1/1.5
ax.set_ylim(0, ymax)

fontsize = 9.5

ax.text(x=0, y=ymax,
    s= f"Population status and\nsets in species range",
    fontsize=fontsize, va='top'
)

# ax.arrow(x=0, y=3.5, dx=2, dy=.8, width=0.3, color=arrow_color)#bird_colors[bird_name])
# ax.arrow(x=12.7, y=4.9, dx=2, dy=-.8, width=0.3, color=arrow_color)#bird_colors[bird_name])
# ax.arrow(x=0, y=2, dx=2, dy=0, width=0.3, color=arrow_color)#bird_colors[bird_name])
# ax.arrow(x=11, y=2, dx=2, dy=0, width=0.3, color=arrow_color)#bird_colors[bird_name])
# ax.arrow(x=13, y=2, dx=-2, dy=0, width=0.3, color=arrow_color)#bird_colors[bird_name])
ax.set_axis_off()



ax = axs[2]
highlight_textprops = [{"fontsize":fontsize, "color":'k'}]
ax_text(x=.2, y=1, va='top',
         s='<    Species range>',
                 highlight_textprops=highlight_textprops,
         ax=ax)
ax.set_axis_off()


ax = axs[3]
highlight_textprops = [{"fontsize":fontsize, "color":'k'},
                      {"fontsize":fontsize, "color":'k'}]
ax_text(x=-1.5, y=1, va='top',
         s='<Distribution>\n<of sets>',
                 highlight_textprops=highlight_textprops,
         ax=ax)
ax.set_axis_off()


# ###

ax = axs[4]

starting_color = "#A900AA"
ending_color = "#029A38"
highlight_textprops = [{"fontsize":fontsize, "color":'white', 
                        "bbox": {"facecolor": starting_color,
                                  "linewidth": 0, "pad": 1.5}},
                       {"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'white', 
                        "bbox": {"facecolor": ending_color,
                                  "linewidth": 0, "pad": 1.5}},
                       {"fontsize":fontsize, "color":'k'},
                       {"fontsize":fontsize, "color":'k'}
                      ]
ax_text(x=-.15, y=1, va='top',
         s='<Start>< and ><end>< time of>\n<sets relative to sunrise>',
                 highlight_textprops=highlight_textprops,
         ax=ax)
ax.set_axis_off()



# plt.savefig("birdmaps_thretened_radial.png",dpi=300, bbox_inches='tight')
plt.show()
# -










