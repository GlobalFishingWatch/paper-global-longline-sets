# paper-global-longline-sets
This repository contains code supporting the paper "Global prevalence of setting longlines at dawn highlights bycatch risk for threatened albatross." 

## Overview
One of the most effective ways to reduce seabird bycatch is for pelagic longliners to set their hooks entirely at night, when albatross are least active. The purpose of this repository is to archive the code used in the paper. The code produces estimates for global coverage of longline sets, overlap with albatross regions, ratio of sets happening at night in different regions, and an evaluation of the machine learning model.

**Authors**: David Kroodsma<sup>1</sup>, Joanna Turner<sup>1</sup>, Cian Luck<sup>1</sup>, Tim Hochberg<sup>1</sup>, Nathan Miller<sup>1</sup>, Philip Augustyn<sup>2</sup>, Stephanie Prince<sup>2</sup>

Affiliations: 
<sup>1</sup>Global Fishing Watch, Washington, DC 20036, USA 
<sup>2</sup>BirdLife International Marine Programme, RSPB, The Lodge, Sandy, Bedfordshire, SG19 2DL, UK

## Code

**Note**: The raw AIS data inputs to this analysis can not be made public due to data licensing restrictions and, as a result, some of the code cannot be run externally.

[fishing_detection_model/](fishing_detection_model/): Directory containing the code for the model that identifies longline sets.

[notebooks/](notebooks/): Directory containing notebooks with analyses and figures for the paper.

[queries/](queries/): Directory containing queries for building BigQuery tables used in the analyses.

[data/](data/): Directory containing data to support the creation of tables and figures.

## Data

Data is available (and referenced in the notebooks in this repo) in the public BigQuery dataset `global-fishing-watch:paper_global_longline_sets`.

```python

```
