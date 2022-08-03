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

# # Categorize sets to every five minutes

# +
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from jinja2 import Template
import subprocess
import numpy as np

from google.cloud import bigquery
client = bigquery.Client()

# -

def query_to_table(query, table_id, max_retries=100, retry_delay=60):
    for _ in range(max_retries):

        config = bigquery.QueryJobConfig(
            destination=table_id, write_disposition="WRITE_TRUNCATE"
        )

        job = client.query(query, job_config=config)

        if job.error_result:
            err = job.error_result["reason"]
            msg = job.error_result["message"]
            if err == "rateLimitExceeded":
                print(f"retrying... {msg}")
                time.sleep(retry_delay)
                continue
            elif err == "notFound":
                print(f"skipping... {msg}")
                return
            else:
                raise RuntimeError(msg)

        job.result()  # wait to complete
        print(f"completed {table_id}")
        return

    raise RuntimeError("max_retries exceeded")


# used to be scratch_joanna.sets_24hr_categorised_v20211104
sets_categorized_table = 'birdlife.longline_sets_categorised_v20220701'


sets_5min_table = "birdlife.sets_5min_cat_v20220701"

command = f"bq mk --time_partitioning_type=DAY {sets_5min_table}"
subprocess.run(command.split())


with open('../queries/categorize_daily.sql.j2', 'r') as f:
    query_template = Template(f.read())





def process_day(the_date):
    query = query_template.render(the_date=f"{the_date:%Y-%m-%d}",
                                 sets_categorized_table=sets_categorized_table)
    
    table_id= f"world-fishing-827.{sets_5min_table}${the_date:%Y%m%d}"
    query_to_table(query, table_id)
#     return(query)


process_day(datetime(2021,1,1))

the_dates = np.arange(datetime(2017,1,1), datetime(2020,12,31), timedelta(days=1)).astype(datetime)


# +

with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates:
        e.submit(process_day, d)
# -


# # Grid sets 
#
# Identify all grid cells, at a 10th of a degree, that are within 30km of a setting longline on a given day

# +
# # !bq rm -f birdlife.sets_grid_10
# -

grid_table10 = "birdlife.sets_grid_10_v20220701"


command = f"bq mk --time_partitioning_type=DAY {grid_table10}"
subprocess.run(command.split())


# +

    

def process_day_grid(the_date, one_over_cellsize, grid_table):
    with open('../queries/grid_daily.sql.j2', 'r') as f:
        query_template2 = Template(f.read())
    
    query = query_template2.render(the_date=f"{the_date:%Y-%m-%d}",
                                   one_over_cellsize=one_over_cellsize,
                                  sets_5min_table=sets_5min_table)
    
    table_id= f"world-fishing-827.{grid_table}${the_date:%Y%m%d}"
    query_to_table(query, table_id)
#     return(query)


# -

process_day_grid(datetime(2020,1,1),10,grid_table10)

the_dates = np.arange(datetime(2017,1,1), datetime(2020,12,31), timedelta(days=1)).astype(datetime)


with ThreadPoolExecutor(max_workers=16) as e:
    for d in the_dates[16:]:
        e.submit(process_day_grid ,d, 10, grid_table10)




grid_table20 = "birdlife.sets_grid_20_v20220701"
command = f"bq mk --time_partitioning_type=DAY {grid_table20}"
subprocess.run(command.split())

process_day_grid(datetime(2020,5,1),20, grid_table20)

process_day_grid(datetime(2020,5,2),20, grid_table20)

# # Grid sets 
#
# Identify all grid cells, at a 40th of a degree, that are within 30km of a setting longline on a given day, for one day for figure 1

grid_table40 = "birdlife.sets_grid_40_v20220701"
command = f"bq mk --time_partitioning_type=DAY {grid_table40}"
subprocess.run(command.split())

process_day_grid(datetime(2020,5,1),40, grid_table40)

process_day_grid(datetime(2020,5,2),40, grid_table40)


