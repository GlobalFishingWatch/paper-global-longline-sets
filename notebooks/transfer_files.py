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

import subprocess
import pandas as pd

q = '''select distinct _table_suffix t from 
`world-fishing-827.machine_learning_dev_ttl_120d.longline_fishing_*` 
order by t'''
df = pd.read_gbq(q)

for t in df.t.values[1:]:
    try:
        int(t)
        command = f'bq cp machine_learning_dev_ttl_120d.longline_fishing_{t} global-fishing-watch:paper_global_longline_sets.longline_fishing_{t}'
        subprocess.call(command.split())
    except:
        continue

df.t.values


