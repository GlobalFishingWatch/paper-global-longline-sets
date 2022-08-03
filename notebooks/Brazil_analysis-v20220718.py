# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light
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

# # Compare Brazil Logbook Data with Predictions
#
# [describe this does]
#
# This shows that Brazil logbook data predicts... [quote what we say in the paper]

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns

sns.set_theme()


# +
def convert_time(dates):
    return (
        pd.to_datetime(dates, utc=True) - pd.to_datetime("1970-01-01", utc=True)
    ) // pd.Timedelta("1s")


def seg_intersection_over_union(segA, segB):
    assert segA[1] > segA[0]
    assert segB[1] > segB[0]
    xA = max(segA[0], segB[0])
    xB = min(segA[1], segB[1])

    # compute the length of intersection
    interLength = max(0, xB - xA + 1)
    # compute the Length of both the prediction and ground-truth
    segALength = segA[1] - segA[0] + 1
    segBLength = segB[1] - segB[0] + 1
    # compute the intersection over union by taking the intersection
    # and dividing it by the sum of prediction + ground-truth
    # Lengths - the interesection area
    iou = interLength / float(segALength + segBLength - interLength)
    # return the intersection over union value
    return iou


def gbq(q):
    return pd.read_gbq(q)


# -

# ## Get predicted events

# +
q = f"""

SELECT 
  *, 
  ABS(TIMESTAMP_DIFF(end_time, start_time, minute))/60 AS duration, 
FROM 
  `global-fishing-watch.paper_global_longline_sets.brazil_longline_events_20220802_*` 
Order by id, start_time"""

df_preds = gbq(q)
# -

# ## Get logged events 

# +
# this queries pipe_brazil_production_v20211126.messages_scored_ and bridlife.brazil_longline_logbook_2021,
# which are not public dables

q = f"""
WITH
-- logbook data, start and end of each event: sets and hauls
  logbook AS(
  SELECT
    *,
    -- timestamp in logbook is in UTC -3, convert to gmt
    DATETIME(set_in_timestamp, '+03') AS set_in_timestamp_utc,
    DATETIME(set_end_timestamp, '+03') AS set_end_timestamp_utc,
    DATETIME(haul_in_timestamp, '+03') AS haul_in_timestamp_utc,
    DATETIME(haul_end_timestamp, '+03') AS haul_end_timestamp_utc
  FROM
    `birdlife.brazil_longline_logbook_2021`
),
-- get a list of vms ssvid from each vessel present in logbook table
ssvid_in_logbooks AS(
    SELECT
        DISTINCT ssvid,
        UPPER((split(shipname, '/'))[safe_ordinal(1)]) as shipname,
        n_shipname,
        n_imo
        FROM
    `pipe_brazil_production_v20211126.messages_scored_*`
  WHERE
    _TABLE_SUFFIX between "20210101" AND "20220101"
    AND
    (UPPER(shipname) LIKE '%ALFA%' OR
UPPER(shipname) LIKE '%ANA AMARAL I%' OR
UPPER(shipname) LIKE '%AUSTRIA%' OR
UPPER(shipname) LIKE '%AZTECA III%' OR
UPPER(shipname) LIKE '%BRISA C%' OR
UPPER(shipname) LIKE '%BRISA DO MAR II%' OR
UPPER(shipname) LIKE '%DOM BERNARDO%' OR
UPPER(shipname) LIKE '%DONA ILVA%' OR
UPPER(shipname) LIKE '%EDSON MATHEUS I%' OR
UPPER(shipname) LIKE '%ELIAS SEIF%' OR
UPPER(shipname) LIKE '%ESTRELA DE KALY I%' OR
UPPER(shipname) LIKE '%FILHO DA PROMESSA C%' OR
UPPER(shipname) LIKE '%FLORIPA SL 03%' OR
UPPER(shipname) LIKE '%FLORIPA SL 3%' OR
UPPER(shipname) LIKE '%GUADALAJARA%' OR
UPPER(shipname) LIKE '%IAN CARLOS%' OR
UPPER(shipname) LIKE '%IAN CARLOS S%' OR
UPPER(shipname) LIKE '%IBIZA%' OR
UPPER(shipname) LIKE '%ISADORA I%' OR
UPPER(shipname) LIKE '%IZADORA I%' OR
UPPER(shipname) LIKE '%JOÃO VICTOR IV%' OR
UPPER(shipname) LIKE 'JR LUCAS III%' OR
UPPER(shipname) LIKE 'KADOSH II%' OR
UPPER(shipname) LIKE 'KIYOMA%' OR
UPPER(shipname) LIKE 'KOPESCA I%' OR
UPPER(shipname) LIKE 'KOPESCA IV%' OR
UPPER(shipname) LIKE 'LEAL SANTOS 7%' OR
UPPER(shipname) LIKE 'MACEDO I%' OR
UPPER(shipname) LIKE 'MARBELLA I%' OR
UPPER(shipname) LIKE 'MARIA%' OR
UPPER(shipname) LIKE 'MARIA CLARA%' OR
UPPER(shipname) LIKE 'MARLIN II%' OR
UPPER(shipname) LIKE 'MARTIM VAS%' OR
UPPER(shipname) LIKE 'MARTIM VAZ%' OR
UPPER(shipname) LIKE 'MERIDIANO 3%' OR
UPPER(shipname) LIKE 'NATAL PESCA IX%' OR
UPPER(shipname) LIKE 'NATAL PESCA VII%' OR
UPPER(shipname) LIKE 'NETUNO S%' OR
UPPER(shipname) LIKE 'RIO JAPURÁ%' OR
UPPER(shipname) LIKE 'RIO POTENGI%' OR
UPPER(shipname) LIKE 'RIO POTENGUI%' OR
UPPER(shipname) LIKE 'SAFADI SEIF I%' OR
UPPER(shipname) LIKE 'Y. ABE%' OR
UPPER(shipname) LIKE 'YAMAYA III')
ORDER BY shipname
)

SELECT 
  a.*,
  b.ssvid,
  b.n_shipname,
  b.n_imo
FROM 
  logbook as a
LEFT JOIN 
  ssvid_in_logbooks as b
ON (a.shipname = b.shipname)

"""

df_log = gbq(q)
# -

# ## Clean log data

# Remove logged sets that start at the same time
df_log = df_log.drop_duplicates(
    subset=["shipname", "start_trip", "end_trip", "set_in_timestamp", "ssvid"]
).copy()

df_log = df_log[~df_log.set_number.isna()].copy()
df_log["set_duration"] = pd.to_datetime(df_log.set_end_timestamp_utc) - pd.to_datetime(
    df_log.set_in_timestamp_utc
)
df_log["set_duration_h"] = df_log["set_duration"].dt.total_seconds() / (60 * 60)
# Remove log sets that are too long
df_log = df_log[df_log.set_duration_h <= 15].copy()

df_log = df_log[~df_log.set_end_timestamp_utc.isna()].copy()

# ## Remove vessels that round their start and end times to the hour in the log data
#
# Some vessels recorded set start and end times with only a 1 hour accuracy (that is, they rounded off their times to the nearest hour). We need to eliminate these to provide only higher percision data.

keep_ssvid = []
for ssvid in df_log.ssvid.unique():
    df_log_s = df_log[df_log.ssvid == ssvid].copy().reset_index(drop=True)
    print(ssvid)
    not_rounded = [
        not round(x, 3).is_integer() for x in df_log_s.set_duration_h.unique()
    ]
    print(df_log_s.set_duration_h.unique())
    print(not_rounded)
    if sum(not_rounded) > 1:
        keep_ssvid.append(ssvid)
    df_log_s.set_duration_h.hist(bins=24)
    plt.show()


df_log = df_log[df_log.ssvid.isin(keep_ssvid)].copy().reset_index(drop=True)

# ## Find matching logged sets and predicted sets

# +
matching_sets = []
df_log["start_unix"] = convert_time(df_log["set_in_timestamp_utc"])
df_log["end_unix"] = convert_time(df_log["set_end_timestamp_utc"])

df_preds["start_unix"] = convert_time(df_preds["start_time"])
df_preds["end_unix"] = convert_time(df_preds["end_time"])
all_preds = []
all_gt = []
for ssvid in df_log.ssvid.unique():

    # Get vessel from log data
    print(ssvid)
    track_log = df_log[(df_log.ssvid.astype(str) == str(ssvid))].copy()
    track_log["set_id"] = [
        row["ssvid"] + str(row["start_unix"]) for _, row in track_log.iterrows()
    ]
    min_time = pd.to_datetime(track_log.set_in_timestamp_utc.min(), utc=True)
    max_time = pd.to_datetime(track_log.set_end_timestamp_utc.max(), utc=True)
    # Trim prediction track to log book start end times
    track_preds = (
        df_preds[
            (df_preds.label == "setting")
            & (df_preds.id == ssvid)
            & (df_preds.start_time <= max_time)
            & (df_preds.end_time >= min_time)
        ]
        .copy()
        .reset_index()
    )
    track_preds["set_id"] = [
        row["id"] + str(row["start_unix"]) for _, row in track_preds.iterrows()
    ]

    all_preds.append(track_preds)
    all_gt.append(track_log)
    for idx, row_p in track_preds.iterrows():
        pred = row_p.label

        if pred == "setting":
            for idx2, row_l in track_log.iterrows():
                seg_gt = [row_l["start_unix"], row_l["end_unix"]]
                seg_dt = [row_p["start_unix"], row_p["end_unix"]]
                iou = seg_intersection_over_union(seg_gt, seg_dt)
                overlap = False
                if iou > 0:

                    pred_range = [
                        pd.to_datetime(row_p.start_time, utc=True),
                        pd.to_datetime(row_p.end_time, utc=True),
                    ]
                    log_range = [
                        pd.to_datetime(row_l.set_in_timestamp_utc, utc=True),
                        pd.to_datetime(row_l.set_end_timestamp_utc, utc=True),
                    ]
                    matching_sets.append(
                        [
                            ssvid,
                            pred_range[0],
                            pred_range[1],
                            log_range[0],
                            log_range[1],
                            row_p.duration,
                            row_l.set_duration_h,
                            row_l.set_id,
                            row_p.set_id,
                        ]
                    )
# -

# ## Remove duplicates from matching sets list

df_ms = pd.DataFrame(
    data=matching_sets,
    columns=[
        "ssvid",
        "pred_start",
        "pred_end",
        "log_start",
        "log_end",
        "pred_duration",
        "log_duration",
        "set_id_log",
        "set_id_pred",
    ],
)
df_ms = (
    df_ms.drop_duplicates(
        subset=[
            "ssvid",
            "pred_start",
            "pred_end",
            "log_start",
            "log_end",
            "pred_duration",
            "log_duration",
            "set_id_log",
            "set_id_pred",
        ]
    )
    .reset_index()
    .copy()
)

# # Compute difference in time between log and predicted sets

df_ms["diff_start"] = pd.to_datetime(df_ms["pred_start"]) - pd.to_datetime(
    df_ms["log_start"]
)
df_ms["diff_start"] = df_ms["diff_start"].dt.total_seconds() / (60 * 60)
df_ms["diff_end"] = df_ms["pred_end"] - df_ms["log_end"]
df_ms["diff_end"] = df_ms["diff_end"].dt.total_seconds() / (60 * 60)
df_ms["diff_dur"] = df_ms["pred_duration"] - df_ms["log_duration"]

# ## Check predicted sets that match 2 log sets: the log sets overlap each other 
#
# sometimes the logbook data has two sets at the same time -- there are six (three pairs) of these in the data. We are just ispecting them here.

vc = df_ms.set_id_pred.value_counts()

vc[vc > 1]

df_ms[df_ms.set_id_pred == "49931521632857550"]

# ## Compute results of matching and non matching sets

all_preds = pd.concat(all_preds)
all_gt = pd.concat(all_gt)
all_fps = all_preds[~all_preds.set_id.isin(df_ms.set_id_pred)]
all_fns = all_gt[~all_gt.set_id.isin(df_ms.set_id_log)]

print(len(all_gt))
print(len(all_preds))
print("tp: ", len(df_ms))
print("fp: ", len(all_fps))
print("fn: ", len(all_fns))


# ## Difference in start and end time histogram

df_ms.diff_start.hist(bins=100)

start_mean = df_ms.diff_start.mean()
start_std = df_ms.diff_start.std()
x = np.linspace(-6, 6)
y = np.exp(-0.5 * ((x - start_mean) / start_std) ** 2)
dx = x[1] - x[0]
y = y / y.sum() / dx
plt.plot(x, y)
df_ms.diff_start.hist(bins=100, density=True)


df_ms.diff_end.hist(bins=100)

end_mean = df_ms.diff_end.mean()
end_std = df_ms.diff_end.std()
x = np.linspace(-6, 6)
y = np.exp(-0.5 * ((x - end_mean) / end_std) ** 2)
dx = x[1] - x[0]
y = y / y.sum() / dx
plt.plot(x, y)
df_ms.diff_end.hist(bins=100, density=True)


# # ratio of set start and end times within 2 hours difference

sum(df_ms.diff_start.abs() <= 2) / len(df_ms)

sum(df_ms.diff_end.abs() <= 2) / len(df_ms)

# ## Mean start and end difference (minutes)

df_ms.diff_start.mean() * 60

df_ms.diff_end.mean() * 60

df_ms.diff_dur.mean() * 60

# ## Standard deviation of start and end time differences (hours)

df_ms.diff_start.std()

df_ms.diff_end.std()

df_ms.diff_dur.std()




