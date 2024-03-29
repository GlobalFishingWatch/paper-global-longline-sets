# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light
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

# # Compare Brazil Logbook Data with Predictions
#
# This notebook matches predicted sets to sets reported in logbooks from Brazil. For matching sets, the difference in start time, end times, and duration is computed.
#
# Assessing model accuracy: "out of 855 sets in logbook data, 169 were not identified by the model, giving a recall of 80 % (the same as accuracy), suggesting that sets may be undercounted by the model. The start and end times of the remaining 686 sets, though, were accurately estimated, especially in the aggregate. The mean start time of the model was, on average, 2 min earlier than reported set time, and 8 min earlier than the reported end time. The standard deviation of the difference of start time and end time was 1.8 and 1.7 h, respectively."

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

# ## Get logged events 

# +
# this queries birdlife.brazil_logbook_test_events,
# which is not a public table

q = f"""
SELECT * FROM `birdlife.brazil_logbook_test_events`
"""

df_log = gbq(q)
# -

# ## Get predicted events

# +
# this queries birdlife.brazil_longline_events_20220802_,
# which is not a public table
q = f"""

SELECT 
  *, 
  ABS(TIMESTAMP_DIFF(end_time, start_time, minute))/60 AS duration, 
FROM 
  `birdlife.brazil_longline_events_20220802_*` 
Order by id, start_time"""

df_preds = gbq(q)
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
    not_rounded = [
        not round(x, 3).is_integer() for x in df_log_s.set_duration_h.unique()
    ]
    if sum(not_rounded) > 1:
        keep_ssvid.append(ssvid)



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
#     print(ssvid)
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

df_ms["diff_start"] = df_ms["pred_start"] - df_ms["log_start"]
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

# +
print("number of sets: ",len(all_gt))
print("number of predictions", len(all_preds))
print("true positives: ", len(df_ms))
print("false positives: ", len(all_fps))
print("false negatives: ", len(all_fns))

recall = len(df_ms) / (len(df_ms) + len(all_fns))
print("recall: ", round(recall, 2))

# -


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




