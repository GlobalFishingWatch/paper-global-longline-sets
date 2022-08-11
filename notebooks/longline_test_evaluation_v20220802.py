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

# # Evaluate longline model
# This notebook produces the `precision` and `recall` for the `longline model`, as described in `Materials and Methods`. With a test set of 100 days of longlining, predicted sets that overlap with ground truth sets are counted as true positives. 
#
# "Each predicted set within the selected days was checked for overlap with the ground truth sets. If there was overlap between a predicted set and a ground truth set, this was recorded as a true positive (TP). Predicted sets for which there were no overlapping ground truth sets were recorded as false positives (FP). Ground truth sets for which there were no overlapping predicted sets were recorded as false negatives (FN). `Recall, computed as TP/(TP+FN) was 90% and precision, computed as TP/(TP+FP), was 98%`. The precision and recall for each region were mostly consistent with these results. This result suggests that our model is conservative; we may actually be missing some from vessels, but the sets that are identified are likely done so correctly."

import pandas as pd


# +
def gbq(q):
    return pd.read_gbq(q, project_id="global-fishing-watch")


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


def convert_time(dates):
    return (
        pd.to_datetime(dates, utc=True) - pd.to_datetime("1970-01-01", utc=True)
    ) // pd.Timedelta("1s")


# -

# ## Load ground truth 

df_seg_gt = pd.read_csv("longline_groundtruth_sets_v20220802.csv")
df_seg_gt["mmsi"] = df_seg_gt["mmsi"].astype(str)
df_seg_gt.start_time = pd.to_datetime(df_seg_gt.start_time)
df_seg_gt.end_time = pd.to_datetime(df_seg_gt.end_time)



ssvid_list = tuple(df_seg_gt.mmsi.unique())

# ## Get list of days and mmsi that were selected randomly for testing

test_sets = pd.read_csv("longline_test_days_v20220613.csv")
test_sets.mmsi = test_sets.mmsi.astype(str)

len(test_sets.mmsi.unique())

# ## Get model predictions for ssvid in test set

# +
q = f"""

SELECT * FROM `paper_global_longline_sets.longline_sets_categorised_v20220801`
where ssvid in {ssvid_list}
Order by ssvid, start_time"""

df_preds = gbq(q)
# -

# # find overlapping predicted sets and ground truth sets
# When matching predicted sets to ground truth sets (to find false positives), include ground truth sets that occurred the day before and the day after the test day.
#
# When matching ground truth sets to predicted sets (to find false negatives), include predicted sets that occurred the day before and the day after the test day.

bad_tracks = ["431005009", "710096180", "601423000"]
matching_sets = []
matching_sets_gtpad = []
matching_sets_dtpad = []
all_dt = []
all_gt = []
test_sets = test_sets[~test_sets.mmsi.isin(bad_tracks)].copy()
for index, row in test_sets.iterrows():

    ssvid = row.mmsi
    start_str = str(row.year) + "-" + str(row.month) + "-" + str(row.day)
    print(index, ": ", row.mmsi, start_str)

    start_date = pd.to_datetime(start_str, utc=True)
    start_date_pad = pd.to_datetime(start_str, utc=True) + pd.DateOffset(-1)
    end_date = pd.DatetimeIndex([start_date]) + pd.DateOffset(1)
    end_date_pad = pd.DatetimeIndex([start_date]) + pd.DateOffset(2)
    end_str = (
        str(end_date.year[0])
        + "-"
        + str(end_date.month[0])
        + "-"
        + str(end_date.day[0])
    )
    end_str_pad = (
        str(end_date_pad.year[0])
        + "-"
        + str(end_date_pad.month[0])
        + "-"
        + str(end_date_pad.day[0])
    )
    start_str_pad = (
        str(start_date_pad.year)
        + "-"
        + str(start_date_pad.month)
        + "-"
        + str(start_date_pad.day)
    )
    #     print(end_str)
    track_gt = (
        df_seg_gt[
            (df_seg_gt.mmsi == ssvid)
            & (df_seg_gt.label == "setting")
            & (df_seg_gt.end_time >= start_str)
            & (df_seg_gt.start_time <= end_str)
        ]
        .copy()
        .reset_index()
    )
    track_gt_pad = (
        df_seg_gt[
            (df_seg_gt.mmsi == ssvid)
            & (df_seg_gt.label == "setting")
            & (df_seg_gt.end_time >= start_str_pad)
            & (df_seg_gt.start_time <= end_str_pad)
        ]
        .copy()
        .reset_index()
    )
    track_dt = (
        df_preds[
            (df_preds.mmsi == ssvid)
            & (df_preds.label == "setting")
            & (df_preds.end_time >= start_str)
            & (df_preds.start_time <= end_str)
        ]
        .copy()
        .reset_index()
    )
    track_dt_pad = (
        df_preds[
            (df_preds.mmsi == ssvid)
            & (df_preds.label == "setting")
            & (df_preds.end_time >= start_str_pad)
            & (df_preds.start_time <= end_str_pad)
        ]
        .copy()
        .reset_index()
    )
    track_gt["start_unix"] = convert_time(track_gt["start_time"])
    track_gt["end_unix"] = convert_time(track_gt["end_time"])
    track_gt_pad["start_unix"] = convert_time(track_gt_pad["start_time"])
    track_gt_pad["end_unix"] = convert_time(track_gt_pad["end_time"])
    track_dt["start_unix"] = convert_time(track_dt["start_time"])
    track_dt["end_unix"] = convert_time(track_dt["end_time"])
    track_dt_pad["start_unix"] = convert_time(track_dt_pad["start_time"])
    track_dt_pad["end_unix"] = convert_time(track_dt_pad["end_time"])
    track_dt["region"] = row.region
    track_gt["region"] = row.region
    track_gt["test_index"] = index
    track_dt["test_index"] = index

    all_dt.append(track_dt)
    all_gt.append(track_gt)
    for _, row_gt in track_gt.iterrows():
        matches = 0
        for _, row_dt in track_dt.iterrows():
            seg_gt = [row_gt["start_unix"], row_gt["end_unix"]]
            seg_dt = [row_dt["start_unix"], row_dt["end_unix"]]
            iou = seg_intersection_over_union(seg_gt, seg_dt)
            if iou > 0:

                matching_sets.append(
                    [
                        ssvid,
                        row_gt.start_time,
                        row_gt.end_time,
                        row_dt.start_time,
                        row_dt.end_time,
                        iou,
                        row.region,
                        ssvid + str(row_gt["start_unix"]),
                        row_dt.set_id,
                    ]
                )
    for _, row_gt in track_gt_pad.iterrows():
        matches = 0
        for _, row_dt in track_dt.iterrows():
            seg_gt = [row_gt["start_unix"], row_gt["end_unix"]]
            seg_dt = [row_dt["start_unix"], row_dt["end_unix"]]
            iou = seg_intersection_over_union(seg_gt, seg_dt)
            if iou > 0:

                matching_sets_gtpad.append(
                    [
                        ssvid,
                        row_gt.start_time,
                        row_gt.end_time,
                        row_dt.start_time,
                        row_dt.end_time,
                        iou,
                        row.region,
                        ssvid + str(row_gt["start_unix"]),
                        row_dt.set_id,
                    ]
                )
    for _, row_gt in track_gt.iterrows():
        matches = 0
        for _, row_dt in track_dt_pad.iterrows():
            seg_gt = [row_gt["start_unix"], row_gt["end_unix"]]
            seg_dt = [row_dt["start_unix"], row_dt["end_unix"]]
            iou = seg_intersection_over_union(seg_gt, seg_dt)
            if iou > 0:

                matching_sets_dtpad.append(
                    [
                        ssvid,
                        row_gt.start_time,
                        row_gt.end_time,
                        row_dt.start_time,
                        row_dt.end_time,
                        iou,
                        row.region,
                        ssvid + str(row_gt["start_unix"]),
                        row_dt.set_id,
                    ]
                )

all_dt = pd.concat(all_dt)
all_gt = pd.concat(all_gt)

df_ms = pd.DataFrame(
    data=matching_sets,
    columns=[
        "ssvid",
        "gt_start",
        "gt_end",
        "dt_start",
        "dt_end",
        "iou",
        "region",
        "gt_set_id",
        "dt_set_id",
    ],
)
df_ms = df_ms.drop_duplicates().copy()

df_ms_gtl = pd.DataFrame(
    data=matching_sets_gtpad,
    columns=[
        "ssvid",
        "gt_start",
        "gt_end",
        "dt_start",
        "dt_end",
        "iou",
        "region",
        "gt_set_id",
        "dt_set_id",
    ],
)
df_ms_gtl = df_ms_gtl.drop_duplicates().copy()

df_ms_dtl = pd.DataFrame(
    data=matching_sets_dtpad,
    columns=[
        "ssvid",
        "gt_start",
        "gt_end",
        "dt_start",
        "dt_end",
        "iou",
        "region",
        "gt_set_id",
        "dt_set_id",
    ],
)
df_ms_dtl = df_ms_dtl.drop_duplicates().copy()

# ## Compute false positives 

all_fps = all_dt[~all_dt.set_id.isin(df_ms_gtl.dt_set_id)]

# ## Compute false negatives 

all_gt["set_id"] = [
    row["mmsi"] + str(row["start_unix"]) for _, row in all_gt.iterrows()
]
all_fns = all_gt[~all_gt.set_id.isin(df_ms_dtl.gt_set_id)]

print("number of ground truth sets: ", len(all_gt))
print("number of detected sets: ", len(all_dt))
print("detection tp: ", len(df_ms_gtl))
print("ground truth tp: ", len(df_ms_dtl))
print("fp: ", len(all_fps))
print("fn: ", len(all_fns))

# ## Precision and recall

precision = len(df_ms_gtl) / (len(df_ms_gtl) + len(all_fps))
print("precision: ", round(precision, 2))
recall = len(df_ms_dtl) / (len(df_ms_dtl) + len(all_fns))
print("recall: ", round(recall, 2))

# ## Precision and recall by region 

regions = [
    "North_Pacific",
    "South_Pacific",
    "South_Atlantic",
    "South_Indian",
    "Other_Region",
]
for region in regions:
    df_ms_r_dt = df_ms_gtl[df_ms_gtl.region == region].copy()
    df_ms_r_gt = df_ms_dtl[df_ms_dtl.region == region].copy()
    all_fps_r = all_fps[all_fps.region == region].copy()
    all_fns_r = all_fns[all_fns.region == region].copy()
    print(region)
    precision = len(df_ms_r_dt) / (len(df_ms_r_dt) + len(all_fps_r))
    print("precision: ", round(precision, 2))
    recall = len(df_ms_r_gt) / (len(df_ms_r_gt) + len(all_fns_r))
    print("recall: ", round(recall, 2))


