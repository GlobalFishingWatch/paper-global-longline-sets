# flake8: noqa
# isort: skip_file
import os
import pickle
import sys
from collections import Counter, defaultdict, namedtuple

import tensorflow as tf

import gcsfs
import h5py
import matplotlib.pyplot as plts

# +
import numpy as np
import pandas as pd
from skimage import morphology
from sklearn.model_selection import KFold

sys.path.append("../..")

from fishing_detection_model.model import fishing_model_attn6, generic_fishing_model
from fishing_detection_model.training import longline_trainer, longline_training_data
from fishing_detection_model.training import training_data as training_data_mod

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# -

training_data = longline_training_data.load("../data/untracked/tim_data_20211026.h5")

model = fishing_model_attn6.FishingModelAttn(
    inputs=[
        "x",
        "y",
        "cos_course_degrees",
        "sin_course_degrees",
        "speed_knots",
    ],
    outputs=["other", "setting", "hauling"],
    sample_length=97,
    sample_interval=3,
)

# ## Train Model

results = longline_trainer.LonglineTrainer.train(
    fishing_model_attn6.FishingModelAttn, training_data, fold=0
)

if True:
    results = longline_trainer.LonglineTrainer.train(
        fishing_model_attn6.FishingModelAttn, training_data, fold=None
    )
    results.model.save_model("../data/untracked/fishing_model_attn6_vXXXXXXXX.hdf5")

# To load the model use:
#
#     loaded_model = generic_fishing_model.GenericFishingModel.load_model(
#         "../data/untracked/fishing_model_attn6_vXXXXXXXX.hdf5")
