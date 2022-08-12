# Longline Fishing Detection Model

## Model

The model is located in the [fishing_model_attn6](fishing_detection_model/model/fishing_model_attn6.py) in the
`fishing_detection_model.model` package.

An example of how to load a model with pretrained weights and predict on some provided synthetic data is shown
[here](fishing_detection_model/model/inference_demo.ipynb). This also includes a demo of how to extract events
from the raw predictions.

### Pretrained Weights

Weights can be downloaded from gs://paper-global-longline-sets/fishing_model_attn6_v20211027.hdf5. Install
`gsutil` as described [here](https://cloud.google.com/storage/docs/gsutil_install) then copy the file locally:

        gsutil cp gs://paper-global-longline-sets/fishing_model_attn6_v20211027.hdf5 \
                  fishing_detection_model/data/untracked/

## Training

Training support code is located in the `fishing_detection_model.training` package. An example of how the model
is trained is shown in [train_longline_fishing](fishing_detection_model/training/train_longline_fishing.ipynb).
Unfortunately, we cannot currently provide the data used to train the model due to licensing restrictions, but
pre-trained model weights are available as discussed above.

