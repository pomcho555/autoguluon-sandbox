import json
import os
from io import StringIO

import numpy as np
import pandas as pd
from autogluon.vision import ImagePredictor


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    model = ImagePredictor.load(model_dir)
    globals()["column_names"] = model.feature_metadata_in.get_features()

    return model


def transform_fn(
    model, request_body, input_content_type, output_content_type="application/json"
):

    if input_content_type == "image/jpeg":
        buf = StringIO(request_body)
        # TODO: Write transform process
        data = buf
    else:
        raise NotImplementedError(f"{input_content_type} content type not supported")

    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    prediction = pd.concat([pred, pred_proba], axis=1).values

    return json.dumps(prediction.tolist()), output_content_type
