from sagemaker import image_uris
from sagemaker.deserializers import JSONDeserializer

# from sagemaker.estimator import Framework
from sagemaker.mxnet import MXNetModel
from sagemaker.mxnet.model import MXNetPredictor
from sagemaker.predictor import Predictor


class AutoGluonPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, serializer=None, deserializer=JSONDeserializer(), **kwargs
        )


class AutoGluonInferenceModel(MXNetModel):
    def __init__(
        self,
        model_data,
        role,
        entry_point,
        region,
        framework_version,
        py_version,
        instance_type,
        **kwargs,
    ):
        image_uri = image_uris.retrieve(
            "autogluon",
            region=region,
            version=framework_version,
            py_version=py_version,
            image_scope="inference",
            instance_type=instance_type,
        )
        super().__init__(
            model_data,
            role,
            entry_point,
            image_uri=image_uri,
            framework_version="1.8.0",
            **kwargs,
        )
