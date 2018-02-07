import coremltools


FULL_PRECISION = coremltools.models.model._MLMODEL_FULL_PRECISION
HALF_PRECISION = coremltools.models.model._MLMODEL_HALF_PRECISION


def export_caffe(model, model_precision=FULL_PRECISION):
    return coremltools.converters.keras.convert(
        model=model,
        model_precision=model_precision
    )


def export_to_file(coreml_model, filename):
    coreml_model.save(filename)
