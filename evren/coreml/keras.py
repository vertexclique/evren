
import coremltools
import keras


FULL_PRECISION = coremltools.models.model._MLMODEL_FULL_PRECISION
HALF_PRECISION = coremltools.models.model._MLMODEL_HALF_PRECISION


def __custom_layer_loader(model, custom_objects):
    return keras.models.load_model(model, custom_objects=custom_objects)


def export_keras(model, custom_layer_objects=None, model_precision=FULL_PRECISION):
    try:
        mlmodel = coremltools.converters.keras.convert(
            model=model,
            model_precision=model_precision
        )
    except ValueError as e:
        if "Unknown layer" in str(e):
            if not custom_layer_objects:
                raise ValueError("Pass custom_layer_objects to resolve custom classes.")
            else:
                custom_model = __custom_layer_loader(model, custom_layer_objects)
                mlmodel = coremltools.converters.keras.convert(
                    model=custom_model,
                    model_precision=model_precision
                )
    return mlmodel


def export_keras_to_file(model, filename, custom_layer_objects=None, model_precision=FULL_PRECISION):
    coreml_model = export_keras(model, custom_layer_objects, model_precision)
    coreml_model.save(filename)
