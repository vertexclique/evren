
from evren.android.keras import *

from keras.models import Sequential
from keras.layers.core import Dense


def main():
    MODEL_NAME = "dense_android_test"

    # Define Keras model
    model = Sequential()
    model.add(Dense(10, input_shape=(10, 1)))
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Dense(1, activation='sigmoid'))

    # Train it in here, just because it is demonstration we ignore that
    # . . . . . . TRAINING . . . . . .

    # If you know the output nodes you
    # don't need to call underlying output nodes list

    output_nodes = get_output_node_names()

    # Export Keras (Low-level Tensorflow) model as raw data

    android_model = export_keras(MODEL_NAME,
                                 input_names=map(str, range(1, 10)),
                                 output_names=output_nodes)
    print(str(android_model))

    # Or write Keras model with weights as protobuf file
    # for use with Tensorflow Lite and Android NDK

    export_keras_to_file(model_name=MODEL_NAME,
                         input_names=map(str, range(1, 10)),
                         output_names=output_nodes,
                         out_file_name=MODEL_NAME)


if __name__ == "__main__":
    main()