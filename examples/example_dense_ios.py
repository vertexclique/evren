
from evren.ios.keras import *

from keras.models import Sequential
from keras.layers.core import Dense


def main():
    MODEL_FILE = "dense_ios_test.mlmodel"

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

    # Export Keras (Low-level Tensorflow) model as raw data

    ios_model = export_keras(model)
    print(str(ios_model))

    # Or write Keras model with weights as PDP-11 file
    # for use with Apple CoreML SDK

    export_keras_to_file(model, MODEL_FILE)


if __name__ == "__main__":
    main()