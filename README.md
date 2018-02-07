# Evren

![Evren](https://i.imgur.com/ZHrX1l3.png)

[![Build Status](https://travis-ci.org/vertexclique/evren.svg?branch=master)](https://travis-ci.org/vertexclique/evren)
[![codecov.io](https://codecov.io/gitlab/hbetts/orbitalpy/coverage.svg?branch=master)](https://codecov.io/gitlab/vertexclique/evren?branch=master)

**NOTE:** Project is in experimental phase.

**Evren** is pre-optimizer and exporter for machine learning models to embedded systems, android and iOS platforms.
Currently Android and iOS supported, for iOS it can export `caffe` and `keras` models, for Android it exports `keras` models.
Keras models should use Tensorflow backend.

## Installation

After cloning use `tox` to setup environment.
```bash
$ tox
```

## Usage

You can find examples in `examples` directory. Documentation will be made available soon.

Minimal export code can be:

```python
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

# Write Keras model with weights as protobuf file
# for use with Tensorflow Lite and Android NDK
# It will produce `dense_android_test.pb` with weights embedded in.
export_keras_to_file(model_name=MODEL_NAME,
                     input_names=map(str, range(1, 10)),
                     output_names=output_nodes,
                     out_file_name=MODEL_NAME)
```

## TODO

* Caffee export for Android
* Extra optimization techniques for Android and iOS.
* Travis setup with Tox.

## Contributing

Read [CONTRIBUTING](CONTRIBUTING.md).
