"""
evren tests.
"""


import unittest
import subprocess
import importlib

from evren.android.keras import *

from keras.models import Sequential
from keras.layers.core import Dense


class TestAndroidKeras(unittest.TestCase):
    """
    Test evren's Android Keras Exporter.
    """

    model = Sequential()
    model.add(Dense(10, input_shape=(10, 1)))
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Dense(1, activation='sigmoid'))

    output_nodes = get_output_node_names()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_export_keras(self):
        android_model = export_keras('android_test',
                     input_names=map(str, range(1, 10)),
                     output_names=self.output_nodes)

        self.assertEqual(
            str(android_model).count('op: "Sigmoid"'), 1
        )

        self.assertIn("dense_", str(android_model))

    def test_export_keras_to_file(self):
        export_keras_to_file(model_name='android_test',
                             input_names=map(str, range(1, 10)),
                             output_names=self.output_nodes,
                             out_file_name='android_test')