"""
evren tests.
"""


import unittest
import subprocess

from coremltools.models import model
from evren.coreml.keras import *

from keras.models import Sequential
from keras.layers.core import Dense


class TestCoreMLKeras(unittest.TestCase):
    """
    Test evren's CoreML Keras Exporter.
    """

    def setUp(self):
        self.model = Sequential()
        self.model.add(Dense(10, input_shape=(10, 1)))
        self.model.add(Dense(10))
        self.model.add(Dense(10))
        self.model.add(Dense(10))
        self.model.add(Dense(10))
        self.model.add(Dense(10))

    def test_export_keras(self):
        mlmodel = export_keras(self.model)

        self.assertEqual(
            isinstance(mlmodel, model.MLModel),
            True
        )

    def test_export_keras_to_file(self):
        filename = 'test.mlmodel'
        export_keras_to_file(self.model, filename)

        result = subprocess.run(['file', filename], stdout=subprocess.PIPE)
        self.assertIn('PDP-11 pure executable', result.stdout.decode('utf-8'))
