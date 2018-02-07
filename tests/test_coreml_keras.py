
import unittest
import coremltools.models.model as MLModel
from evren.coreml.keras import *

from keras.models import Sequential
from keras.layers.core import Dense

class TestCoreML(unittest.TestCase):
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
            isinstance(mlmodel, MLModel),
            True
        )
