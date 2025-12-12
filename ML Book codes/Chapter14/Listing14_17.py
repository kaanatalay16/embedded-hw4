import tensorflow as tf
import keras
from tflite2cc import convert_tflite2cc

TFLITE_MODEL_PATH = "MNIST_LSTM_model.tflite"
MNIST_LSTM_model = keras.models.load_model("MNIST_LSTM_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(MNIST_LSTM_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
MNIST_LSTM_lite = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as tflite_file:
    tflite_file.write(MNIST_LSTM_lite)

convert_tflite2cc(TFLITE_MODEL_PATH, "MNIST_LSTM_model.cc")
