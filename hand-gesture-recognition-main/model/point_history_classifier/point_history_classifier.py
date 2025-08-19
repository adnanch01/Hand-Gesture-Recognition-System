#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class PointHistoryClassifier(object):
    def __init__(
        self,
        model_path='model/point_history_classifier/point_history_classifier.tflite',
        score_th=0.5,
        invalid_value=0,
        backend='tflite',  # 'tflite' or 'keras'
        num_threads=1,
    ):
        self.backend = backend
        self.score_th = score_th
        self.invalid_value = invalid_value
        if self.backend == 'tflite':
            self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model = tf.keras.models.load_model(model_path)

    def __call__(self, point_history):
        if self.backend == 'tflite':
            idx = self.input_details[0]['index']
            self.interpreter.set_tensor(idx, np.array([point_history], dtype=np.float32))
            self.interpreter.invoke()
            out_idx = self.output_details[0]['index']
            result = np.squeeze(self.interpreter.get_tensor(out_idx))
            ridx = int(np.argmax(result))
            if result[ridx] < self.score_th:
                return self.invalid_value
            return ridx
        else:
            pred = self.model.predict(np.array([point_history], dtype=np.float32), verbose=0)[0]
            ridx = int(np.argmax(pred))
            if pred[ridx] < self.score_th:
                return self.invalid_value
            return ridx
