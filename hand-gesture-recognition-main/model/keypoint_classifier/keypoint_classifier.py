#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        backend='tflite',  # 'tflite' or 'keras'
        num_threads=1,
    ):
        self.backend = backend
        if self.backend == 'tflite':
            self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model = tf.keras.models.load_model(model_path)

    def __call__(self, landmark_list):
        if self.backend == 'tflite':
            idx = self.input_details[0]['index']
            self.interpreter.set_tensor(idx, np.array([landmark_list], dtype=np.float32))
            self.interpreter.invoke()
            out_idx = self.output_details[0]['index']
            result = self.interpreter.get_tensor(out_idx)
            return int(np.argmax(np.squeeze(result)))
        else:
            pred = self.model.predict(np.array([landmark_list], dtype=np.float32), verbose=0)
            return int(np.argmax(pred[0]))
