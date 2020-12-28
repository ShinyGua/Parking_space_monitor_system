from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np
import tensorflow as tf
import copy

interpreter = Interpreter("tflite_model_64.tflite")


def predict(img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, height, width, _ = input_details[0]['shape']

    img = copy.deepcopy(cv2.resize(img, (width, height)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(width, height))
    img = tf.convert_to_tensor(img)
    img = img / 255
    img = tf.reshape(img,[1, width, height, 3])

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()
    results = interpreter.get_tensor(output_details[0]['index'])
    out = results[0][0]
    if out < 1:
        out = -1
    else:
        out = 1
    #print(out)
    return out

