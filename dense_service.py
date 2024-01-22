import tensorflow
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np

DENSE_MODEL = load_model('models/model_dense_final.keras')


def dense_predict(params):
    input_arr = []
    input_arr.append(1.0 - float(params['div_area']))
    input_arr.append(1.0 - float(params['div_dist']))
    input_arr.append(float(params['resnet']))
    input_arr.append(float(params['conv_next']))
    expanded_array = np.expand_dims(input_arr, axis=0)
    predict = DENSE_MODEL.predict(expanded_array)
    return {
        'predict': predict[0][0]
    }


if __name__ == '__main__':
    dense_predict({})
    # print(yolo_predict('C1_0050_D4.png', 0.6875))
    # print(segm_unet('C1_0050_D4.png', 0.6875))
    pass
