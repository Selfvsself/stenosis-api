import tensorflow
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np

IMG_HEIGHT = 224
IMG_WIDTH = 224


class LayerScale(layers.Layer):
    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tensorflow.Variable(
            self.init_values * tensorflow.ones((self.projection_dim,))
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


RESNET_MODEL = load_model('models/model_resnet_final.keras')
CONV_NEXT_MODEL = load_model('models/model_convnext_final.keras', compile=False,
                             custom_objects={"LayerScale": LayerScale})


def resnet_predict(image_path):
    img = tensorflow.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tensorflow.keras.utils.img_to_array(img)
    expanded_array = np.expand_dims(img_array, axis=0)
    predict = RESNET_MODEL.predict(expanded_array)
    return {
        'predict': predict[0][0]
    }


def conv_next_predict(image_path):
    img = tensorflow.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tensorflow.keras.utils.img_to_array(img)
    expanded_array = np.expand_dims(img_array, axis=0)
    predict = CONV_NEXT_MODEL.predict(expanded_array)
    return {
        'predict': predict[0][0]
    }
