"""
Taejoon Byun

Model inspired by the end-to-end model from NVIDIA

"End to End Deep Learning For Self-Driving Cars
- https://images.nvidia.com/content/tegra/automotive/images/2016/solution
        /pdf/end-to-end-dl-using-px.pdf
   
"""


from keras.models import Sequential
from keras.layers import Conv2D, Dense,Dropout
from keras.layers import Activation, Flatten, BatchNormalization
from keras.regularizers import l2


def SimpleModel(input_shape=(300,120,3), weight_penalty=0.0001, momentum=.9):
    """Creates a simple image regressor model with specified parameters
    :param input_shape: shape of the input tensor
    :returns: a Keras Model
    """
    model = Sequential() 
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="valid",
        use_bias=False, kernel_regularizer=l2(weight_penalty),
        input_shape=input_shape))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Activation('elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2),
        padding="valid", use_bias=False,
        kernel_regularizer=l2(weight_penalty)))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Activation('elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding="valid",
        use_bias=False, kernel_regularizer=l2(weight_penalty)))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Activation('elu'))
    model.add(Conv2D(64, (3, 3), padding="valid", use_bias=False,
        kernel_regularizer=l2(weight_penalty)))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Activation('elu'))
    model.add(Conv2D(64, (3, 3), padding="valid", use_bias=False,
        kernel_regularizer=l2(weight_penalty)))
    model.add(BatchNormalization(momentum=momentum))
    model.add(Activation('elu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='elu', kernel_initializer='he_normal',
        bias_regularizer=l2(weight_penalty),
        kernel_regularizer=l2(weight_penalty)))
    model.add(Dense(50, activation='elu', kernel_initializer='he_normal',
        bias_regularizer=l2(weight_penalty),
        kernel_regularizer=l2(weight_penalty)))
    model.add(Dense(10, activation='elu',
        kernel_initializer='he_normal',
        bias_regularizer=l2(weight_penalty),
        kernel_regularizer=l2(weight_penalty)))
    model.add(Dense(2, activation='softsign', kernel_initializer='he_normal')) 
    return model

if __name__ == '__main__':
    model = SimpleModel()
    model.summary()

