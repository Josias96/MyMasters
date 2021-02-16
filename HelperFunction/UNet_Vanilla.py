import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import concatenate


class UNet_Vanilla:
    def __init__(self, input_shape,showSummary=True):
        self.input_shape = input_shape
        self.showSummary = showSummary
        
    def Conv2D_block(self,input_tensor, kernel_size, filters):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        conv = Activation("relu")(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(conv)
        conv = Activation("relu")(conv)
        return conv


    def UpConvolution(self,input_tensor, skip_tensor, kernel_size, filters):
        upconv = Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer="he_normal",
                        padding="same")(UpSampling2D(size=(2, 2))(input_tensor))
        upconv = Activation("relu")(upconv)
        upconv = concatenate([upconv, skip_tensor])
        return upconv


    def CreateUnet(self):

        input_layer = Input(self.input_shape)

        c1 = self.Conv2D_block(input_layer, kernel_size=(3, 3), filters=64)
        p1 = MaxPool2D(pool_size=(2, 2))(c1)

        c2 = self.Conv2D_block(p1, kernel_size=(3, 3), filters=128)
        p2 = MaxPool2D(pool_size=(2, 2))(c2)

        c3 = self.Conv2D_block(p2, kernel_size=(3, 3), filters=256)
        p3 = MaxPool2D(pool_size=(2, 2))(c3)

        c4 = self.Conv2D_block(p3, kernel_size=(3, 3), filters=512)
        p4 = MaxPool2D(pool_size=(2, 2))(c4)

        c5 = self.Conv2D_block(p4, kernel_size=(3, 3), filters=1024)
        d5 = tensorflow.keras.layers.Dropout(0.2)(c5)

        u1 = self.UpConvolution(d5, c4, kernel_size=(3, 3), filters=512)
        c6 = self.Conv2D_block(u1, kernel_size=(3, 3), filters=512)

        u2 = self.UpConvolution(c6, c3, kernel_size=(3, 3), filters=256)
        c7 = self.Conv2D_block(u2, kernel_size=(3, 3), filters=256)

        u3 = self.UpConvolution(c7, c2, kernel_size=(3, 3), filters=128)
        c8 = self.Conv2D_block(u3, kernel_size=(3, 3), filters=128)

        u4 = self.UpConvolution(c8, c1, kernel_size=(3, 3), filters=64)
        c9 = self.Conv2D_block(u4, kernel_size=(3, 3), filters=64)

        output_layer = Conv2D(filters=1, kernel_size=(1, 1),
                            activation="sigmoid")(c9)

        MyModel = tensorflow.keras.models.Model(
            inputs=input_layer, outputs=output_layer)

        if self.showSummary:
            MyModel.summary()
        return MyModel
