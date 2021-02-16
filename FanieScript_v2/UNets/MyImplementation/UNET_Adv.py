import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization



class UNet_Adv:
    def __init__(self, input_shape,showSummary=True):
        self.input_shape = input_shape
        self.showSummary = showSummary

    def Conv2D_block_Down(self,input_tensor, kernel_size, filters, depth_multiplier=1, resNet = True, batchNorm = True, firstBlock = False):
        if firstBlock:
            conv = Conv2D(filters=filters, kernel_size=kernel_size,
                        kernel_initializer="he_normal", padding="same")(input_tensor)
            conv = Activation("relu")(conv)
            if batchNorm:
                conv = BatchNormalization()(conv)
            conv = Conv2D(filters=filters, kernel_size=kernel_size,
                        kernel_initializer="he_normal", padding="same")(conv)
            if resNet:
                input_tensor = Conv2D(filters=filters, kernel_size=(1,1),
                        kernel_initializer="he_normal", padding="same")(input_tensor)
                if batchNorm:
                    input_tensor = BatchNormalization()(input_tensor)
                conv = tf.keras.layers.Add()([conv, input_tensor])
            conv = Activation("relu")(conv)
            if batchNorm:
                conv = BatchNormalization()(conv)
            return conv
        else:
            input_tensor = Conv2D(filters=filters, kernel_size=(1,1),
                        kernel_initializer="he_normal", padding="same")(input_tensor)
            conv = DepthwiseConv2D(depth_multiplier=depth_multiplier, kernel_size=kernel_size,
                        kernel_initializer="he_normal", padding="same")(input_tensor)
            conv = Activation("relu")(conv)
            if batchNorm:
                conv = BatchNormalization()(conv)
            conv = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size,
                        kernel_initializer="he_normal", padding="same")(conv)
            if resNet:
                conv = tf.keras.layers.Add()([conv, input_tensor])
            conv = Activation("relu")(conv)
            if batchNorm:
                conv = BatchNormalization()(conv)
            return conv

    def Conv2D_block_Up(self,input_tensor, kernel_size, filters, depth_multiplier=1, resNet = True, batchNorm = True, firstBlock = False):
        input_tensor = Conv2D(filters=filters, kernel_size=(1,1),
                        kernel_initializer="he_normal", padding="same")(input_tensor)
        conv = DepthwiseConv2D(depth_multiplier=depth_multiplier, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        conv = Activation("relu")(conv)
        if batchNorm:
            conv = BatchNormalization()(conv)
        conv = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(conv)
        if resNet:
            input_tensor = Conv2D(filters=filters, kernel_size=(1,1),
                    kernel_initializer="he_normal", padding="same")(input_tensor)
            conv = tf.keras.layers.Add()([conv, input_tensor])
        conv = Activation("relu")(conv)
        if batchNorm:
            conv = BatchNormalization()(conv)
        return conv


    def UpConvolution(self,input_tensor, skip_tensor, kernel_size, filters, batchNorm=True):
        upconv = Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer="he_normal",
                        padding="same")(UpSampling2D(size=(2, 2))(input_tensor))
        upconv = Activation("relu")(upconv)
        if batchNorm:
            upconv = BatchNormalization()(upconv)
        upconv = concatenate([upconv, skip_tensor])
        return upconv


    def CreateUnet(self):

        input_layer = Input(self.input_shape)

        c1 = self.Conv2D_block_Down(input_layer, kernel_size=(3, 3), filters=64, firstBlock = True)
        p1 = MaxPool2D(pool_size=(2, 2), name="p1")(c1)

        c2 = self.Conv2D_block_Down(p1, kernel_size=(3, 3), depth_multiplier=2, filters=128)
        p2 = MaxPool2D(pool_size=(2, 2), name="p2")(c2)

        c3 = self.Conv2D_block_Down(p2, kernel_size=(3, 3), depth_multiplier=2, filters=256)
        p3 = MaxPool2D(pool_size=(2, 2), name="p3")(c3)

        c4 = self.Conv2D_block_Down(p3, kernel_size=(3, 3), depth_multiplier=2, filters=512)
        p4 = MaxPool2D(pool_size=(2, 2), name="p4")(c4)

        c5 = self.Conv2D_block_Down(p4, kernel_size=(3, 3), depth_multiplier=2, filters=1024)
        d5 = tensorflow.keras.layers.Dropout(0.2, name="d5")(c5)

        u1 = self.UpConvolution(d5, c4, kernel_size=(3, 3), filters=1024)
        c6 = self.Conv2D_block_Up(u1, kernel_size=(3, 3), filters=512)

        u2 = self.UpConvolution(c6, c3, kernel_size=(3, 3), filters=512)
        c7 = self.Conv2D_block_Up(u2, kernel_size=(3, 3), filters=256)

        u3 = self.UpConvolution(c7, c2, kernel_size=(3, 3), filters=256)
        c8 = self.Conv2D_block_Up(u3, kernel_size=(3, 3),filters=128 )

        u4 = self.UpConvolution(c8, c1, kernel_size=(3, 3), filters=128)
        c9 = self.Conv2D_block_Up(u4, kernel_size=(3, 3), filters=64)

        output_layer = Conv2D(filters=1, kernel_size=(1, 1),
                            activation="sigmoid")(c9)

        MyModel = tensorflow.keras.models.Model(
            inputs=input_layer, outputs=output_layer)

        if self.showSummary:
            MyModel.summary()
        return MyModel
