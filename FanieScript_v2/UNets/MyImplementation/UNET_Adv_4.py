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

#This version made more use of depthwise CNNs

class UNet_Adv:
    def __init__(self, input_shape,n_filters,showSummary=True):
        self.input_shape = input_shape
        self.showSummary = showSummary
        self.n_filters = n_filters
    
    def Conv2D_block_1(self,input_tensor, kernel_size, filters, activation = "relu"):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        conv = Activation(activation)(conv)           
        conv = BatchNormalization(renorm=True)(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(conv)
        conv = Activation(activation)(conv)
        conv = BatchNormalization(renorm=True)(conv)
        return conv

    def Conv2D_block_Down(self,input_tensor, kernel_size, filters, depth_multiplier, activation = "relu"):
        conv = DepthwiseConv2D(depth_multiplier=depth_multiplier, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        conv = Activation(activation)(conv)
        conv = BatchNormalization(renorm=True)(conv)
        conv = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(conv)
        conv = BatchNormalization(renorm=True)(conv)
        #----------- Residual Block --------------
        input_tensor_Res = Conv2D(filters=filters, kernel_size=(1,1),
                        kernel_initializer="he_normal", padding="same")(input_tensor)
        input_tensor_Res = BatchNormalization(renorm=True)(input_tensor_Res)
        #-----------------------------------------
        conv = tf.keras.layers.Add()([conv, input_tensor_Res])
        conv = Activation(activation)(conv)
        conv = BatchNormalization(renorm=True)(conv)
        return conv

    def Conv2D_block_Up(self,input_tensor, kernel_size, filters, activation = "relu"):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        conv = Activation(activation)(conv)
        conv = BatchNormalization(renorm=True)(conv)
        conv = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(conv)
        conv = BatchNormalization(renorm=True)(conv)
        #----------- Residual Block --------------
        input_tensor_Res = Conv2D(filters=filters, kernel_size=(1,1),
                        kernel_initializer="he_normal", padding="same")(input_tensor)
        input_tensor_Res = BatchNormalization(renorm=True)(input_tensor_Res)
        #-----------------------------------------
        conv = tf.keras.layers.Add()([conv, input_tensor_Res])
        conv = Activation(activation)(conv)
        conv = BatchNormalization(renorm=True)(conv)
        return conv


    def UpConvolution(self,input_tensor, skip_tensor, kernel_size, filters, activation = "relu"):
        upconv = Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer="he_normal",
                        padding="same")(UpSampling2D(size=(2, 2))(input_tensor))
        upconv = Activation(activation)(upconv)
        upconv = BatchNormalization(renorm=True)(upconv)
        upconv = concatenate([upconv, skip_tensor])
        return upconv


    def CreateUnet(self):

        input_layer = Input(self.input_shape)

        c1 = self.Conv2D_block_1(input_layer, kernel_size=(3, 3), filters=self.n_filters)
        p1 = MaxPool2D(pool_size=(2, 2), name="p1")(c1)

        c2 = self.Conv2D_block_Down(p1, kernel_size=(3, 3), filters=self.n_filters*2, depth_multiplier=2)
        p2 = MaxPool2D(pool_size=(2, 2), name="p2")(c2)

        c3 = self.Conv2D_block_Down(p2, kernel_size=(3, 3), filters=self.n_filters*4, depth_multiplier=2)
        p3 = MaxPool2D(pool_size=(2, 2), name="p3")(c3)

        c4 = self.Conv2D_block_Down(p3, kernel_size=(3, 3), filters=self.n_filters*8, depth_multiplier=2)
        p4 = MaxPool2D(pool_size=(2, 2), name="p4")(c4)

        c5 = self.Conv2D_block_Down(p4, kernel_size=(3, 3), filters=self.n_filters*16, depth_multiplier=2)
        p5 = MaxPool2D(pool_size=(2, 2), name="p5")(c5)

        c6 = self.Conv2D_block_Down(p5, kernel_size=(3, 3), filters=self.n_filters*32, depth_multiplier=2)
        d6 = tensorflow.keras.layers.Dropout(0.2, name="d6")(c6)

        u1 = self.UpConvolution(d6, c5, kernel_size=(3, 3), filters=self.n_filters*16)
        c7 = self.Conv2D_block_Up(u1, kernel_size=(3, 3), filters=self.n_filters*16)

        u2 = self.UpConvolution(c7, c4, kernel_size=(3, 3), filters=self.n_filters*8)
        c8 = self.Conv2D_block_Up(u2, kernel_size=(3, 3), filters=self.n_filters*8)

        u3 = self.UpConvolution(c8, c3, kernel_size=(3, 3), filters=self.n_filters*4)
        c9 = self.Conv2D_block_Up(u3, kernel_size=(3, 3), filters=self.n_filters*4)

        u4 = self.UpConvolution(c9, c2, kernel_size=(3, 3), filters=self.n_filters*2)
        c10 = self.Conv2D_block_Up(u4, kernel_size=(3, 3), filters=self.n_filters*2)

        u5 = self.UpConvolution(c10, c1, kernel_size=(3, 3), filters=self.n_filters)
        c11 = self.Conv2D_block_Up(u5, kernel_size=(3, 3), filters=self.n_filters)

        output_layer = Conv2D(filters=1, kernel_size=(1, 1),
                            activation="sigmoid", name="Convolution_c9")(c11)

        MyModel = tensorflow.keras.models.Model(
            inputs=input_layer, outputs=output_layer)

        if self.showSummary:
            MyModel.summary()
        return MyModel
