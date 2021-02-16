import tensorflow as tf
import tensorflow.keras
import numpy as np

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects

def swish(x, beta = 1):
        return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': Activation(swish)})

class UNet_Adv:
    def __init__(self, input_shape,n_filters,showSummary=True, activation="relu"):
        self.input_shape = input_shape
        self.showSummary = showSummary
        self.n_filters = n_filters
        self.activation = activation
    
    def Conv2D_TailBlock(self,input_tensor, kernel_size, filters):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)        
        conv = Activation(self.activation)(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(conv)
        conv = BatchNormalization(renorm=True)(conv)

         #----------- Residual Block --------------
        conv_in = Conv2D(filters=filters, kernel_size=(1,1),
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        conv_in = BatchNormalization(renorm=True)(conv_in)
        out = tf.keras.layers.Add()([conv, conv_in])
        #-----------------------------------------     
        return out

    def Conv2D_Block(self,input_tensor, kernel_size, filters, bottleNeckF=1):
        #----------- BottleNeck --------------
        conv_in = Conv2D(filters=np.int(filters/bottleNeckF), kernel_size=(1,1),
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        conv = self.DenseNet_Block(conv_in, kernel_size)
        #----------- Residual Block --------------
        out = tf.keras.layers.Add()([conv_in, conv])
        #-----------------------------------------
        #out = Conv2D(filters=filters, kernel_size=(1,1),
        #            kernel_initializer="he_normal", padding="same")(out)
        out = BatchNormalization(renorm=True)(out)
        return out

    def DenseNet_Chunk(self,input_tensor, kernel_size, BatchNorm=False, activation=False):
        #Assume DenseNet_Block does not need to increase or decrease filters!
        DC_In = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        if activation:
            DC_In = Activation(self.activation)(DC_In)
        if BatchNorm:
            DC_In = BatchNormalization(renorm=True)(DC_In)
        DC_Out = tf.keras.layers.Add()([DC_In, input_tensor])
        return DC_Out, DC_In

    def DenseNet_Block(self,input_tensor, kernel_size):
        DN_1, DN_A = self.DenseNet_Chunk(input_tensor, kernel_size, activation=True, BatchNorm=True)
        DN_2, DN_B = self.DenseNet_Chunk(DN_1, kernel_size,  BatchNorm=True)
        DN_3, DN_C = self.DenseNet_Chunk(DN_2, kernel_size, activation=True, BatchNorm=True)
        DN_Out = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(DN_3)
        DN_Out = tf.keras.layers.Add()([DN_Out, DN_A, DN_B, DN_C])
        DN_Out = BatchNormalization(renorm=True)(DN_Out)
        return DN_Out

    def UpConvolution(self,input_tensor, skip_tensor, kernel_size, filters):
        upconv = Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer="he_normal",
                        padding="same")(UpSampling2D(size=(2, 2))(input_tensor))
        upconv = BatchNormalization(renorm=True)(upconv)
        upconv = concatenate([upconv, skip_tensor])
        return upconv


    def CreateUnet(self):

        input_layer = Input(self.input_shape)

        c1 = self.Conv2D_TailBlock(input_layer, kernel_size=(3, 3), filters=self.n_filters)
        p1 = MaxPool2D(pool_size=(2, 2), name="p1")(c1)

        c2 = self.Conv2D_Block(p1, kernel_size=(3, 3), filters=self.n_filters*2)
        p2 = MaxPool2D(pool_size=(2, 2), name="p2")(c2)

        c3 = self.Conv2D_Block(p2, kernel_size=(3, 3), filters=self.n_filters*4)
        p3 = MaxPool2D(pool_size=(2, 2), name="p3")(c3)

        c4 = self.Conv2D_Block(p3, kernel_size=(3, 3), filters=self.n_filters*8)
        p4 = MaxPool2D(pool_size=(2, 2), name="p4")(c4)

        c5 = self.Conv2D_Block(p4, kernel_size=(3, 3), filters=self.n_filters*16)
        d5 = tensorflow.keras.layers.Dropout(0.2, name="d5")(c5)

        u1 = self.UpConvolution(d5, c4, kernel_size=(2, 2), filters=self.n_filters*8)
        c6 = self.Conv2D_Block(u1, kernel_size=(3, 3), filters=self.n_filters*8)

        u2 = self.UpConvolution(c6, c3, kernel_size=(2, 2), filters=self.n_filters*4)
        c7 = self.Conv2D_Block(u2, kernel_size=(3, 3), filters=self.n_filters*4)

        u3 = self.UpConvolution(c7, c2, kernel_size=(2, 2), filters=self.n_filters*2)
        c8 = self.Conv2D_Block(u3, kernel_size=(3, 3),filters=self.n_filters*2 )

        u4 = self.UpConvolution(c8, c1, kernel_size=(2, 2), filters=self.n_filters)
        c9 = self.Conv2D_TailBlock(u4, kernel_size=(3, 3), filters=self.n_filters)

        output_layer = Conv2D(filters=1, kernel_size=(1, 1),
                            activation="sigmoid", name="Convolution_c10")(c9)

        MyModel = tensorflow.keras.models.Model(
            inputs=input_layer, outputs=output_layer)

        if self.showSummary:
            MyModel.summary()
        return MyModel
