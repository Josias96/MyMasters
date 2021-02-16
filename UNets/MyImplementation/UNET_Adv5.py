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
from tensorflow.keras.layers import SpatialDropout2D

def swish(x):
    #Hard Swish from "Searching for MobileNetV3"
    return (x * (tf.nn.relu6(x+3) / 6))
get_custom_objects().update({'swish': Activation(swish)})

class UNet_Adv:
    def __init__(self, input_shape,n_filters,showSummary=True, activation="relu"):
        self.input_shape = input_shape
        self.showSummary = showSummary
        self.n_filters = n_filters
        self.activation = activation
    
    def Conv2D_TailBlock(self,input_tensor, kernel_size, filters, skipcon=True):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)        
        conv = Activation(self.activation)(conv)
        conv = BatchNormalization(renorm=True)(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(conv)
        conv = Activation(self.activation)(conv)
        conv = BatchNormalization(renorm=True)(conv)
         #----------- Residual Block --------------
        if skipcon:
            conv_in = Conv2D(filters=filters, kernel_size=(1,1),
                        kernel_initializer="he_normal", padding="same")(input_tensor)
            conv_in = Activation(self.activation)(conv_in)
            conv_in = BatchNormalization(renorm=True)(conv_in)
            out = tf.keras.layers.Add()([conv, conv_in])
        else:
            out = conv
        #-----------------------------------------     
        return out

    def Conv2D_Block(self,input_tensor, kernel_size, filters, n_layers, dropout, skipcon=True):
        #----------- BottleNeck --------------
        conv_in = Conv2D(filters=filters, kernel_size=(1,1),
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        if n_layers == 4:
            conv = self.DenseNet_Block_4(conv_in, kernel_size)
        elif n_layers == 5:
            conv = self.DenseNet_Block_5(conv_in, kernel_size)
        elif n_layers == 7:
            conv = self.DenseNet_Block_7(conv_in, kernel_size)
        elif n_layers == 10:
            conv = self.DenseNet_Block_10(conv_in, kernel_size)
        elif n_layers == 12:
            conv = self.DenseNet_Block_12(conv_in, kernel_size)
        elif n_layers == 15:
            conv = self.DenseNet_Block_15(conv_in, kernel_size)
        else:
            print("Error with DenseNet_Blocks. Confirm n_layers is one of:\n 4, 5, 7, 10, 12, 15")
        #----------- Residual Block --------------
        if skipcon:
            out = tf.keras.layers.Add()([conv_in, conv])
        else:
            out = conv
        #-----------------------------------------
        out = Activation(self.activation)(out)
        out = BatchNormalization(renorm=True)(out)
        out = SpatialDropout2D(dropout)(out)
        return out

    def DenseNet_Chunk(self,input_tensor, kernel_size,BatchNorm=True, activation=True):
        DC_In = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        if activation:
            DC_In = Activation(self.activation)(DC_In)
        if BatchNorm:
            DC_In = BatchNormalization(renorm=True)(DC_In)
        DC_Out = tf.keras.layers.Add()([DC_In, input_tensor])
        return DC_Out, DC_In

    def DenseNet_Block_4(self,input_tensor, kernel_size):
        DN_1, DN_A = self.DenseNet_Chunk(input_tensor, kernel_size)
        DN_2, DN_B = self.DenseNet_Chunk(DN_1, kernel_size)
        DN_3, DN_C = self.DenseNet_Chunk(DN_2, kernel_size)
        DN_Out = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(DN_3)
        DN_Out = tf.keras.layers.Add()([DN_Out, DN_A, DN_B, DN_C])
        DN_Out = Activation(self.activation)(DN_Out)
        DN_Out = BatchNormalization(renorm=True)(DN_Out)
        return DN_Out

    def DenseNet_Block_5(self,input_tensor, kernel_size):
        DN_1, DN_A = self.DenseNet_Chunk(input_tensor, kernel_size)
        DN_2, DN_B = self.DenseNet_Chunk(DN_1, kernel_size)
        DN_3, DN_C = self.DenseNet_Chunk(DN_2, kernel_size)
        DN_4, DN_D = self.DenseNet_Chunk(DN_3, kernel_size)
        DN_Out = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(DN_4)
        DN_Out = tf.keras.layers.Add()([DN_Out, DN_A, DN_B, DN_C, DN_D])
        DN_Out = Activation(self.activation)(DN_Out)
        DN_Out = BatchNormalization(renorm=True)(DN_Out)
        return DN_Out

    def DenseNet_Block_7(self,input_tensor, kernel_size):
        DN_1, DN_A = self.DenseNet_Chunk(input_tensor, kernel_size)
        DN_2, DN_B = self.DenseNet_Chunk(DN_1, kernel_size)
        DN_3, DN_C = self.DenseNet_Chunk(DN_2, kernel_size)
        DN_4, DN_D = self.DenseNet_Chunk(DN_3, kernel_size)
        DN_5, DN_E = self.DenseNet_Chunk(DN_4, kernel_size)
        DN_6, DN_F = self.DenseNet_Chunk(DN_5, kernel_size)
        DN_Out = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(DN_6)
        DN_Out = tf.keras.layers.Add()([DN_Out, DN_A, DN_B, DN_C, DN_D, DN_E, DN_F])
        DN_Out = Activation(self.activation)(DN_Out)
        DN_Out = BatchNormalization(renorm=True)(DN_Out)
        return DN_Out

    def DenseNet_Block_10(self,input_tensor, kernel_size):
        DN_1, DN_A = self.DenseNet_Chunk(input_tensor, kernel_size)
        DN_2, DN_B = self.DenseNet_Chunk(DN_1, kernel_size)
        DN_3, DN_C = self.DenseNet_Chunk(DN_2, kernel_size)
        DN_4, DN_D = self.DenseNet_Chunk(DN_3, kernel_size)
        DN_5, DN_E = self.DenseNet_Chunk(DN_4, kernel_size)
        DN_6, DN_F = self.DenseNet_Chunk(DN_5, kernel_size)
        DN_7, DN_G = self.DenseNet_Chunk(DN_6, kernel_size)
        DN_8, DN_H = self.DenseNet_Chunk(DN_7, kernel_size)
        DN_9, DN_I = self.DenseNet_Chunk(DN_8, kernel_size)
        DN_Out = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(DN_9)
        DN_Out = tf.keras.layers.Add()([DN_Out, DN_A, DN_B, DN_C, DN_D, DN_E, DN_F, DN_G, DN_H, DN_I])
        DN_Out = Activation(self.activation)(DN_Out)
        DN_Out = BatchNormalization(renorm=True)(DN_Out)
        return DN_Out

    def UpConvolution(self,input_tensor, skip_tensor, kernel_size, filters):
        upconv = Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer="he_normal",
                        padding="same")(UpSampling2D(size=kernel_size)(input_tensor))
        upconv = BatchNormalization(renorm=True)(upconv)
        upconv = concatenate([upconv, skip_tensor])
        return upconv

    def CreateUnet(self):

        input_layer = Input(self.input_shape)

        c1 = self.Conv2D_TailBlock(input_layer, kernel_size=(3, 3), filters=self.n_filters)
        p1 = MaxPool2D(pool_size=(2, 2), name="p1")(c1)

        c2 = self.Conv2D_Block(p1, kernel_size=(3, 3), filters=self.n_filters*2, n_layers=4, dropout=0.2)
        p2 = MaxPool2D(pool_size=(2, 2), name="p2")(c2)

        c3 = self.Conv2D_Block(p2, kernel_size=(3, 3), filters=self.n_filters*4, n_layers=5, dropout=0.2)
        p3 = MaxPool2D(pool_size=(2, 2), name="p3")(c3)

        c4 = self.Conv2D_Block(p3, kernel_size=(3, 3), filters=self.n_filters*8, n_layers=7, dropout=0.2)
        p4 = MaxPool2D(pool_size=(2, 2), name="p4")(c4)

        c5 = self.Conv2D_Block(p4, kernel_size=(3, 3), filters=self.n_filters*16, n_layers=10, dropout=0.2)

        u1 = self.UpConvolution(c5, c4, kernel_size=(2, 2), filters=self.n_filters*8)
        c6 = self.Conv2D_Block(u1, kernel_size=(3, 3), filters=self.n_filters*8, n_layers=7, dropout=0.2)

        u2 = self.UpConvolution(c6, c3, kernel_size=(2, 2), filters=self.n_filters*4)
        c7 = self.Conv2D_Block(u2, kernel_size=(3, 3), filters=self.n_filters*4, n_layers=5, dropout=0.2)

        u3 = self.UpConvolution(c7, c2, kernel_size=(2, 2), filters=self.n_filters*2)
        c8 = self.Conv2D_Block(u3, kernel_size=(3, 3),filters=self.n_filters*2, n_layers=4, dropout=0.2)

        u4 = self.UpConvolution(c8, c1, kernel_size=(2, 2), filters=self.n_filters)
        c9 = self.Conv2D_TailBlock(u4, kernel_size=(3, 3), filters=self.n_filters)

        output_layer = Conv2D(filters=1, kernel_size=(1, 1),
                            activation="sigmoid", name="OutLayer")(c9)

        MyModel = tensorflow.keras.models.Model(
            inputs=input_layer, outputs=output_layer)

        if self.showSummary:
            MyModel.summary()
        return MyModel
