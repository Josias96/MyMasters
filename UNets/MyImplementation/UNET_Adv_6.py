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

#https://arxiv.org/pdf/1604.04112.pdf
#FAST  ANDACCURATEDEEPNETWORKLEARNING  BYEXPONENTIALLINEARUNITS(ELUS)

#This version made more use of depthwise CNNs
#Uses Elu instead of ReLu "Rethinking ReLU to Train Better CNNs"
#Fewer activation functions
#Fewer BatchNorm

class UNet_Adv:
    def __init__(self, input_shape,n_filters,showSummary=True, activation="elu"):
        self.input_shape = input_shape
        self.showSummary = showSummary
        self.n_filters = n_filters
        self.activation = activation
    
    def Conv2D_block_1(self,input_tensor, kernel_size, filters):
        conv = BatchNormalization(renorm=True)(input_tensor)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(conv)        
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

    def Conv2D_block_Down(self,input_tensor, kernel_size, filters):
        conv = DepthwiseConv2D(depth_multiplier=2, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        conv = Activation(self.activation)(conv)
        conv = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(conv)
        conv = BatchNormalization(renorm=True)(conv)

        #----------- Residual Block --------------
        conv_in = DepthwiseConv2D(depth_multiplier=2, kernel_size=(1,1),
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        conv_in = BatchNormalization(renorm=True)(conv_in)
        out = tf.keras.layers.Add()([conv, conv_in])
        #-----------------------------------------
        
        return out

    def Conv2D_block_Up(self,input_tensor, kernel_size, filters):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        conv = Activation(self.activation)(conv)
        conv = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(conv)
        conv = BatchNormalization(renorm=True)(conv)

        #----------- Residual Block --------------
        conv_in = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        conv_in = BatchNormalization(renorm=True)(conv_in)
        out = tf.keras.layers.Add()([conv, conv_in])
        #-----------------------------------------
        return out

    def Conv2D_block_Up_Last(self,input_tensor, kernel_size, filters):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        conv = Activation(self.activation)(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(conv)
        conv = BatchNormalization(renorm=True)(conv)

        #----------- Residual Block --------------
        conv_in = Conv2D(filters=filters, kernel_size=kernel_size,
                    kernel_initializer="he_normal", padding="same")(input_tensor)
        conv_in = BatchNormalization(renorm=True)(conv_in)
        out = tf.keras.layers.Add()([conv, conv_in])
        #-----------------------------------------
        return out


    def UpConvolution(self,input_tensor, skip_tensor, kernel_size, filters):
        upconv = Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer="he_normal",
                        padding="same")(UpSampling2D(size=(2, 2))(input_tensor))
        upconv = BatchNormalization(renorm=True)(upconv)
        upconv = concatenate([upconv, skip_tensor])
        return upconv


    def CreateUnet(self):

        input_layer = Input(self.input_shape)

        c1 = self.Conv2D_block_1(input_layer, kernel_size=(3, 3), filters=self.n_filters)
        p1 = MaxPool2D(pool_size=(2, 2), name="p1")(c1)

        c2 = self.Conv2D_block_Down(p1, kernel_size=(3, 3), filters=self.n_filters*2)
        p2 = MaxPool2D(pool_size=(2, 2), name="p2")(c2)

        c3 = self.Conv2D_block_Down(p2, kernel_size=(3, 3), filters=self.n_filters*4)
        p3 = MaxPool2D(pool_size=(2, 2), name="p3")(c3)

        c4 = self.Conv2D_block_Down(p3, kernel_size=(3, 3), filters=self.n_filters*8)
        p4 = MaxPool2D(pool_size=(2, 2), name="p4")(c4)

        c5 = self.Conv2D_block_Down(p4, kernel_size=(3, 3), filters=self.n_filters*16)
        p5 = MaxPool2D(pool_size=(2, 2), name="p5")(c5)

        c6 = self.Conv2D_block_Down(p5, kernel_size=(3, 3), filters=self.n_filters*32)
        p6 = MaxPool2D(pool_size=(2, 2), name="p6")(c6)
        
        c7 = self.Conv2D_block_Down(p6, kernel_size=(3, 3), filters=self.n_filters*64)
        d7 = tensorflow.keras.layers.Dropout(0.2, name="d7")(c7)

        u1 = self.UpConvolution(d7, c6, kernel_size=(2, 2), filters=self.n_filters*32)
        c8 = self.Conv2D_block_Up(u1, kernel_size=(3, 3), filters=self.n_filters*32)

        u2 = self.UpConvolution(c8, c5, kernel_size=(2, 2), filters=self.n_filters*16)
        c9 = self.Conv2D_block_Up(u2, kernel_size=(3, 3), filters=self.n_filters*16)

        u3 = self.UpConvolution(c9, c4, kernel_size=(2, 2), filters=self.n_filters*8)
        c10 = self.Conv2D_block_Up(u3, kernel_size=(3, 3), filters=self.n_filters*8)

        u4 = self.UpConvolution(c10, c3, kernel_size=(2, 2), filters=self.n_filters*4)
        c11 = self.Conv2D_block_Up(u4, kernel_size=(3, 3), filters=self.n_filters*4)

        u5 = self.UpConvolution(c11, c2, kernel_size=(2, 2), filters=self.n_filters*2)
        c12 = self.Conv2D_block_Up(u5, kernel_size=(3, 3), filters=self.n_filters*2)

        u6 = self.UpConvolution(c12, c1, kernel_size=(2, 2), filters=self.n_filters)
        c13 = self.Conv2D_block_Up_Last(u6, kernel_size=(3, 3), filters=self.n_filters)

        output_layer = Conv2D(filters=1, kernel_size=(1, 1),
                            activation="sigmoid", name="Convolution_c14")(c13)

        MyModel = tensorflow.keras.models.Model(
            inputs=input_layer, outputs=output_layer)

        if self.showSummary:
            MyModel.summary()
        return MyModel
