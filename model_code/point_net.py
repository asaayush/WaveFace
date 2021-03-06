import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

## This file will contain the primary class definitions for the Point Net

class PointNet:
    def __init__(self):
        """
        Initializing the Point Net Variables/Constants
        param :
        """
        self.t_net_layer_name_cursor = 0
        pass

    def initialize_layers(self):
        point_cloud = layers.Input(shape=(100, 3))
        K = self.input_transform_net(point_cloud)
        print("Layers Initialized")
        return K
        pass
    
    def input_transform_net(self, point_cloud, output_channels=3):
        """
        This function helps create and maintain the T-Net network which essentially operates as 
        finding the perfect affine transformation for the data such that the eventual PointNet model
        is invariant to the ordering of the data.

        For this, we use the concept of "Shared Multi Layer Perceptron". Now the interesting bit here,
        is that the implementation of a shared weight multi layer perceptron is similar to that of a 
        1x1 convolution operation. This is an important point to remember as you read & implement this 
        code.

        param:: point_cloud: Point Cloud Data (tensor) of the shape B * N * 3
        param:: output_channels: The final number of output channels desired after multiplication of 
                                 transformation matrix. This is by default 3 and cannot be lower.

        return:: transformed_point_cloud
        """
        ### Fixed Variables
        mlp_layer_dims = [64, 128, 1024]
        fc_layer_dims = [512, 256]

        ### Input Validation - Yet to implement
        batch_size = point_cloud.shape[0]
        num_points = point_cloud.shape[1]
        
        ### Layers
        tr_image = tf.expand_dims(point_cloud, axis=-1)
        tr_image_processed = self.convolution_operation(tr_image, mlp_layer_dims[0], [1, 3])
        tr_image_processed = self.convolution_operation(tr_image_processed, mlp_layer_dims[1], [1, 1])
        tr_image_processed = self.convolution_operation(tr_image_processed, mlp_layer_dims[2], [1, 1])

        layer_name = 'TNetLayer' + str(self.t_net_layer_name_cursor) + '_MaxPool'
        tr_image_processed = layers.MaxPool2D((num_points, 1), name=layer_name)(tr_image_processed)
        self.t_net_layer_name_cursor += 1

        layer_name = 'TNetLayer' + str(self.t_net_layer_name_cursor) + '_Flatten'
        tr_image_processed = layers.Flatten(name=layer_name)(tr_image_processed)
        self.t_net_layer_name_cursor += 1

        tr_image_processed = self.fully_connected_operation(tr_image_processed, fc_layer_dims[0])
        tr_image_processed = self.fully_connected_operation(tr_image_processed, fc_layer_dims[1])

        # Testing Model
        k = keras.Model(inputs=point_cloud, outputs=tr_image_processed)
        k.summary()

        return k


    ### Convolution Operation Definition
    def convolution_operation(self, input_tensor, filters, kernel_size, padding='valid'):
        # Each MLP is a shared weight process, which is best represented by this.
        # Each convolution is also followed by batch norm and relu activation.
        layer_name = 'TNetLayer' + str(self.t_net_layer_name_cursor) + '_MLP'
        x = layers.Conv2D(filters, kernel_size, name=layer_name)(input_tensor)
        layer_name = 'TNetLayer' + str(self.t_net_layer_name_cursor) + '_BN'
        x = layers.BatchNormalization(name=layer_name)(x)
        layer_name = 'TNetLayer' + str(self.t_net_layer_name_cursor) + '_ReLU'
        x = layers.ReLU(name=layer_name)(x)

        self.t_net_layer_name_cursor += 1
        return x

    def fully_connected_operation(self, input_tensor, units):
        layer_name = 'TNetLayer' + str(self.t_net_layer_name_cursor) + '_FC'
        x = layers.Dense(units, name=layer_name)(input_tensor)
        layer_name = 'TNetLayer' + str(self.t_net_layer_name_cursor) + '_BN'
        x = layers.BatchNormalization(name=layer_name)(x)
        layer_name = 'TNetLayer' + str(self.t_net_layer_name_cursor) + '_ReLU'
        x = layers.ReLU(name=layer_name)(x)
        
        self.t_net_layer_name_cursor += 1
        return x
