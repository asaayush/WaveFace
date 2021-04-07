import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

## This file will contain the primary class definitions for the Point Net

class PointNet:
    def __init__(self, classes, input_dims):
        """
        Initializing the Point Net Variables/Constants
        param :
        """
        self.t_net_layer_name_cursor = 0
        self.classes = classes
        self.layer_length_list = [3, 64, 64, 64, 64, 128, 1024, 512, 256, classes]
        self.input_dims = input_dims
        pass

    def initialize_model(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        point_cloud = layers.Input(shape=(100, self.input_dims))
        trans_point_cloud = self.transform_net(point_cloud, output_channels=self.layer_length_list[0],
                                               network_name='TNetLayer_')
        trans_point_cloud = tf.expand_dims(trans_point_cloud, axis=1)
        features = self.convolution_operation(trans_point_cloud, self.layer_length_list[1], [1, 1], 
                                              net_name='FeatureLayer1_')

        features = self.convolution_operation(features, self.layer_length_list[2], [1, 1], 
                                              net_name='FeatureLayer2_')
        
        features = tf.squeeze(features, axis=1)
        
        trans_features = self.transform_net(features, output_channels=self.layer_length_list[3],
                                            network_name='FNetLayer_')
        trans_features = tf.expand_dims(trans_features, axis=1)
        
        trans_features = self.convolution_operation(trans_features, self.layer_length_list[4], [1, 1])
        trans_features = self.convolution_operation(trans_features, self.layer_length_list[5], [1, 1])
        trans_features = self.convolution_operation(trans_features, self.layer_length_list[6], [1, 1])

        print(trans_features.shape)
        trans_features = tf.squeeze(trans_features, axis=1)
        
        max_pool_output = layers.MaxPool1D((trans_features.shape[1]))(trans_features)
        max_pool_output = tf.squeeze(max_pool_output, axis=1)

        fc_output = self.fully_connected_operation(max_pool_output, self.layer_length_list[7], 'Final_FC_Layer_1_')
        fc_output = layers.Dropout(0.7)(fc_output)
        fc_output = self.fully_connected_operation(fc_output, self.layer_length_list[8], 'Final_FC_Layer_2_')
        fc_output = layers.Dropout(0.7)(fc_output)
        fc_output = layers.Dense(self.layer_length_list[9])(fc_output)

        K = keras.Model(inputs=point_cloud, outputs=fc_output)
        K.summary()
        return K
    
    def transform_net(self, point_cloud, output_channels=3, network_name=''):
        """
        This function helps create and maintain the T-Net network which essentially operates as 
        finding the perfect affine transformation for the data such that the eventual PointNet model
        is invariant to the ordering of the data.

        For this, we use the concept of "Shared Multi Layer Perceptron". Now the interesting bit here,
        is that the implementation of a shared weight multi layer perceptron is similar to that of a 
        1x1 convolution operation. This is an important point to remember as you read & implement this 
        code.

        The transform net architecture is very straightforward. 3 shared MLP layers of dimensions 64, 128
        and 1024. Then two fully connected layers of 512 and 256. Each layer is followed by BatchNorm and 
        ReLU. The output layer is then d

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
        tr_image_processed = self.convolution_operation(tr_image, mlp_layer_dims[0], [1, output_channels], 
                                                        net_name=network_name)
        tr_image_processed = self.convolution_operation(tr_image_processed, mlp_layer_dims[1], [1, 1],
                                                        net_name=network_name)
        tr_image_processed = self.convolution_operation(tr_image_processed, mlp_layer_dims[2], [1, 1],
                                                        net_name=network_name)

        layer_name = network_name + str(self.t_net_layer_name_cursor) + '_MaxPool'
        tr_image_processed = layers.MaxPool2D((num_points, 1), name=layer_name)(tr_image_processed)
        self.t_net_layer_name_cursor += 1

        layer_name = network_name + str(self.t_net_layer_name_cursor) + '_Flatten'
        tr_image_processed = layers.Flatten(name=layer_name)(tr_image_processed)
        self.t_net_layer_name_cursor += 1

        tr_image_processed = self.fully_connected_operation(tr_image_processed, fc_layer_dims[0],
                                                            net_name=network_name)
        tr_image_processed = self.fully_connected_operation(tr_image_processed, fc_layer_dims[1],
                                                            net_name=network_name)
        
        # Steps for Output Layer
        # Note that instead of individually adding bias, we just initialize them as an 
        # identity matrix.
        tr_image_processed = layers.Dense(3 * output_channels, 
                                          kernel_initializer='identity',
                                          name=network_name+'_Output')(tr_image_processed)
        tr_image_processed = tf.reshape(tr_image_processed, (-1, output_channels, output_channels))
        transformed_point_cloud = tf.matmul(point_cloud, tr_image_processed)

        self.t_net_layer_name_cursor = 0
        # Testing Model
        return transformed_point_cloud


    ### Convolution Operation Definition
    def convolution_operation(self, input_tensor, filters, kernel_size, padding='valid', net_name=''):
        # Each MLP is a shared weight process, which is best represented by this.
        # Each convolution is also followed by batch norm and relu activation.
        layer_name = net_name + str(self.t_net_layer_name_cursor) + '_MLP'
        x = layers.Conv2D(filters, kernel_size, name=layer_name)(input_tensor)
        layer_name = net_name + str(self.t_net_layer_name_cursor) + '_BN'
        x = layers.BatchNormalization(name=layer_name)(x)
        layer_name = net_name + str(self.t_net_layer_name_cursor) + '_ReLU'
        x = layers.ReLU(name=layer_name)(x)

        self.t_net_layer_name_cursor += 1
        return x

    def fully_connected_operation(self, input_tensor, units, net_name=''):
        layer_name = net_name + str(self.t_net_layer_name_cursor) + '_FC'
        x = layers.Dense(units, name=layer_name)(input_tensor)
        layer_name = net_name + str(self.t_net_layer_name_cursor) + '_BN'
        x = layers.BatchNormalization(name=layer_name)(x)
        layer_name = net_name + str(self.t_net_layer_name_cursor) + '_ReLU'
        x = layers.ReLU(name=layer_name)(x)
        
        self.t_net_layer_name_cursor += 1
        return x



k = PointNet(4, 4)
model = k.initialize_model()