import tensorflow as tf
from tensorflow.keras import layers


class CnnNetwork:
    def __init__(self, num_points, with_snr=True):
        self.num_points = num_points
        if with_snr:
            self.input_shape = (num_points, 4)
        else:
            self.input_shape = (num_points, 3)

    def initialize_model(self, classes):
        x_in = layers.Input(shape=self.input_shape)
        x = self.locally_connected_block(x_in, k_sizes=[3, 7, 15, 21, 33], filters=256)
        # x = self.locally_connected_block(x, k_sizes=[3, 7, 15, 21, 33], filters=512)
        x = self.locally_connected_block(x, k_sizes=[3, 7, 15, 21, 33], filters=32)
        x = layers.MaxPool1D(3)(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=x_in, outputs=x)
        # model.summary()
        return model

    @staticmethod
    def locally_connected_block(input_value, k_sizes=None, filters=256):
        if k_sizes is None:
            k_sizes = [1, 3, 5, 7, 9]
        else:
            assert len(k_sizes) == 5
        # 5 Unique sized filters applied simultaneously and the output is concatenated.
        output_1 = layers.Conv1D(filters=filters, kernel_size=k_sizes[0],
                                 padding='same', activation='relu')(input_value)
        output_1 = layers.BatchNormalization()(output_1)

        output_2 = layers.Conv1D(filters=filters, kernel_size=k_sizes[1],
                                 padding='same', activation='relu')(input_value)
        output_2 = layers.BatchNormalization()(output_2)

        output_3 = layers.Conv1D(filters=filters, kernel_size=k_sizes[2],
                                 padding='same', activation='relu')(input_value)
        output_3 = layers.BatchNormalization()(output_3)

        output_4 = layers.Conv1D(filters=filters, kernel_size=k_sizes[3],
                                 padding='same', activation='relu')(input_value)
        output_4 = layers.BatchNormalization()(output_4)

        output_5 = layers.Conv1D(filters=filters, kernel_size=k_sizes[4],
                                 padding='same', activation='relu')(input_value)
        output_5 = layers.BatchNormalization()(output_5)

        output = layers.concatenate([output_1, output_2, output_3, output_4, output_5])

        return output


