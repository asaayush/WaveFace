import numpy as np
from mmwave_code.data_loader import DataLoader
import tensorflow as tf

"""
The purpose of this function is to convert point data into a usable form that will easily be fed into the model.

In essence this function does the following:
1) Choose 'x' high SNR points from the region of interest, (h1, h2, h3).
2) The 'x'' points are then arranged in a matrix of size 100x3.
3) Then a tuple is created in (input, label) form. Where the 'x' points are the input, and label is the class 
   representation either in one-hot form or singular integer value.
4) The data structure is then returned.


NOTE that this is a per-frame operation. So your main file should run this in a loop or your training steps should
reflect this accordingly.
"""


def prepare_data(frame_data, label, label_mode='one_hot', points=100):
    # Frame Data is expected to be a ____x4 matrix with x,y,z and SNR data for each point. We need to arrange the data
    # with respect to SNR first.
    arranged_frame = np.sort(frame_data, axis=-1)
    # short_frame = arranged_frame[:points, :]
    output_tuple = (frame_data, label)

    return output_tuple


def create_dataset(file_name_list, label_list, train_split=0.70, test_split=0.15, val_split=0.15, batch_size=8,
                   aggregate=1, classes=0, with_snr=True):
    if classes == 0:
        classes = len(file_name_list)
    combined_np_data = []
    labels = []
    for index, file in enumerate(file_name_list):
        np_data = np.load(file)
        if with_snr:
            np_data = np_data[0:np_data.shape[0]:aggregate, :, :]
        else:
            np_data = np_data[0:np_data.shape[0]:aggregate, :, :3]

        label = np.zeros(shape=(np_data.shape[0], classes), dtype=np.int8)
        label[:, label_list[index]] = 1
        if index == 0:
            combined_np_data = np_data
            labels = label
        else:
            combined_np_data = np.append(combined_np_data, np_data, axis=0)
            labels = np.append(labels, label, axis=0)

    combined_tf_data = tf.convert_to_tensor(combined_np_data)
    combined_tf_labels = tf.convert_to_tensor(labels)
    dataset = tf.data.Dataset.from_tensor_slices((combined_tf_data, combined_tf_labels))

    # Shuffle Dataset:
    dataset = dataset.shuffle(buffer_size=len(dataset)+100)

    # Split Dataset:
    split_check = train_split + test_split + val_split
    assert np.round(split_check) == 1

    training_samples = np.int32(train_split * len(dataset))
    training_data = dataset.take(training_samples)
    testing_data = dataset.skip(training_samples)
    validation_samples = np.int32(val_split * len(testing_data))
    validation_data = testing_data.take(validation_samples)
    testing_data = testing_data.skip(validation_samples)

    # Batch the dataset
    training_data = training_data.batch(batch_size, drop_remainder=False)
    validation_data = validation_data.batch(batch_size, drop_remainder=False)
    testing_data = testing_data.batch(batch_size, drop_remainder=False)

    # Return training and testing data
    return training_data, validation_data, testing_data
