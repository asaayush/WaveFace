import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
from mmwave_code.display_utils import *
from mmwave_code.data_loader import DataLoader
from mmwave_code.data_loader import get_points
from mmwave_code.mmwave_processor import MMWaveProcessor
from mmwave_code.prepare_data import create_dataset
import gc
from model_code.basicCNN import CnnNetwork
from tqdm import tqdm


class Evaluator:
    def __init__(self, file_names, model_params):
        self.file_names = file_names
        self.data_source_names = file_names  # + [name + "_with_mask" for name in file_names]
        self.frame_aggregate = model_params["frame_aggregate"]
        self.num_points = [500]
        self.range_fft_values = [256]
        self.model_params = model_params
        pass

    def generate_data(self, mm_wave_params):
        print("Data will be generated for the following names: \n")
        print(self.data_source_names)
        for name in self.data_source_names:
            for value in self.range_fft_values:
                dl = DataLoader(mm_wave_params)
                print("Loading & Processing Data for FFT Value: "+str(value))
                dl.range_bins = value
                adc_data = dl.get_data("data/final_captures/" + name + "/", "face_scan_" + name)
                print("Loaded ADC Data for " + name)
                processor = MMWaveProcessor(dl.range_resolution, dl.doppler_resolution, range_bins=dl.range_bins,
                                            doppler_bins=dl.doppler_bins, angular_bins=dl.angle_bins)
                print("Processing Data...")
                for i, n_points in enumerate(self.num_points):
                    points = get_points(adc_data, processor, dl, aggregate=self.frame_aggregate,
                                        num_points=n_points)
                    np.save("data/final_numpy/" +
                            str(value) + "_" + name + "_" + str(n_points) + "_data.npy", points)
                    print("\n" + str(i + 1) + " out of " + str(len(self.num_points)) + " completed\n")

                gc.collect()

            print("Processed!\n")

        """
        By the end here, you have generated numpy arrays for:
            7 Names
                Each Name generates data for:
                    3 Range FFT Values
                    3 Num of Points
        
        Therefore 7*9 = 63 Numpy Files
        Individual Models need to be trained for each subset of 7 data.
        """

        pass

    @staticmethod
    def generate_data_set(train_names, test_names, bs=64, frame_aggregate=10, with_snr=True, mask_params=None):
        # Let this function generate only the necessary dataset for that iteration of training and testing.
        label_list = [i for i in range(len(train_names))]
        tr_data, val_data, test_data = create_dataset(train_names,
                                                      label_list, batch_size=bs,
                                                      aggregate=frame_aggregate,
                                                      classes=len(train_names), with_snr=with_snr)
        if mask_params is None:
            mask_data, _, _ = create_dataset(test_names,
                                             label_list, train_split=1.0, test_split=0, val_split=0,
                                             batch_size=bs, aggregate=frame_aggregate, classes=len(train_names),
                                             with_snr=with_snr)
            return tr_data, val_data, test_data, mask_data
        else:
            mask_data_tr, mask_data_val, mask_data_test = create_dataset(test_names, label_list,
                                                                         train_split=mask_params["train"],
                                                                         test_split=mask_params["test"],
                                                                         val_split=mask_params["val"], batch_size=bs,
                                                                         aggregate=frame_aggregate,
                                                                         classes=len(train_names), with_snr=with_snr)
            return tr_data, val_data, test_data, mask_data_tr, mask_data_val, mask_data_test

    def evaluate_model(self, n_points, fft_value, return_confusion=False, with_snr=True, mask_params=None):
        # Initialize Model
        model_class = CnnNetwork(num_points=n_points, with_snr=with_snr)

        # Obtain Datasets
        train_names = ["data/final_numpy/" + str(fft_value) + "_" + name + "_" + str(n_points) + "_data.npy"
                       for name in self.file_names]
        test_names = ["data/final_numpy/" + str(fft_value) + "_" + name + "_with_mask_" + str(n_points) + "_data.npy"
                      for name in self.file_names]
        if mask_params is None:
            tr_data, val_data, test_data, mask_test = self.generate_data_set(train_names, test_names,
                                                                             self.model_params["batch_size"],
                                                                             self.frame_aggregate,
                                                                             with_snr=with_snr)
        else:
            tr_data, val_data, test_data, \
                mask_tr, mask_val, mask_test = self.generate_data_set(train_names, test_names,
                                                                      self.model_params["batch_size"],
                                                                      self.frame_aggregate,
                                                                      with_snr=with_snr,
                                                                      mask_params=mask_params)
            tr_data = tr_data.concatenate(mask_tr)
            val_data = val_data.concatenate(mask_val)

        # Train Model
        model = model_class.initialize_model(classes=len(train_names))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_params["learning_rate"]),
                      loss=self.model_params["loss_function"], metrics=self.model_params["metrics"])

        history = model.fit(x=tr_data, validation_data=val_data, epochs=self.model_params["epochs"])

        np.save("final_model_performance/" + str(fft_value) + "_" + str(n_points) + "_model_perf", history.history)
        title = "Num Points: " + str(n_points) + "        Range FFT Size: " + str(fft_value)
        plot_history(history, title)

        print("\n***Testing Model***")
        test_perf = model.evaluate(test_data)

        print("\nMASK PERFORMANCE")
        mask_test_perf = model.evaluate(mask_test)

        if return_confusion:
            _, _, _, mask_test = self.generate_data_set(train_names, test_names, 1, self.frame_aggregate,
                                                        with_snr=with_snr)
            predicted_labels = np.zeros(len(mask_test))
            true_labels = np.zeros(len(mask_test))
            i = 0
            for data_set, label_set in tqdm(mask_test):
                predicted_labels_set = model.predict(data_set)
                if predicted_labels_set.any():
                    predicted_labels[i] = np.argwhere(predicted_labels_set == np.max(predicted_labels_set))[:, 1]
                true_labels[i] = np.int8(np.argwhere(label_set == 1)[:, 1])
                i += 1

            print(predicted_labels)
            print(true_labels)
            conf_matrix = tf.math.confusion_matrix(tf.convert_to_tensor(true_labels),
                                                   tf.convert_to_tensor(predicted_labels), len(train_names))
            return test_perf, mask_test_perf, conf_matrix
        else:
            return test_perf, mask_test_perf

    def perf_monitor(self, get_c_matrix=False, with_snr=True, mask_params=None):
        for fft_value in self.range_fft_values:
            for n_point in self.num_points:
                print("\nEvaluating Model for \nRange FFT:  "+str(fft_value)+"       \tNum Points:  "+str(n_point))
                if get_c_matrix:
                    if mask_params is None:
                        test_acc, mask_test_acc, conf_matrix = self.evaluate_model(n_point, fft_value,
                                                                                   return_confusion=get_c_matrix,
                                                                                   with_snr=with_snr)
                    else:
                        print("***")
                        print(" Poisoning Training with "+str(mask_params["train"]*100)+"% of Masked Data")
                        test_acc, mask_test_acc, conf_matrix = self.evaluate_model(n_point, fft_value,
                                                                                   return_confusion=get_c_matrix,
                                                                                   with_snr=with_snr,
                                                                                   mask_params=mask_params)
                    confusion_matrix(conf_matrix, name_len=len(self.file_names))
                else:
                    if mask_params is None:
                        test_acc, mask_test_acc = self.evaluate_model(n_point, fft_value)
                    else:
                        test_acc, mask_test_acc = self.evaluate_model(n_point, fft_value, mask_params=mask_params)

                print("\n Achieved Testing Accuracy Without Mask:    " + str(test_acc[1]))
                print("\n Achieved Testing Accuracy With Mask:    " + str(mask_test_acc[1]))



