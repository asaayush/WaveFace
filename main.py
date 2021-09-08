from evaluate_models import *
"""import model_code.point_net as point_net
from mmwave_code.data_loader import DataLoader
from mmwave_code.data_loader import get_points
from mmwave_code.mmwave_processor import MMWaveProcessor
from mmwave_code.display_utils import plot_history
from mmwave_code.display_utils import confusion_matrix
from tqdm import tqdm
import numpy as np
from mmwave_code.prepare_data import create_dataset
import tensorflow as tf
from model_code.basicCNN import CnnNetwork"""


# Import and Process ADC Data
mm_wave_params = {'adc_samples': 256,
                  'adc_sample_rate': 4000,
                  'freq_slope': 45.480,
                  'real_only': False,
                  'frames': 2500,
                  'chirp_loops': 180,
                  'number_rx_antennas': 4,
                  'number_tx_antennas': 3,
                  'angle_bins': 64,
                  'ramp_end_time': 87.73,
                  'idle_time': 10,
                  'start_freq': 77,
                  'tx_ant': 3,
                  'rx_ant': 4,
                  'lvds_lanes': 4,
                  'adc_start_time': 6.4}

"""names = ["aayush",
         "natalija",
         "vaibhav",
         "shagun",
         "keerthana",
         "balvansh",
         "sanjana"]
test_names = ["aayush_with_mask",
              "natalija_with_mask",
              "vaibhav_with_mask",
              "shagun_with_mask",
              "keerthana_with_mask",
              "balvansh_with_mask",
              "sanjana_with_mask"]
frame_aggregate = 10
n_points = 300
l_r = 1e-5
epoch = 200
bs = 64

for name in names:
    dl = DataLoader(mm_wave_params)
    adc_data = dl.get_data("data/final_captures/"+name+"/", "face_scan_"+name)
    print("Loaded ADC Data for "+name)
    processor = MMWaveProcessor(dl.range_resolution, dl.doppler_resolution, range_bins=dl.range_bins,
                                doppler_bins=dl.doppler_bins, angular_bins=dl.angle_bins)
    print("Processing Data...")
    points = get_points(adc_data, processor, dl, aggregate=frame_aggregate, num_points=n_points)
    print("Processed!\n")
    np.save("data/final_numpy/"+name+"_data.npy", points)

files = ["data/final_numpy/" + name + "_data.npy" for name in names]
mask_files = ["data/final_numpy/" + name + "_data.npy" for name in test_names]
# label_list = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
label_list = [0, 1, 2, 3, 4, 5, 6]

tr_data, val_data, test_data = create_dataset(files, label_list, batch_size=bs, aggregate=frame_aggregate, classes=7)
mask_test_data, _, _ = create_dataset(mask_files, label_list, train_split=0.9, test_split=0.05, val_split=0.05,
                                      batch_size=bs, aggregate=frame_aggregate, classes=7)

k = CnnNetwork(num_points=n_points)

model = k.initialize_model(classes=7)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_r),
              loss='categorical_crossentropy', metrics=['categorical_accuracy', 'AUC'])

history = model.fit(x=tr_data, validation_data=val_data, epochs=epoch)
plot_history(history)
print(np.mean(history.history["val_categorical_accuracy"]))
np.save("final_model_performance/no_mask_7_class_training_history", history.history)


model = tf.keras.models.load_model("models/final_model_all_7_with_mask")
history = np.load("final_model_performance/all_data_training_history.npy")

plot_history(history)


# Loading the actual labels
test_data, _, _ = create_dataset(files, label_list, batch_size=1, aggregate=frame_aggregate, classes=7,
                                 train_split=0.9, test_split=0.05, val_split=0.05)
predicted_labels = np.zeros(len(mask_test_data))
true_labels = np.zeros(len(mask_test_data))
i = 0
for data_set, label_set in mask_test_data:
    predicted_labels_set = np.int8(tf.math.round(model.predict(data_set)))
    predicted_labels[i] = np.argwhere(predicted_labels_set == 1)[:, 1]
    true_labels[i] = np.int8(np.argwhere(label_set == 1)[:, 1])
    i += 1

confusion_matrix(tf.math.confusion_matrix(tf.convert_to_tensor(true_labels),
                                          tf.convert_to_tensor(predicted_labels), 7))

print("\n***Testing Model***")
test_perf = model.evaluate(test_data)

print("\nMASK PERFORMANCE")
mask_test_perf = model.evaluate(mask_test_data)
"""


model_params = {
    "epochs": 200,
    "metrics": ['categorical_accuracy', 'AUC'],
    "batch_size": 64,
    "learning_rate": 1e-5,
    "loss_function": 'categorical_crossentropy',
    "frame_aggregate": 10
}


# Enter your desired people here. Like 1, 4, 7, 2 etc.
file_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

mask_params = {"train": 0.0,
               "val": 0.0,
               "test": 1.0}

evaluator = Evaluator(file_names, model_params)
evaluator.num_points = [500]
evaluator.range_fft_values = [256]
# evaluator.generate_data(mm_wave_params)

evaluator.perf_monitor(with_snr=True, get_c_matrix=True, mask_params=mask_params)

