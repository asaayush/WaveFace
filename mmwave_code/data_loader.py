import numpy as np
from tqdm import tqdm
import os
from mmwave_code.mmwave_processor import MMWaveProcessor
import matplotlib.pyplot as plt
from time import time
from mmwave_code.display_utils import display_3d_image


class DataLoader:
    def __init__(self, params, debug=False):
        """This function sets all mmwave chirp parameters as a part of the dataloader
        class. This ensures that the parameters are accessible throughout the class
        without the explicit need to provide it as input in each function.

        Args:
            params (dictionary): A dictionary of every fundamental parameter.
                The dictionary should contain the following information:
                1)  Frames: Number of frames captured.
                2)  ADC Samples: Number of ADC Samples per chirp captured.
                3)  TX_Ant: Number of Transmission Antennas used. (Should be greater than 2)
                4)  RX_Ant: Number of Receiving Antennas used.
                5)  Chirp Loops: Number of Chirp Loops per Frame
                6)  ADC Sample Rate: The rate at which the ADC samples were collected (ksps)
                7)  Freq Slope: The slope of the chirp ramp frequency. (MHz/us)
                8)  Start Freq: The starting frequency of the chirp ramp. (GHz)
                9)  Ramp End Time: Time for ramp end. (us)
                10) Idle Time: The idle time before which the next chirp begins. (us)
                11) LVDS Lanes: The number of LVDS lanes used in the data capture.
                12) ADC Start Time: Time before the ADC starts collecting samples.

            debug (Boolean): Flag to output debug information about bandwidth and resolutions.

        """
        # Setting up class variables.
        self.frames = params['frames']
        self.adc_samples = params['adc_samples']
        self.tx_antennas = params['tx_ant']
        self.rx_antennas = params['rx_ant']
        self.chirp_loops = params['chirp_loops']
        self.adc_sample_rate = params['adc_sample_rate']
        self.freq_slope = params['freq_slope']
        self.start_freq = params['start_freq']
        self.ramp_end_time = params['ramp_end_time']
        self.idle_time = params['idle_time']
        self.lvds_lanes = params['lvds_lanes']
        self.adc_start_time = params['adc_start_time']


        # Some parameters need to be calculated from these values.
        self.chirps_per_frame = self.chirp_loops * self.tx_antennas
        self.range_bins = self.adc_samples
        self.doppler_bins = self.chirp_loops
        # Decided using the number of Tx antenna. If more than 32 antenna, go for higher angle bins.
        self.angle_bins = 64 

        # Computing resolutions
        self.range_resolution, self.bandwidth = self.calculate_range_resolution(self.freq_slope,
                                                                                self.ramp_end_time)
        
        self.doppler_resolution = self.calculate_doppler_resolution(self.bandwidth, self.start_freq, self.ramp_end_time,
                                                                    self.idle_time, self.chirp_loops, self.tx_antennas)
        if debug:
            print("\n Range Resolution: " + str(self.range_resolution * 1e2) + " cm")
            print("\n Doppler Resolution: " + str(self.doppler_resolution) + " m/s")
            print("\n Bandwidth: " + str(self.bandwidth) + " MHz")

    def get_data(self, directory, bin_file_name, debug_flag = False):
        def check_multiple_files(dir, f_name, d_flag):
            multi_file_name = dir + f_name
            file_len = 1
            flag = True
            if os.path.exists(multi_file_name + "_0.bin") & d_flag:
                print("\n Multiple Files Exists")
            while flag:
                if os.path.exists(multi_file_name + "_" + str(file_len) + ".bin"):
                    file_len += 1
                else:
                    flag = False
            return os.path.exists(multi_file_name + "_0.bin"), file_len

        # Check if multiple files exists
        multiple_file_flag, num_files = check_multiple_files(directory, bin_file_name, debug_flag)
        # Initializing ADC Data
        adc_data = np.zeros(shape=(self.frames * self.chirps_per_frame * self.adc_samples, self.lvds_lanes),
                            dtype=complex)
        lengths = []

        # Load the files based on whether a single or multiple files exist
        for i in range(num_files):
            # Load file name
            if num_files == 1:
                file_name = directory + bin_file_name + ".bin"
            else:
                file_name = directory + bin_file_name + "_" + str(i) + ".bin"

            file = open(file_name, mode='rb')

            # Append if 2nd file or higher
            init_data = np.fromfile(file, dtype=np.int16)
            """if i == 0:
                
            else:
                init_data = np.append(init_data, np.fromfile(file, dtype=np.int16))"""

            # Reshape Data based on LVDS Lanes
            init_data = np.reshape(init_data, newshape=(-1, self.lvds_lanes * 2))

            # Converting I-Q data to Complex Data
            if debug_flag:
                print(" Converting IQ Data to Complex Form")
                t1 = time()

            temp_adc_data = np.zeros(shape=(init_data.shape[0], 4), dtype=complex)
            temp_adc_data[:, 0] = init_data[:, 0] + (1j * init_data[:, 4])
            temp_adc_data[:, 1] = init_data[:, 1] + (1j * init_data[:, 5])
            temp_adc_data[:, 2] = init_data[:, 2] + (1j * init_data[:, 6])
            temp_adc_data[:, 3] = init_data[:, 3] + (1j * init_data[:, 7])

            if debug_flag:
                t2 = time() - t1
                print(" Time Taken is: " + str(t2) + " s")
            if i == 0:
                lengths.append(init_data.shape[0])
            else:
                new_length = lengths[i-1] + init_data.shape[0]
                lengths.append(new_length)

            if i == 0:
                adc_data[:lengths[0]] = temp_adc_data
            else:
                adc_data[lengths[i-1]:lengths[i]] = temp_adc_data

        # Reorganizing ADC Data
        adc_data = np.ravel(adc_data)
        adc_data = np.reshape(adc_data, newshape=(self.frames, self.chirps_per_frame,
                                                  self.adc_samples, self.rx_antennas))
        adc_data = np.transpose(adc_data, axes=(0, 1, 3, 2))

        if debug_flag:
            print("\n ADC Data Shape: ", adc_data.shape)
        return adc_data

    @staticmethod
    def calculate_range_resolution(freq_slope, ramp_end_time):
        """Static method to compute range resolution from given chirp information.

        Args:
            ramp_end_time (float): The period of observation or the chirp period Tc.
            freq_slope (float): The rate of increase of frequency in a chirp.

        Returns:
            range_resolution (float): The gap between each range bin. (m)
            bandwidth (float): How much bandwidth is this configuration taking. (in MHz)
        """
        c = 299792458
        bandwidth = freq_slope * ramp_end_time
        range_resolution = c / (2 * bandwidth * 1e6)

        return range_resolution, bandwidth

    @staticmethod
    def calculate_doppler_resolution(bandwidth, start_freq, ramp_time, idle_time, chirp_loops, tx_ant=3):
        """A static method to compute the doppler resolution.

        Args:
            bandwidth (float): Bandwidth of each chirp ramp (MHz)
            start_freq (int): The frequency at which the ramp begins. (GHz)
            ramp_time (int): Duration for which the ramp exists. (us)
            idle_time (int): Time the chip is idle between chirps.
            chirp_loops (int): The number of loops in each frame
            tx_ant (int, optional): The number of transmission antenna. Defaults to 3.

        Returns:
            doppler_resolution (float): The resolution of the doppler bins. (m/s)
        """
        c = 299792458
        # BE CAREFUL WITH UNITS!
        center_freq = ((start_freq * 1e9) + (bandwidth * 1e6)) / 2
        chirp_interval = (ramp_time + idle_time) * 1e-6
        doppler_resolution = c / (2 * chirp_loops * tx_ant * center_freq * chirp_interval)

        return doppler_resolution


def get_points(adc_data, data_processor, data_loader, aggregate=4, num_points=100, save=False, with_snr=True):
    # Final Output should be in the shape (frames, x, y, z, snr)
    if with_snr:
        final_points = np.zeros((adc_data.shape[0], num_points, 4))
    else:
        final_points = np.zeros((adc_data.shape[0], num_points, 3))
    all_frame_points = []

    for frame_index, frame in enumerate(tqdm(adc_data)):
        # Compute per-frame Range FFT
        range_fft = data_processor.range_fft(frame, 'BLACKMAN')
        # Compute per-frame Doppler FFT
        d_display, aoa_input = data_processor.doppler_fft(range_fft, 'BLACKMAN', tx_ant=3)
        # Conduct CFAR Thresholding
        raw_processed_data = data_processor.cfar_thresholding(d_display, data_loader.tx_antennas)
        # Conduct Peak Pruning Opeations
        processed_data = data_processor.peak_operations(raw_processed_data, d_display, data_loader.doppler_bins)
        # Conduct Angle of Arrival Estimation
        x_y_z_vectors = data_processor.angular_map(aoa_input[processed_data['rangeIdx'], :,
                                                             processed_data['dopplerIdx']].T,
                                                   processed_data, rx_ant=data_loader.rx_antennas)
        # Range Based Point Limiting
        xyz_vector, snr_values = data_processor.range_based_filtering(x_y_z_vectors, processed_data['SNR'],
                                                                      range_threshold=20.0, angular_threshold=20.0)

        # Combine Points with SNR (x, y, z, SNR)
        if with_snr:
            frame_points = np.zeros((4, xyz_vector.shape[1]))
            frame_points[:3, :] = xyz_vector
            frame_points[3, :] = snr_values
        else:
            frame_points = np.zeros((3, xyz_vector.shape[1]))
            frame_points[:3, :] = xyz_vector

        all_frame_points.append(frame_points)

        # Aggregate Points
        if frame_index > aggregate-1:
            aggregated_points = []
            for i in range(aggregate):
                if i == 0:
                    # print(all_frame_points[frame_index])
                    aggregated_points = all_frame_points[frame_index]
                else:
                    aggregated_points = np.append(aggregated_points, all_frame_points[frame_index - i], axis=1)
            # print("Aggregated Frame Length :", aggregated_points.shape)
        else:
            continue

        display_3d_image(frame_points, "Frame "+str(frame_index))

        # Final Point Limiter
        # Important Assumption here is that the aggregated points ARE GREATER THAN REQUIRED POINTS!
        # First Sort the Data Based on SNR Values
        snr_val = aggregated_points.T[:, 3]
        sorted_snr_val = np.sort(snr_val)[-num_points:]
        # Combine Data and Store them!
        for i, element in enumerate(sorted_snr_val):
            indices = np.argwhere(aggregated_points == element)[0][1]
            final_points[frame_index, i, :] = aggregated_points[:, indices]

    final_points = final_points[aggregate:, :, :]
    if save:
        file_name = input("\nEnter File Name: ")
        np.save(file_name, final_points)
        print("\nSaved Data as NumPy File.")

    return final_points


# Display


