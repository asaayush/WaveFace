import numpy as np
from tqdm import tqdm
import os
from mmwave_processor import MMWaveProcessor
import matplotlib.pyplot as plt
from time import time

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
            print("\n Range Resolution: " + str(self.range_resolution) + " cm")
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

        # Load the files based on whether a single or multiple files exist
        for i in range(num_files):
            # Load file name
            if num_files == 1:
                file_name = directory + bin_file_name + ".bin"
            else:
                file_name = directory + bin_file_name + "_" + str(i) + ".bin"

            file = open(file_name, mode='rb')

            # Append if 2nd file or higher
            if i == 0:
                init_data = np.fromfile(file, dtype=np.int16)
            else:
                init_data = np.append(init_data, np.fromfile(file, dtype=np.int16))

        print(" Convert ADC Data into Usable Form")
        # Reshape Data based on LVDS Lanes
        init_data = np.reshape(init_data, newshape=(-1, self.lvds_lanes*2))
        # Initializing ADC Data
        adc_data = np.zeros(shape=(self.frames * self.chirps_per_frame * self.adc_samples, self.lvds_lanes),
                            dtype=complex)
        # Converting I-Q data to Complex Data
        if debug_flag:
            print(" Converting IQ Data to Complex Form")
            t1 = time()
        adc_data[:, 0] = init_data[:, 0] + (1j * init_data[:, 4])
        adc_data[:, 1] = init_data[:, 1] + (1j * init_data[:, 5])
        adc_data[:, 2] = init_data[:, 2] + (1j * init_data[:, 6])
        adc_data[:, 3] = init_data[:, 3] + (1j * init_data[:, 7])
        if debug_flag:
            t2 = time() - t1
            print(" Time Taken is: " + str(t2) + " s")
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
            range_resolution (float): The gap between each range bin. (cm)
            bandwidth (float): How much bandwidth is this configuration taking. (in MHz)
        """
        c = 299792458
        bandwidth = freq_slope * ramp_end_time
        range_resolution = c / (2 * bandwidth * 1e6) * 1e2

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


mm_wave_params = {'adc_samples': 256,
                  'adc_sample_rate': 4000,
                  'freq_slope': 45.48,
                  'real_only': False,
                  'frames': 400,
                  'chirp_loops': 128,
                  'rx_ant': 4,
                  'tx_ant': 3,
                  'ramp_end_time': 87.73,
                  'idle_time': 7,
                  'start_freq': 77,
                  'lvds_lanes': 4,
                  'adc_start_time': 6.4}

mm_wave_params = {'adc_samples': 1024,
                  'adc_sample_rate': 4500,
                  'freq_slope': 14.339,
                  'real_only': False,
                  'frames': 1000,
                  'chirp_loops': 30,
                  'rx_ant': 4,
                  'tx_ant': 3,
                  'angle_bins': 64,
                  'ramp_end_time': 277.73,
                  'idle_time': 100,
                  'start_freq': 77,
                  'lvds_lanes': 4,
                  'adc_start_time': 6.4}

dl = DataLoader(mm_wave_params, debug=True)
adc_data = dl.get_data("data_model_testing/", "adc_data_new_1", debug_flag=True)
processor = MMWaveProcessor(dl.range_resolution, dl.doppler_resolution)
for frame in adc_data:
    range_fft = processor.range_fft(frame, 'BLACKMAN')
    d_display, aoa_input = processor.doppler_fft(range_fft, 'BLACKMAN', tx_ant=3)
    raw_processed_data = processor.cfar_thresholding(d_display, dl.tx_antennas)
    processor.peak_operations(raw_processed_data, d_display, dl.doppler_bins)
    break
