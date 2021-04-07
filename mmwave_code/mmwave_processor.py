from tqdm import tqdm
import numpy as np
from cfar_algorithm import cell_averaging as cfar_ca
import matplotlib.pyplot as plt
from mmwave import dsp

class MMWaveProcessor:
    def __init__(self, range_resolution, doppler_resolution, range_bins=512, doppler_bins=128, angular_bins=64):
        self.range_resolution = range_resolution
        self.doppler_resolution = doppler_resolution
        self.range_bins = range_bins
        self.doppler_bins = doppler_bins
        self.angle_bins = angular_bins


    def range_fft(self, adc_data, window = None, axis=-1):
        """Conducting the Range FFT on the ADC Data

        Args:
            adc_data (ndarray): The ADC data per frame on which Range FFT is being conducted.
                                Shape should be: #ChirpsPerFrame*#TxAntennas, #Receivers, #ADCsamples
            window (string, optional): The windowing technique to apply. Defaults to None. Options are:
                                       BARTLETT or bartlett
                                       BLACKMAN or blackman
                                       HAMMING or hamming
                                       HANNING or hanning
            axis (int, optional): Axis along which windowing will be done. Defaults to -1.

        Returns:
            radar_cube (ndarray): Final output of RangeFFT.
        """
        if window:
            windowed_data = self.windowing_method(adc_data, window, axis)
        else:
            windowed_data = adc_data

        radar_cube = np.fft.fft(windowed_data, axis=axis)

        return radar_cube


    def doppler_fft(self, range_fft, window=None, tx_ant=2, fftshift=False):
        """A function to generate the doppler FFT on the range FFT data or the 2D fft on the ADC data.
        IMPORTANT_NOTE: This functional also seperates the chirps and tx antennas. This process is needed
        before thte Doppler FFT is conducted. It is also important to be mindful of the fact that Virtual
        Antennas is the product of Receivers and Transmitters. For the IWR1443 with all the antennas in use,
        this value is 12 = 4 * 3. 

        Args:
            range_fft (ndarray): The Range FFT per frame on which Doppler FFT is being conducted.
                                 Shape should be: #ChirpsPerFrame*#TxAntennas, #Receivers, #RangeBins
            window (string, optional): The windowing technique to apply. Defaults to None. Options are:
                                       BARTLETT or bartlett
                                       BLACKMAN or blackman
                                       HAMMING or hamming
                                       HANNING or hanning
            tx_ant (int, optional): Number of Tx Antennas. Defaults to 2.

        Returns:
            doppler_display (ndarray): Doppler FFT ready for display. (2 Dim)
            doppler_fft_data (ndarray): Needed for Next Steps (3 Dim)
        """
        # Data by default SHOULD NOT BE interleaved. This process helps seperate the Tx Antennas into
        # Virtual Antennas. This step is critical, as this aids our angle of arrival prediction as well.
        reordering = np.arange(len(range_fft.shape))
        range_fft = range_fft.transpose(reordering)
        output = np.concatenate([range_fft[i::tx_ant, ...] for i in range(tx_ant)], axis=1)

        # Reorganizing the Data as (#Range Bins, #NumVirtualAntennas, #Doppler Bins)
        range_fft = np.transpose(output, axes=(2, 1, 0))
        # Windowing Function
        if window:
            windowed_data = self.windowing_method(range_fft, window, axis=-1)
        else:
            windowed_data = range_fft
        # Doppler FFT
        doppler_fft_data = np.fft.fft(windowed_data)

        # For Display as a Common Output, we usually conduct:
        #           LOG2(ABS) >> CUMULATION >> FFT SHIFT
        # In that order. Cumulation is along all virtual antenna.
        doppler_display = np.log2(np.abs(doppler_fft_data))
        doppler_display = np.sum(doppler_display, axis=1)
        if fftshift:
            doppler_display = np.fft.fftshift(doppler_display, axes=1)

        return doppler_display, doppler_fft_data

    
    def cfar_thresholding(self, detection_matrix, tx_antennas):
        det_matrix = detection_matrix.astype(np.int64)
        threshold_doppler, noise_floor_doppler = np.apply_along_axis(func1d=cfar_ca,
                                                                     axis=0,
                                                                     arr=det_matrix.T,
                                                                     lower_bound=1.5,
                                                                     guard_len=2,
                                                                     training_len=5)
        threshold_range, noise_floor_range = np.apply_along_axis(func1d=cfar_ca,
                                                                 axis=0,
                                                                 arr=det_matrix,
                                                                 lower_bound=2.5,
                                                                 guard_len=1,
                                                                 training_len=3)
        threshold_doppler, noise_floor_doppler = threshold_doppler.T, noise_floor_doppler.T
        det_doppler_mask = (det_matrix > threshold_doppler)
        det_range_mask = (det_matrix > threshold_range)
        full_mask = (det_doppler_mask & det_range_mask)
        det_peaks_indices = np.argwhere(full_mask == True)  # WEIRD but do not change the == to 'is'
        peak_values = detection_matrix[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

        snr = peak_values - noise_floor_range[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        dtype_det_obj2d = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                   'formats': ['<i4', '<i4', '<f4', '(' + str(tx_antennas) + ',)<f4',
                                               '<f4']})
        detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_det_obj2d)
        detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detObj2DRaw['peakVal'] = peak_values.flatten()
        detObj2DRaw['SNR'] = snr.flatten()
        return detObj2DRaw
        
    
    def peak_operations(self, det_obj_2d_raw, det_matrix, doppler_bins):
        
        ### PEAK GROUPING USING NAIVE APPROACH
        range_index = det_obj_2d_raw['rangeIdx']
        doppler_index = det_obj_2d_raw['dopplerIdx']

        next_index = doppler_index + 1
        prev_index = doppler_index - 1
        # Take care of border situations. Basically making it cyclic.
        next_index[doppler_index == doppler_bins - 1] = 0
        prev_index[doppler_index == 0] = doppler_bins - 1

        prev_values = det_matrix[range_index, prev_index]
        current_value = det_matrix[range_index, doppler_index]
        next_values = det_matrix[range_index, next_index]

        pruned_index = (current_value > prev_values) & (current_value > next_values)

        detObj2D = det_obj_2d_raw[pruned_index]

        ### PEAK GROUPING USING FILTER APPROACH
        ### Yet to Implement

        # More pruning along Doppler
        # detObj2D = dsp.peak_grouping_along_doppler(det_obj_2d_pruned, det_matrix, doppler_bins)

        # Code below is used for Range & SNR Based Pruning
        SNRThresholds2 = np.array([[1, 10], [10, 25], [30, 30]])
        peakValThresholds2 = np.array([[1, 500], [1, 400], [500, 0]])
        detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, self.range_bins, 0,
                                           self.range_resolution)
        return detObj2D


    def angular_map(self, azimuth_input, det_obj_2d, rx_ant=4):
        num_det_obj = azimuth_input.shape[1]
        # Azimuth Angle Estimation
        azimuth_antenna1 = azimuth_input[:4, :]
        azimuth_antenna2 = azimuth_input[8:, :]
        azimuth_antenna_padded = np.zeros(shape=(self.angle_bins, num_det_obj), dtype=np.complex_)
        azimuth_antenna_padded[:4, :] = azimuth_antenna1
        azimuth_antenna_padded[4:8, :] = azimuth_antenna2
        azimuth_fft = np.fft.fft(azimuth_antenna_padded, axis=0)
        k_max = np.argmax(np.abs(azimuth_fft), axis=0)
        azimuth_peaks = np.zeros_like(k_max, dtype=np.complex_)
        for j in range(len(k_max)):
            azimuth_peaks[j] = azimuth_fft[k_max[j], j]

        k_max[k_max > (self.angle_bins//2) - 1] = k_max[k_max > (self.angle_bins//2) - 1] - self.angle_bins
        wx = 2 * np.pi / self.angle_bins * k_max
        x_vector = wx / np.pi

        # Elevation Angle Estimation
        elevation_antenna = azimuth_input[4:8, :]
        elevation_antenna_padded = np.zeros(shape=(self.angle_bins, num_det_obj), dtype=np.complex_)
        elevation_antenna_padded[:rx_ant, :] = elevation_antenna
        elevation_fft = np.fft.fft(elevation_antenna_padded, axis=0)
        k_max_2 = np.argmax(np.log2(np.abs(elevation_fft)), axis=0)
        elevation_peaks = np.zeros_like(k_max_2, dtype=np.complex_)
        for j in range(len(k_max_2)):
            elevation_peaks[j] = elevation_fft[k_max_2[j], j]

        wz = np.angle(azimuth_peaks * elevation_peaks.conj() * np.exp(1j * 2 * wx))
        z_vector = wz / np.pi
        y_vector = np.sqrt(np.abs(1 - x_vector ** 2 - z_vector ** 2))

        return self.add_range_resolution(x_vector, y_vector, z_vector, det_obj_2d)
    

    def add_range_resolution(self, x, y, z, det_obj_2d):
        xyz_vec_n = np.zeros((3, x.shape[0]))
        xyz_vec_n[0] = x * self.range_resolution * det_obj_2d['rangeIdx']
        xyz_vec_n[1] = y * self.range_resolution * det_obj_2d['rangeIdx']
        xyz_vec_n[2] = z * self.range_resolution * det_obj_2d['rangeIdx']

        return xyz_vec_n


    @staticmethod
    def windowing_method(input_data, window_type, axis=0):
        """
        Function to generate the window and conduct the windowing operation on the input data.

        Args:
            input_data (ndarray): The input data on which the windowing operation needs to be
                                  conducted.
            window_type (string): The windowing technique to apply. Options are:
                                  BARTLETT or bartlett
                                  BLACKMAN or blackman
                                  HAMMING or hamming
                                  HANNING or hanning
            axis (int, optional): Axis along which windowing will be done. Defaults to 0.

        Raises:
            ValueError: Incorrect entry for windowing operation. The windowing operation has not
                        been implemented.

        Returns:
            output_data : Windowed data ready for FFT.
        """
        window_length = input_data.shape[axis]
        if (window_type == "BARTLETT") or (window_type == "bartlett"):
            window = np.bartlett(window_length)
        elif (window_type == "BLACKMAN") or (window_type == "blackman"):
            window = np.blackman(window_length)
        elif (window_type == "HAMMING") or (window_type == "hamming"):
            window = np.hamming(window_length)
        elif (window_type == "HANNING") or (window_type == "hanning"):
            window = np.hanning(window_length)
        else:
            raise ValueError("\n Incorrect Windowing Operation. Not Implemented.")

        output_data = input_data * window

        return output_data


    @staticmethod
    def range_based_filtering(xyz_vector, snr_values, range_threshold, angular_threshold=1):
        counter = 0
        new_xyz_vector = np.zeros(xyz_vector.shape)
        new_snr_values = np.zeros(snr_values.shape)
        for i in range(xyz_vector.shape[1]):
            if (xyz_vector[1, i] < range_threshold) and (np.abs(xyz_vector[0, i]) < angular_threshold) \
                    and (np.abs(xyz_vector[2, i]) < angular_threshold):
                new_xyz_vector[:, counter] = [xyz_vector[0, i], xyz_vector[1, i], xyz_vector[2, i]]
                new_snr_values[counter] = snr_values[i]
                counter += 1
        new_xyz_vector = new_xyz_vector[:, :counter]
        new_snr_values = new_snr_values[:counter]
        return new_xyz_vector, new_snr_values
