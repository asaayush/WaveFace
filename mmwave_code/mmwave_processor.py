from tqdm import tqdm
import numpy as np
from cfar_algorithm import cell_averaging as cfar_ca
import matplotlib.pyplot as plt

class MMWaveProcessor:
    def __init__(self, range_resolution, doppler_resolution):
        self.range_resolution = range_resolution
        self.doppler_resolution = doppler_resolution

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

        new_detected_peaks = det_obj_2d_raw[pruned_index]

        ### PEAK GROUPING USING FILTER APPROACH
        print(" Inside PEAK OPERATIONS \n")

        current_value = det_matrix[range_index, doppler_index]

        print(" Before Pruning: ", len(det_matrix[new_detected_peaks['rangeIdx'], 
              new_detected_peaks['dopplerIdx']]))
        print( " After Pruning: ", len(current_value))


        
        # detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, self.doppler_bins, reserve_neighbor=True)

        # detObj2D = dsp.peak_grouping_along_doppler(det_obj_2d_raw, det_matrix, self.doppler_bins)

        # Code below is used for Range Based Pruning
        """SNRThresholds2 = np.array([[2, 8], [7, 10], [35, 12]])
        peakValThresholds2 = np.array([[1, 500], [1, 400], [500, 0]])
        detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, self.range_bins, 0,
                                           self.range_resolution)"""
        return detObj2D
    
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


