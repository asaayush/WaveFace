"""
Constant False Alarm Rate
"""

# Import Statements
import numpy as np

# Cell Averaging Strategy CFAR
def cell_averaging(input_signal, guard_len, training_len, lower_bound=4000):
    """
    From MATLAB Website:
        Cell averaging CFAR detector is probably the most widely used CFAR Detector. In cell averaging 
        CFAR detector, noise samples are extracted from both leading and lagging cells (called training
        cells) around the "Cell Under Test" or CUT. In general the leading and lagging training cells are
        the same. Guard cells are placed adjacent to the CUT. This is done to avoid signal components
        from leaking into the training cells, which could affect the noise estimate adversely.

        Training Cells   **
        Guard Cells      ##
        CUT              ?

        Example 1D Signal:       ********** ## ? ## **********

    Args:
        input_signal ([ndarray]): Signal to operate on.
        guard_len ([int]): Length of guard cells from the CUT
        training_len ([int]): Length of samples considered from the CUT
        lower_bound ([int]): A minimum attached to the noise floor
    """

    assert type(input_signal) == np.ndarray

    # Define Kernel for CA CFAR
    kernel = np.ones(1 + (2 * guard_len) + (2 * training_len), dtype=type(input_signal)) / (2 * training_len)
    kernel[training_len:(training_len + (2 * guard_len) + 1)] = 0

    # Convolution
    noise_floor = np.convolve(input_signal, kernel, 'same')

    # Adding Lower Bound to Noise Floor
    threshold = noise_floor + lower_bound
    
    return threshold, noise_floor



