import matplotlib.pyplot as plt
import numpy as np

def display_image(input_data, title, pause_interval=0.05):
    image = np.abs(input_data)
    image -= np.min(image)
    image /= np.max(image)
    plt.imshow(image)
    plt.title(title)
    plt.pause(pause_interval)
    plt.clf()

