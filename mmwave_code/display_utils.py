import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd


def display_image(input_data, title, pause_interval=0.05):
    image = np.abs(input_data)
    image -= np.min(image)
    image /= np.max(image)
    plt.imshow(image)
    plt.title(title)
    plt.pause(pause_interval)
    plt.clf()


def display_3d_image(frame_data, title):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(frame_data[0, :], frame_data[1, :],
                     frame_data[2, :], c=frame_data[3, :], cmap=plt.hot())
    fig.colorbar(img)
    ax.set_xlabel('Azimuth (m)')
    ax.set_ylabel('Range (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_xlim([-10, 10])
    ax.set_zlim([-10, 10])
    ax.set_ylim([0, 20])
    ax.set_title(title)
    plt.show()


def plot_history(history, title=''):
    plt.suptitle(title)
    plt.subplot(3, 1, 1)
    plt.plot(history.history["categorical_accuracy"])
    plt.plot(history.history["val_categorical_accuracy"])
    plt.legend(["Training", "Validation"])
    plt.title("Accuracy Plots")
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["Training", "Validation"])
    plt.title("Loss Plots")
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(history.history["auc"])
    plt.plot(history.history["val_auc"])
    plt.legend(["Training", "Validation"])
    plt.title("AUC Plots")
    plt.grid()
    plt.show()


def confusion_matrix(c_matrix, name_len):
    df_cm = pd.DataFrame(c_matrix.numpy(), index=[i for i in np.arange(1, name_len+1, 1)],
                         columns=[i for i in np.arange(1, name_len+1, 1)])
    plt.figure(figsize=(10, 7))
    plt.title("Confusion Matrix With SNR")
    sb.heatmap(df_cm, annot=True)
    plt.show()


plt.figure()
# x = [0, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
# y = [43.57, 46.12, 57.11, 66.3, 68.26, 74.19, 81.1, 88.28, 94.79, 95.52, 98.92, 99.1, 99.63, 99.15, 100, 100, 100]

x = [1, 4, 10]
y1 = [98.77, 98.69, 100]
y2 = [88.81, 96.39, 100]

plt.plot(x, y1, 'r', x, y2, 'b', x, y1, 'ko', x, y2, 'kx')
plt.legend(['300 Points', '500 Points'])
plt.grid()
plt.xlabel('Aggregated Frames')
plt.ylabel('Testing Accuracy')
plt.title('Frame Aggregation vs Testing Accuracy & Num Points')
plt.show()

