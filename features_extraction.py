from os import listdir
from os.path import isfile, join
import math
import sys
import pickle
import cv2
import numpy as np
from pprint import pprint

from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter


def get_label(filename):
    """
    Function that extracts the label from the filename
    :param filename: name of the digit image
    :return: label in string format
    """
    return filename.split("_")[1].split(".")[0]


def extract_histogram(image, size):
    """
    Function that extracts the histogram of the image with a given size, horizontally and vertically
    :param image: loaded image to extract histogram from
    :param size: size of the stripes
    :return: flattened vector of characteristics of the image
    """
    x_vector = []
    y_vector = []
    rows, cols = image.shape
    for i in range(math.floor(rows / size)):
        y_vector.append(np.count_nonzero(image[i * size:(i + 1) * size, :]))
    for i in range(math.floor(cols / size)):
        x_vector.append(np.count_nonzero(image[:, i * size:(i + 1) * size]))
    return x_vector, y_vector


def save_features(vectors, save_path):
    """
    Function that saves an array of features and labels to a pickle file
    :param save_path: path in which the features will be saved
    :param vectors: array of features to save
    """
    with open(join(save_path, "features_dump.pkl"), "wb") as writer:
        writer.write(pickle.dumps(vectors))
    with open(join(save_path, "features.txt"), "w") as writer:
        writer.write(str(vectors))


def show_histogram(image, h_histogram_y, v_histogram_y, image_name):
    # fig = plt.figure()
    # gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    # (ax1, ax2), (ax3, ax4) = gs.subplots()
    # ax1.imshow(image, aspect="auto")
    # ax2.invert_yaxis()
    # ax2.stairs(np.arange(0, image.shape[1], 4), y_vector + [0], linewidth=2.5)
    # ax3.stairs(x_vector, linewidth=2.5)
    # ax3.invert_yaxis()

    rows, cols = image.shape
    h_histogram_x = np.arange(2, cols, 4)
    v_histogram_x = np.arange(2, rows, 4)

    # Define the locations for the axes
    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left + width + 0.02

    # Set up the geometry of the three plots
    rect_temperature = [left, bottom, width, height]  # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.25]  # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.25, height]  # dimensions of y-histogram

    # Set up the size of the figure
    fig = plt.figure(1, figsize=(9.5, 9))

    # Make the three plots
    axTemperature = plt.axes(rect_temperature)  # temperature plot
    axHistx = plt.axes(rect_histx)  # x histogram
    axHisty = plt.axes(rect_histy)  # y histogram

    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    axHistx.bar([str(i) for i in range(2, math.floor(cols), 4)], h_histogram_y, color='blue')
    axHistx.margins(x=0)
    axHistx.set_title("Histograma horizontal del caracter")
    axHisty.barh([str(i) for i in range(2, math.floor(rows), 4)], v_histogram_y[::-1], color='red')
    axHisty.margins(y=0)
    axHisty.set_title("Histograma vertical\ndel caracter")

    axTemperature.imshow(image, aspect='auto')

    fig.savefig(f"data/histograms/{image_name}_histogram.png")

    fig.clear()


def main(folder_path, save_path):
    """
    Function that retrieves files on the folder, computes their histogram and saves the features vector to the output pickle file
    :param save_path: path in which the features will be saved
    :param folder_path: path where the specimens are
    """
    file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    vectors_and_labels = []
    for file_name in file_names:
        label = get_label(file_name)
        image = cv2.imread(join(folder_path, file_name), cv2.IMREAD_GRAYSCALE)
        x_vector, y_vector = extract_histogram(image, 4)
        show_histogram(image, x_vector, y_vector, file_name.split(".")[0])
        vectors_and_labels.append((x_vector + y_vector, label))

    save_features(vectors_and_labels, save_path)


if __name__ == "__main__":
    # This function will only be run if the current file is run directly
    if len(sys.argv) < 3:
        raise Exception("Missing machine code argument.")
    main(sys.argv[1], sys.argv[2])
