###############################################################################################################
#                                                                                                             #
#   This file contains the custom layers that can be used in the DigitCustomConvolution class as layers       #
#                                                                                                             #
###############################################################################################################

import cv2
import math
import numpy as np


def resize(shape):
    """
    Function that returns the layer function that resizes an image to a given shape
    :param shape: target shape, must be 2D
    :return: layer function
    """

    def resize_inner(labeled_image):
        image, labels = labeled_image
        return cv2.resize(image, shape), labels

    return resize_inner


def discard_empty_space():
    """
    Function that returns the layer function that discards empty space at the edges of an image
    :return: layer function
    """

    def discard_empty_space_inner(labeled_image):
        image, labels = labeled_image
        print(image.shape)
        coords = cv2.findNonZero(image)  # Find all non-zero points
        x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return image[y:y + h, x:x + w], labels

    return discard_empty_space_inner


def erode_and_dilate(erosion_iter=2, dilation_iter=2, kernel_size=(2, 2)):
    """
    Function that returns the layer function that erodes an image and then dilates it to discard not desired spots
    :param kernel_size: size of the kernel used to erode and dilate
    :param erosion_iter: iteration count for the erosion operation
    :param dilation_iter:  iteration count for the dilation operation
    :return: layer function
    """

    def erode_and_dilate_inner(labeled_image):
        image, labels = labeled_image
        print(f"Erode and dilate shape {image.shape}.")
        kernel = np.ones(kernel_size, np.uint8)  # Using a 2x2 kernel to erode and dilate
        img_erosion = cv2.erode(image, kernel, iterations=erosion_iter)
        img_dilation = cv2.dilate(img_erosion, kernel, iterations=dilation_iter)
        return img_dilation, labels

    return erode_and_dilate_inner


def erode(erosion_iter=1, kernel_size=(2, 2)):
    """
    Function that returns the layer function that erodes an image discard not desired spots
    :param kernel_size: size of the kernel used to erode and dilate
    :param erosion_iter: iteration count for the erosion operation
    :return: layer function
    """

    def erode_inner(labeled_image):
        image, labels = labeled_image
        kernel = np.ones(kernel_size, np.uint8)  # Using a 2x2 kernel to erode and dilate
        img_erosion = cv2.erode(image, kernel, iterations=erosion_iter)
        return img_erosion, labels

    return erode_inner


def dilate(dilation_iter=2, kernel_size=(2, 2)):
    """
    Function that returns the layer function dilates and image to enhance values
    :param kernel_size: size of the kernel used to erode and dilate
    :param dilation_iter:  iteration count for the dilation operation
    :return: layer function
    """

    def dilate_inner(labeled_image):
        image, labels = labeled_image
        kernel = np.ones(kernel_size, np.uint8)  # Using a 2x2 kernel to erode and dilate
        img_dilation = cv2.dilate(image, kernel, iterations=dilation_iter)
        return img_dilation, labels

    return dilate_inner


def cut_frame(threshold_ratio=0.90):
    """
    Function that returns the layer function that cuts an image by its borders discard surrounding information
    :param threshold_ratio: percentage of cut to the image
    :return: layer function
    """

    def cut_frame_inner(labeled_image):
        image, labels = labeled_image
        rows, cols = image.shape
        # width_cut = math.floor(cols * cut_ratio)
        # height_cut = math.floor(rows * cut_ratio)
        # return (image[math.floor(height_cut*1.5): rows - height_cut - 1, math.floor(width_cut*1.5): cols - width_cut - 1],
        #         labels)
        rows_th = math.floor(threshold_ratio * rows)
        cols_th = math.floor(threshold_ratio * cols)

        rows_min = 0
        rows_max = rows - 1
        cols_min = 0
        cols_max = cols - 1

        for i in range(rows):
            if cv2.countNonZero(image[i]) >= cols_th:
                rows_min += 1
            else:
                break
        for i in range(rows)[::-1]:
            if cv2.countNonZero(image[i]) >= cols_th:
                rows_max -= 1
            else:
                break
        for i in range(cols):
            if cv2.countNonZero(image[:, i]) >= rows_th:
                cols_min += 1
            else:
                break
        for i in range(cols)[::-1]:
            if cv2.countNonZero(image[:, i]) >= rows_th:
                cols_max -= 1
            else:
                break

        return image[rows_min: rows_max, cols_min: cols_max], labels

    return cut_frame_inner


def czoom_in(zoom_rate=0.1):
    """
    Function that returns the layer function that cuts an image by its borders discard surrounding information
    :param zoom_rate: percentage of cut to the image
    :return: layer function
    """

    def zoom_in_inner(labeled_image):
        image, labels = labeled_image
        rows, cols = image.shape
        width_cut = math.floor(cols * zoom_rate)
        height_cut = math.floor(rows * zoom_rate)
        return image[height_cut: rows - height_cut - 1, width_cut: cols - width_cut - 1], labels

    return zoom_in_inner


def threshold_and_binary(threshold=150):
    """
    Function that returns the layer function that discards information under a given threshold and binaries the image
    :param threshold: threshold value used to discard information
    :return: layer function
    """

    def threshold_and_binary_inner(labeled_image):
        image, labels = labeled_image
        return cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1], labels

    return threshold_and_binary_inner


def evenly_separate_image(horizontal_slices, vertical_slices):
    """
    Function that returns the layer function that divides the image evenly within the given horizontal and vertical stripes
    :param horizontal_slices: number of desired horizontal slices
    :param vertical_slices: number of desired vertical slices
    :return: layer function
    """

    def evenly_separate_image_inner(labeled_image):
        image, labels = labeled_image

        # Check labels len
        if type(labels) is list and horizontal_slices * vertical_slices != len(labels):
            raise Exception(f"Evenly separate an image got an invalid number of labels, "
                            f" {horizontal_slices * vertical_slices} and got {len(labels)}")

        rows, cols = image.shape

        images = []

        # Compute horizontal and vertical stripe sizes
        horizontal_cut_size = math.floor(cols / horizontal_slices)
        vertical_cut_size = math.floor(rows / vertical_slices)

        for i in range(vertical_slices):
            for j in range(horizontal_slices):

                if i == vertical_slices - 1:
                    cut_image = image[i * vertical_cut_size:, j * horizontal_cut_size:(j + 1) * horizontal_cut_size]
                elif j == horizontal_cut_size - 1:
                    cut_image = image[i * vertical_cut_size:(i + 1) * vertical_cut_size, j * horizontal_cut_size:]
                else:
                    cut_image = image[i * vertical_cut_size:(i + 1) * vertical_cut_size,
                                j * horizontal_cut_size:(j + 1) * horizontal_cut_size]

                label = labels[(i + 1) * (j + 1) - 1] if type(labels) is list else labels
                images.append((cut_image, label))
        return images

    return evenly_separate_image_inner


def morphology_close_layer(kernel_size=(4, 4)):
    def morphology_close_layer_inner(labeled_image):
        image, labels = labeled_image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        mask = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return mask, labels

    return morphology_close_layer_inner
