###############################################################################################################
#                                                                                                             #
#   This file contains the class DigitCustomConvolution which is used to parse the specimens of               #
#   handwritten digits                                                                                        #
#                                                                                                             #
###############################################################################################################

import cv2
from os import listdir
from os.path import isfile, join


def get_labels(file_name, labels_dict):
    """
    Static function that chooses labels from a dict and raises exception if filenames are invalid
    :param file_name:
    :param labels_dict:
    :return:
    """
    if "_" not in file_name:
        raise Exception(f"File name {file_name} found in folder path has an invalid name. Valid file name follow "
                        f"<identifier>_(1|2).<extension>, it's a 1 if has the samples from 0-4 and 2 if has the "
                        f"samples from  5-9")
    split_name = file_name.split("_")
    if len(split_name) < 2 or split_name[1].split(".")[0] not in ["1", "2"]:
        raise Exception(f"File name {file_name} found in folder path has an invalid name. Valid file name follow "
                        f"<identifier>_(1|2).<extension>, it's a 1 if has the samples from 0-4 and 2 if has the "
                        f"samples from  5-9")

    return labels_dict[split_name[1].split(".")[0]]


class DigitCustomConvolution:
    """
    This class is used to parse the digits into more usable images
    """

    def __init__(self):
        """
        Constructor that initializes the attributes of the class
        """
        self.layers = []
        self.current_batch = []
        self.exported_images = 0
        self.output_folder = ""

    def add(self, layer_fun):
        """
        Method that adds a layer to the layer stack
        :param layer_fun: function that receives an image
        """
        self.layers.append(layer_fun)

    def load_batch(self, folder_path, labels_dict):
        """
        Method that loads all the imagen within the folder path and adds their respective labels
        :param folder_path: path to the folder containing the images
        :param labels_dict: dictionary containing the labels
        """
        file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        print(file_names)
        for file_name in file_names:
            file_path = join(folder_path, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.bitwise_not(image)
            labels = get_labels(file_name, labels_dict)
            self.current_batch.append((image, labels))

        print(f"Successfully loaded {len(file_names)} images to the current batch.")

    def recursive_save_image(self, batch):
        """
        Function that recursively saves images
        :param batch: a list of images to be saved
        """
        if type(batch) is tuple:
            filename = join(self.output_folder, f"{self.exported_images}_{batch[1]}.jpg")
            self.exported_images += 1
            cv2.imwrite(filename, batch[0])
        else:
            for image in batch: self.recursive_save_image(image)

    def save_batch(self, folder_path):
        """
        Method that saves all the images in the batch
        :param folder_path: folder path to save the images
        """
        self.output_folder = folder_path
        self.recursive_save_image(self.current_batch)

    def evaluate_batch_recursive(self, recursive_batch, layer_fun):
        """
        Method that evaluates the batch with the layer function recursively
        :param recursive_batch: batch to evaluate
        :param layer_fun: function that contains the actions to be taken
        :return: list or single image evaluated
        """
        if type(recursive_batch) is list:
            return [self.evaluate_batch_recursive(batch, layer_fun) for batch in recursive_batch]
        else:
            return layer_fun(recursive_batch)

    def evaluate(self):
        """
        Method that evaluates all the images in the batch with the layer stack
        """
        if self.current_batch is None:
            raise Exception("No batch loaded, to evaluate a batch must be loaded beforehand")
        if not self.layers:
            raise Exception("No layers loaded, to evaluate at least one layer must be added")

        for i, layer in enumerate(self.layers):
            print(f"Evaluating layer {i}.")
            self.current_batch = self.evaluate_batch_recursive(self.current_batch, layer)
