import sys

import custom_layers
from classes.DigitCustomConvolution import DigitCustomConvolution


def extract_each_digit(source_path, target_path):
    """
    Procedure that inits a DigitCustomConvolution object and adds needed layers to parse the dataset and saves it
    :param source_path: folder where the dataset is located
    :param target_path: folder to save the parsed dataset
    """
    convolution_model = DigitCustomConvolution()

    convolution_model.load_batch(source_path, {"1": list(range(5)), "2": list(range(5, 10))})

    convolution_model.add(custom_layers.threshold_and_binary(30))
    convolution_model.add(custom_layers.erode_and_dilate())
    convolution_model.add(custom_layers.discard_empty_space())
    convolution_model.add(custom_layers.evenly_separate_image(1, 5))
    convolution_model.add(custom_layers.discard_empty_space())
    convolution_model.add(custom_layers.evenly_separate_image(17, 3))
    convolution_model.add(custom_layers.discard_empty_space())
    convolution_model.add(custom_layers.cut_frame(0.7))
    convolution_model.add(custom_layers.zoom_in())
    convolution_model.add(custom_layers.cut_frame(0.7))
    convolution_model.add(custom_layers.erode_and_dilate(2, 3))
    convolution_model.add(custom_layers.threshold_and_binary())
    convolution_model.add(custom_layers.cut_frame())
    convolution_model.add(custom_layers.discard_empty_space())
    convolution_model.add(custom_layers.resize((60, 100)))
    convolution_model.add(custom_layers.erode_and_dilate(3, 2))
    convolution_model.add(custom_layers.threshold_and_binary())

    convolution_model.evaluate()

    convolution_model.save_batch(target_path)


if __name__ == "__main__":
    # This function will only be run if the current file is run directly
    if len(sys.argv) < 3:
        raise Exception("Missing machine code argument.")
    extract_each_digit(sys.argv[1], sys.argv[2])
