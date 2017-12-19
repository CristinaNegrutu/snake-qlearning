# import Image
from PIL import Image
import numpy as np


class ResizedImage:
    def __init__(self, data, width, height):
        self.data = data
        self.width = width
        self.height = height


class ImageResizer:
    @staticmethod
    def resize_image(image, ratio):
        if ratio > 1:
            raise AssertionError("Image resize accepts only ratios less than 1.")

        image_object = Image.fromarray(image)

        resized_img_shape_0 = int(image.shape[0] * ratio)
        resized_img_shape_1 = int(image.shape[1] * ratio)

        image_object.thumbnail((resized_img_shape_0, resized_img_shape_1), Image.ANTIALIAS)

        resized_image = np.zeros((image_object.height, image_object.width))
        for row_idx in range(0, image_object.height):
            for col_idx in range(0, image_object.width):
                resized_image[row_idx][col_idx] = image_object.getpixel((col_idx, row_idx))

        return ResizedImage(data=resized_image, width=image_object.width, height=image_object.height)


class GrayscaleConverter:
    @staticmethod
    def rgb_to_grayscale(rgb_image):
        """
        Image should be a x * y * 3 shape matrix.
        Returns the grayscale equivalent of the supplied rgb image according
        to the 'weighted average' conversion method.
        """
        grayscale_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))
        for row_idx in range(0, rgb_image.shape[0]):
            for col_idx in range(0, rgb_image.shape[1]):
                grayscale_image[row_idx][col_idx] \
                    = GrayscaleConverter.grayscale_weighted_average(rgb_image[row_idx][col_idx])

        return grayscale_image

    @staticmethod
    def grayscale_weighted_average(rgb_pixel):
        """
        Green is the most prominent color (nearly 60%),
        followed by red(30%) and finally blue(11%).

        @:param rgb_pixel: An array with three elements
        (first element for r, second for green third for blue values).
        """
        return 0.299 * rgb_pixel[0] + 0.587 * rgb_pixel[1] + 0.114 * rgb_pixel[2]
