import unittest
import numpy as np

from octLearn.f.graphic import translate_and_crop
from numpy import random as npr
from numpy import testing as npt
from matplotlib import pyplot as plt


def plot_image(img_a, img_b):
    import matplotlib.pyplot as plt
    fig = plt.figure(0)
    ax1 = plt.subplot(2, 2, 2)
    ax2 = plt.subplot(2, 2, 3)
    ax1.imshow(img_a, vmin=0, vmax=2, origin='lower')
    ax1.set_title('truth')
    ax2.imshow(img_b, vmin=0, vmax=2, origin='lower')
    ax2.set_title('product')
    fig.show()
    plt.pause(0.1)


class ImageCropTest(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_image_trans_blank(self):
        # base image
        origin_size = (100, 100)
        no_shift = (0, 0)

        image = np.zeros(origin_size)
        image_a = np.zeros(origin_size)
        image_b = translate_and_crop(image, no_shift, origin_size)
        npt.assert_equal(image_a, image_b)

        for i in range(3):
            random_shift = npr.randint(-50, 50, (2,))
            image_b = translate_and_crop(image, random_shift, origin_size)
            npt.assert_equal(image_a, image_b)

    def test_image_trans_filled(self):
        import math
        # base image
        origin_size = (100, 100)
        no_shift = (0, 0)

        image = np.ones(origin_size)
        # four direction shift
        for x_shift in (-25, 0, 25):
            for y_shift in (-25, 0, 25):
                target_shift = (x_shift, y_shift)
                rect_x_min = max(0, x_shift)
                rect_x_max = min(100, 100 + x_shift)
                rect_y_min = max(0, +y_shift)
                rect_y_max = min(100, 100 + y_shift)

                image_a = np.zeros(origin_size)
                image_a[rect_x_min:rect_x_max, rect_y_min:rect_y_max] = 1
                image_b = translate_and_crop(image, target_shift, origin_size)
                plot_image(image_a, image_b)

                # npt.assert_allclose(image_a, image_b)
                self.assertTrue(np.allclose(image_a, image_b))

        image = np.ones(origin_size)
        for x in range(100):
            for y in range(100):
                image[x, y] = math.sqrt(x ** 2 + y ** 2) / 100
        image_a = image.copy()
        image_b = translate_and_crop(image, no_shift, origin_size)
        self.assertTrue(np.allclose(image_a, image_b))

    # noinspection PyMethodMayBeStatic
    def test_trans_and_complex(self):
        fig, ax = plt.subplots(2, 2)
        image_size = 20
        for _ in range(10):
            points_origin = np.random.randint(0, image_size, size=(20, 2))
            image_origin = np.zeros((image_size, image_size))
            np.put(image_origin, points_origin[:, 0] * image_size + points_origin[:, 1], 1)
            shift = np.random.randint(0, image_size // 2, size=(2,))
            points_shift = points_origin + shift.T
            inbound_index = np.all(np.logical_and(0 <= points_shift, points_shift < image_size), axis=1)
            points_inbound_shift = points_shift[inbound_index]
            image_truth = np.zeros((image_size, image_size))
            np.put(image_truth, points_inbound_shift[:, 0] * image_size + points_inbound_shift[:, 1], 1)
            image_trans = translate_and_crop(image_origin, shift)
            if not np.allclose(image_truth, image_trans):
                image_trans = translate_and_crop(image_origin, shift, debug_ax=ax[0][1])
                ax[0][0].imshow(np.zeros((image_size, image_size)), origin='lower', alpha=0)
                ax[0][0].scatter(*points_shift.T, marker='s')
                ax[1][0].imshow(image_truth, cmap='gray', vmin=0, vmax=1, origin='lower', alpha=1)
                ax[1][0].set_title('excepted')
                ax[1][1].imshow(image_trans, cmap='gray', vmin=0, vmax=1, origin='lower', alpha=1)
                ax[1][1].set_title('result')
                plt.show()
                self.assertTrue(False)

    def test_image_crop_uniform(self):
        fig, ax = plt.subplots()

        # base image
        origin_size = (100, 100)
        image_zeros = np.zeros(origin_size)
        image_ones = np.ones(origin_size)

        for i in range(3):
            random_size = npr.randint(10, 50, (2,))
            random_shift = npr.randint(-25, 25, (2,))

            target_image = translate_and_crop(image_zeros, random_shift, random_size, debug_ax=ax)
            npt.assert_allclose(target_image, np.zeros(random_size))
            self.assertTrue(np.allclose(target_image, np.zeros(random_size)))

            target_image = translate_and_crop(image_ones, random_shift, random_size, debug_ax=ax)
            npt.assert_allclose(target_image, np.ones(random_size))
            self.assertTrue(np.allclose(target_image, np.ones(random_size)))

    # noinspection PyMethodMayBeStatic
    def test_image_oversize_crop(self):
        import math
        # base image
        origin_size = (100, 100)
        input_size = (200, 200)
        no_shift = (0, 0)

        image = np.ones(input_size)
        for x in range(200):
            for y in range(200):
                image[x, y] = math.sqrt(x + y) % 2
                if x % 10 < 5:
                    image[x, y] = 0
        image_a = image.copy()
        image_b = translate_and_crop(image, no_shift, origin_size)
        plot_image(image_a, image_b)

        image_b = translate_and_crop(image, (-67, 3), origin_size)
        plot_image(image_a, image_b)

        image_b = translate_and_crop(image, (-100, 100), origin_size)
        plot_image(image_a, image_b)


if __name__ == '__main__':
    unittest.main()
