import cv2
import numpy as np
from numpy.fft import fft2, ifft2

from utils.ex2_utils import get_patch
from utils.ex3_utils import create_gauss_peak, create_cosine_window
from utils.tracker import Tracker


class CorrelationFilterTracker(Tracker):

    def name(self):
        return "Corr"

    def initialize(self, image, region):
        self.enlarge_factor = 1.1  # best 1
        self.gaussian_sigma = 2  # best 4
        self.filter_lambda = 1  # best 1
        self.update_factor = 0.1  # best 0.2

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.search_window = max(region[2], region[3]) * self.enlarge_factor
        self.ideal_gaussian = create_gauss_peak((int(self.search_window), int(self.search_window)), self.gaussian_sigma)
        self.fft_ideal_gaussian = fft2(self.ideal_gaussian)
        self.patch_size = self.ideal_gaussian.shape

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])
        self.cosine_window = create_cosine_window(self.patch_size)

        patch, _ = get_patch(image, self.position, self.patch_size)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = np.multiply(patch, self.cosine_window)

        self.filter_fft_conj = self.construct_filter(patch)

    def construct_filter(self, patch):
        patch_fft = fft2(patch)
        patch_fft_conj = np.conjugate(patch_fft)

        filter_fft = np.divide(
            np.multiply(self.fft_ideal_gaussian, patch_fft_conj),
            np.add(self.filter_lambda, np.multiply(patch_fft, patch_fft_conj))
        )

        return filter_fft

    def track(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        patch, _ = get_patch(image, self.position, self.patch_size)
        patch = np.multiply(patch, self.cosine_window)

        # LOCALIZATION STEP
        patch_fft = fft2(patch)

        correlation_response = ifft2(
            np.multiply(
                patch_fft,
                self.filter_fft_conj
            )
        )

        # Returns maximum peak in correlation response
        y_max, x_max = np.unravel_index(correlation_response.argmax(), correlation_response.shape)

        if x_max > patch.shape[0] / 2:
            x_max = x_max - patch.shape[0]
        if y_max > patch.shape[1] / 2:
            y_max = y_max - patch.shape[1]

        new_x = self.position[0] + x_max
        new_y = self.position[1] + y_max

        self.position = (new_x, new_y)

        # MODEL UPDATE
        patch, _ = get_patch(image, self.position, self.patch_size)
        patch = np.multiply(patch, self.cosine_window)
        new_filter_fft_conj = self.construct_filter(patch)
        self.filter_fft_conj = (1 - self.update_factor) * self.filter_fft_conj + self.update_factor * new_filter_fft_conj

        return [self.position[0] - (self.size[0] / 2), self.position[1] - (self.size[1] / 2), self.size[0], self.size[1]]
