#!/usr/bin/env python
"""Plant Detection Image Processing.

For Plant Detection.
"""
import sys, os
import numpy as np
import cv2
from Capture import Capture
from Parameters import Parameters


class Image():
    """Provide image processes to Plant Detection"""
    def __init__(self, **kwargs):
        self.image = None # working image
        self.original = None
        self.blurred = None
        self.morphed = None
        self.morphed2 = None
        self.masked = None
        self.masked2 = None
        self.marked = None
        self.output_text = True
        self.reduce_large = True
        self.greyed = None
        self.params = Parameters()
        self.dir = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep

    def _reduce(self):
        height, width = self.original.shape[:2]
        if height > 600:
            self.original = cv2.resize(self.original,
                (int(width * 600 / height), 600), interpolation=cv2.INTER_AREA)

    def load(self, filename):
        self.original = cv2.imread(filename, 1)
        self._reduce()
        self.image = self.original.copy()

    def capture(self):
        self.original = Capture().capture()
        self._reduce()
        self.image = self.original.copy()

    def save(self, title):
        filename = '{}{}.jpg'.format(self.dir, title)
        cv2.imwrite(filename, self.image)

    def blur(self):
        self.blurred = cv2.medianBlur(self.image, self.params.blur_amount)
        self.image = self.blurred.copy()

    def mask(self):
        # Create HSV image
        hsv = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2HSV)
        # Select HSV color bounds for mask and create plant mask
        # Hue range: [0,179], Saturation range: [0,255], Value range: [0,255]
        lower_green = self.params.HSV_min
        upper_green = self.params.HSV_max
        if self.params.HSV_min[0] > self.params.HSV_max[0]:
            hsv_btwn_min = [0, self.params.HSV_min[1], self.params.HSV_min[2]]
            hsv_btwn_max = [179, self.params.HSV_max[1], self.params.HSV_max[2]]
            mask_L = cv2.inRange(hsv, np.array(hsv_btwn_min),
                                 np.array(self.params.HSV_max))
            mask_U = cv2.inRange(hsv, np.array(self.params.HSV_min),
                                 np.array(hsv_btwn_max))
            self.masked = cv2.addWeighted(mask_L, 1.0, mask_U, 1.0, 0.0)
        else:
            self.masked = cv2.inRange(hsv, np.array(self.params.HSV_min),
                               np.array(self.params.HSV_max))
        self.image = self.masked.copy()

    def mask2(self):
            self.masked2 = cv2.bitwise_and(self.image,
                                            self.image,
                                            mask=self.masked)
            temp = self.image
            self.image = self.masked2.copy()
            self.save('masked2')
            self.image = temp

    def morph(self):
        # Create dictionaries of morph types
        kt = {}  # morph kernel type
        kt['ellipse'] = cv2.MORPH_ELLIPSE
        kt['rect'] = cv2.MORPH_RECT
        kt['cross'] = cv2.MORPH_CROSS
        mt = {}  # morph type
        mt['close'] = cv2.MORPH_CLOSE
        mt['open'] = cv2.MORPH_OPEN

        # Process mask to try to make plants more coherent
        if self.params.array is None:
            # Single morphological transformation
            kernel_type = 'ellipse'
            kernel = cv2.getStructuringElement(kt[kernel_type],
                                               (self.params.morph_amount,
                                                self.params.morph_amount))
            morph_type = 'close'
            self.morphed = cv2.morphologyEx(self.masked,
                                    mt[morph_type], kernel,
                                    iterations=self.params.iterations)
        else:
            # List of morphological transformations
            processes = self.params.array
            self.morphed = self.masked
            for p, process in enumerate(processes):
                morph_amount = process[0]; kernel_type = process[1]
                morph_type = process[2]; iterations = process[3]
                kernel = cv2.getStructuringElement(kt[kernel_type],
                                                  (morph_amount, morph_amount))
                if morph_type == 'erode':
                    self.morphed = cv2.erode(self.morphed, kernel,
                                             iterations=iterations)
                elif morph_type == 'dilate':
                    self.morphed = cv2.dilate(self.morphed, kernel,
                                              iterations=iterations)
                else:
                    self.morphed = cv2.morphologyEx(self.morphed,
                                            mt[morph_type], kernel,
                                            iterations=iterations)
        self.image = self.morphed.copy()

    def morph2(self):
        self.morphed2 = cv2.bitwise_and(img, img, mask=proc)
        temp = self.image
        self.image = self.morphed.copy()
        self.save('morphed2')
        self.image = temp

    def clump_buster(self):
        try:
            contours, hierarchy = cv2.findContours(self.morphed,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            unused_img, contours, hierarchy = cv2.findContours(self.morphed,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cnt = contours[i]
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            cv2.line(self.morphed, (rx + rw / 2, ry), (rx + rw / 2, ry + rh),
                     (0), rw / 7)
            cv2.line(self.morphed, (rx, ry + rh / 2), (rx + rw, ry + rh / 2),
                     (0), rh / 7)
        kernel = cv2.getStructuringElement('ellipse',
                                          (self.params.morph_amount,
                                           self.params.morph_amount))
        self.morphed = cv2.dilate(self.morphed, kernel, iterations=1)
        self.image = self.morphed.copy()

    def grey(self):
        # Grey out region not selected by mask
        img = self.original
        grey_bg = cv2.addWeighted(np.full_like(img, 255), 0.4, img, 0.6, 0)
        black_fg = cv2.bitwise_and(grey_bg, grey_bg,
                                   mask=cv2.bitwise_not(self.morphed))
        plant_fg_grey_bg = cv2.add(cv2.bitwise_and(img, img,
                                   mask=self.morphed), black_fg)
        self.greyed = plant_fg_grey_bg.copy()

    def annotate(img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = ["blur kernel size = {}".format(self.blur_amount)]  # blur
        if upper_green is not None:  # color mask
            lines = lines + [
                "HSV green lower bound = {}".format(lower_green),
                "HSV green upper bound = {}".format(upper_green)]
            if self.array is None and kt is not None:  # single morph
                lines = lines + [
                    "kernel type = {}".format(kernel_type),
                    "kernel size = {}".format(self.morph_amount),
                    "morphological transformation = {}".format(morph_type),
                    "number of iterations = {}".format(self.iterations)]
        h = img.shape[0]; w = img.shape[1]
        textsize = w / 1200.
        lineheight = int(40 * textsize); textweight = int(3.5 * textsize)
        add = lineheight + lineheight * len(lines)
        if self.array is not None and kt is not None:  # multiple morphs
            add_1 = add
            add += lineheight + lineheight * len(self.array)
        try:  # color image?
            c = img.shape[2]
            new_shape = (h + add, w, c)
        except IndexError:
            new_shape = (h + add, w)
        annotated_image = np.zeros(new_shape, np.uint8)
        annotated_image[add:, :] = img
        for o, line in enumerate(lines):
            cv2.putText(annotated_image, line,
                        (10, lineheight + o * lineheight),
                        font, textsize, (255, 255, 255), textweight)
        if self.array is not None and kt is not None:  # multiple morphs
            for o, line in enumerate(array):
                cv2.putText(annotated_image, str(line),
                            (10, add_1 + o * lineheight),
                            font, textsize, (255, 255, 255), textweight)
        return annotated_image

if __name__ == "__main__":
    image = Image()

    image.capture()
    image.save('captured')

    if len(sys.argv) == 1:
        dir = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep
        soil_image = dir + 'soil_image.jpg'
    else:
        soil_image = sys.argv[1]
    image.load(soil_image)
    image.save('loaded')

    image.blur()
    image.mask()
    image.morph()
    image.save('processed')
