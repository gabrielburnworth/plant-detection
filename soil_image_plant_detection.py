"""Plant Detection.

Detects green plants on a dirt background
 and marks them with red circles.
"""
import numpy as np
import cv2

def detect_plants(image, **kwargs):
    """Detect plants in image and saves an image with plants marked.
       
       Args:
           image (str): filename of image to process
       Kwargs:
           debug (boolean): output debug images
           morph (int): amount of filtering (default = 5)
           iterations (int): number of morphological iterations (default = 1)
    """
    debug = False
    morph_amount = None
    iterations = None
    for key in kwargs:
        if key == 'debug': debug = kwargs[key]
        if key == 'morph': morph_amount = kwargs[key]
        if key == 'iterations': iterations = kwargs[key]

    def imsavename(step, description):
        name = image[:-4]
        details = description
        if step is not None:
            details = '{}_{}'.format(step, details)
        if step > 2 or (debug and step is None):
            '{}_morph={}'.format(details, morph_amount)
        filename = '{}_{}.png'.format(name, details)
        print "Image saved: {}".format(filename)
        return filename

    def annotate(img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
                 "blur kernel size = {}".format(blur_amount),
                 "HSV green lower bound = {}".format(lower_green),
                 "HSV green upper bound = {}".format(upper_green),
                 "kernel type = ellipse",
                 "kernel size = {}".format(morph_amount),
                 "morphological transformation = close",
                 "number of iterations = {}".format(iterations)
                 ]
        h = img.shape[0]; w = img.shape[1]
        add = 10 + 10 * len(lines)
        try:
            c = img.shape[2]
            new_shape = (h + add, w, c)
        except IndexError:
            new_shape = (h + add, w)
        annotated_image = np.zeros(new_shape, np.uint8)
        annotated_image[add:, :] = img
        for o, line in enumerate(lines):
            cv2.putText(annotated_image, line, (10, 10 + o * 10), font, 0.3, (255,255,255), 1)
        return annotated_image
    
    print "\nProcessing image: {}".format(image)
    
    # Load image and create blurred image
    img = cv2.imread(image, 1)
    blur_amount = 5 # must be odd
    blur = cv2.medianBlur(img, blur_amount)
    if debug:
        img2 = img.copy()
        cv2.imwrite(imsavename(0, 'blurred'), blur)

    # Create HSV image and select HSV color bounds for mask
    # Hue range: [0,179], Saturation range: [0,255], Value range: [0,255]
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([90, 255, 255])

    # Create plant mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    if debug:
        cv2.imwrite(imsavename(1, 'mask'), mask)
        res = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(imsavename(2, 'masked'), res)

    # Process mask to try to make plants more coherent
    if morph_amount is None: morph_amount = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_amount, morph_amount))
    if iterations is None: iterations = 1
    proc = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    if debug:
        proc2 = annotate(proc)
        cv2.imwrite(imsavename(3, 'processed-mask'), proc2)
        res2 = cv2.bitwise_and(img, img, mask=proc)
        res2 = annotate(res2)
        cv2.imwrite(imsavename(4, 'processed-masked'), res2)

    # Find contours (hopefully of outside edges of plants)
    contours, hierarchy = cv2.findContours(proc, 1, 2)
    print "{} plants detected in image.".format(len(contours))

    # Loop through contours
    for i in range(len(contours)):
        # Calculate plant location by using centroid of contour
        cnt = contours[i]
        M = cv2.moments(cnt)
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        except ZeroDivisionError:
            continue
        if i == 0:
            print "Detected plant center pixel coordinates:"
        print "    x={} y={}".format(cx, cy)

        # Mark plant with red circle
        cv2.circle(img, (cx,cy), 20, (0,0,255), 4)

        if debug:
            cv2.drawContours(proc, [cnt], 0, (255,255,255), 3)
            cv2.circle(img2, (cx,cy), 20, (0,0,255), 4)
            cv2.drawContours(img2, [cnt], 0, (255,255,255), 3)

    if debug:
        proc = annotate(proc)
        cv2.imwrite(imsavename(5, 'contours'), proc)
        img2 = annotate(img2)
        cv2.imwrite(imsavename(6, 'img-contours'), img2)

    # Save soil image with plants marked
    cv2.imwrite(imsavename(None, 'marked'), img)

if __name__ == "__main__":
    single_image = True
    if single_image:
        image = "soil_image.jpg"
        detect_plants(image)
    else: # multiple images to process
        images = ["soil_image_{}.jpg".format(i) for i in range(0,7)]
        for image in images:
            detect_plants(image, morph=3, iterations=5, debug=True)
