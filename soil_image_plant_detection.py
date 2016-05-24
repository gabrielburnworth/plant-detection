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
           array (list): list of morphs to run
               [[morph kernel size, morph kernel type, morph type, iterations]]
               example: array=[[3, 'cross', 'dilate', 2],
                               [5, 'rect',  'erode',  1]]
       Examples:
           detect_plants('soil_image.jpg')
           detect_plants('soil_image.jpg', morph=3, iterations=10, debug=True)
           detect_plants('soil_image.jpg', array=[[3, 'cross', 'dilate', 2],
                     [5, 'rect', 'erode', 1],
                     [5, 'cross', 'erode', 1],
                     [5, 'ellipse', 'open', 1]], debug=True)
    """
    debug = False
    morph_amount = None
    iterations = None
    array = None
    for key in kwargs:
        if key == 'debug': debug = kwargs[key]
        if key == 'morph': morph_amount = kwargs[key]
        if key == 'iterations': iterations = kwargs[key]
        if key == 'array': array = kwargs[key]

    def imsavename(step, description):
        name = image[:-4]
        details = description
        if step is not None:
            details = 'debug-{}_{}'.format(step, details)
        filename = '{}_{}.png'.format(name, details)
        print "Image saved: {}".format(filename)
        return filename

    def annotate(img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = ["blur kernel size = {}".format(blur_amount)]
        if upper_green is not None:
            lines = lines + [
                 "HSV green lower bound = {}".format(lower_green),
                 "HSV green upper bound = {}".format(upper_green)]
            if array is None and kt is not None:
                lines = lines + [
                 "kernel type = {}".format(kernel_type),
                 "kernel size = {}".format(morph_amount),
                 "morphological transformation = {}".format(morph_type),
                 "number of iterations = {}".format(iterations)]
        h = img.shape[0]; w = img.shape[1]
        add = 10 + 10 * len(lines)
        if array is not None and kt is not None:
            add_1 = add
            add += 10 + 10 * len(array)
        try:
            c = img.shape[2]
            new_shape = (h + add, w, c)
        except IndexError:
            new_shape = (h + add, w)
        annotated_image = np.zeros(new_shape, np.uint8)
        annotated_image[add:, :] = img
        for o, line in enumerate(lines):
            cv2.putText(annotated_image, line, 
                (10, 10 + o * 10), font, 0.3, (255,255,255), 1)
        if array is not None and kt is not None:
            for o, line in enumerate(array):
                cv2.putText(annotated_image, str(line), 
                    (10, add_1 + o * 10), font, 0.3, (255,255,255), 1)
        return annotated_image
    
    print "\nProcessing image: {}".format(image)
    kt = None; upper_green = None

    # Load image and create blurred image
    img = cv2.imread(image, 1)
    blur_amount = 5 # must be odd
    blur = cv2.medianBlur(img, blur_amount)
    if debug:
        img2 = img.copy()
        blurA = annotate(blur)
        cv2.imwrite(imsavename(0, 'blurred'), blurA)

    # Create HSV image and select HSV color bounds for mask
    # Hue range: [0,179], Saturation range: [0,255], Value range: [0,255]
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([90, 255, 255])

    # Create plant mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    if debug:
        maskA = annotate(mask)
        cv2.imwrite(imsavename(1, 'mask'), maskA)
        res = cv2.bitwise_and(img, img, mask=mask)
        resA = annotate(res)
        cv2.imwrite(imsavename(2, 'masked'), resA)
    
    # Create dictionaries of morph types
    kt = {} # morph kernel type
    kt['ellipse'] = cv2.MORPH_ELLIPSE
    kt['rect'] = cv2.MORPH_RECT
    kt['cross'] = cv2.MORPH_CROSS
    mt = {} # morph type
    mt['close'] = cv2.MORPH_CLOSE
    mt['open'] = cv2.MORPH_OPEN
    
    if array is None:
        # Process mask to try to make plants more coherent
        if morph_amount is None: morph_amount = 5
        kernel_type = 'ellipse'
        kernel = cv2.getStructuringElement(kt[kernel_type], (morph_amount, morph_amount))
        if iterations is None: iterations = 1
        morph_type = 'close'
        proc = cv2.morphologyEx(mask, mt[morph_type], kernel, iterations=iterations)
    else:
        # Array processing
        processes = array; array = None
        proc = mask
        for p, process in enumerate(processes):
            morph_amount = process[0]; kernel_type = process[1]
            morph_type = process[2]; iterations = process[3]
            kernel = cv2.getStructuringElement(kt[kernel_type], (morph_amount, morph_amount))
            if morph_type == 'erode':
                proc = cv2.erode(proc, kernel, iterations=iterations)
            elif morph_type == 'dilate':
                proc = cv2.dilate(proc, kernel, iterations=iterations)
            else:
                proc = cv2.morphologyEx(proc, mt[morph_type], kernel, iterations=iterations)
            proc_step = annotate(proc)
            cv2.imwrite(imsavename('3p{}'.format(p), 'processed-mask'), proc_step)
        array = processes
    if debug:
        procA = annotate(proc)
        cv2.imwrite(imsavename(4, 'processed-mask'), procA)
        res2 = cv2.bitwise_and(img, img, mask=proc)
        res2 = annotate(res2)
        cv2.imwrite(imsavename(5, 'processed-masked'), res2)

    # Find contours (hopefully of outside edges of plants)
    contours, hierarchy = cv2.findContours(proc, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        cv2.imwrite(imsavename(6, 'contours'), proc)
        img2 = annotate(img2)
        cv2.imwrite(imsavename(7, 'img-contours'), img2)

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
            detect_plants(image, morph=3, iterations=10, debug=True)

