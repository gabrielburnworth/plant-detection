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
           debug (boolean): output debug images (default = False)
           blur (int): blur kernel size (must be odd, default = 5)
           morph (int): amount of filtering (default = 5)
           iterations (int): number of morphological iterations (default = 1)
           array (list): list of morphs to run
               [[morph kernel size, morph kernel type, morph type, iterations]]
               example: array=[[3, 'cross', 'dilate', 2],
                               [5, 'rect',  'erode',  1]]
           save (boolean): save images (default = True)
       Examples:
           detect_plants('soil_image.jpg')
           detect_plants('soil_image.jpg', blur=10, morph=3, iterations=10, debug=True)
           detect_plants('soil_image.jpg', blur=15, array=[[5, 'ellipse', 'erode', 2],
                     [3, 'ellipse', 'dilate', 8]], debug=True)
    """
    debug = False # default
    blur_amount = None   # To allow values to be defined as kwargs
    morph_amount = None #  and keep defaults in the relevant 
    iterations = None   #  sections of code.
    array = None
    save = True
    for key in kwargs:
        if key == 'debug': debug = kwargs[key]
        if key == 'blur': blur_amount = kwargs[key]
        if key == 'morph': morph_amount = kwargs[key]
        if key == 'iterations': iterations = kwargs[key]
        if key == 'array': array = kwargs[key]
        if key == 'save': save = kwargs[key]

    def save_image(img, step, description):
        save_image = img
        if step is not None: # debug image
            save_image = annotate(img)
        name = image[:-4]
        details = description
        if step is not None: # debug image
            details = 'debug-{}_{}'.format(step, details)
        filename = '{}_{}.png'.format(name, details)
        if save:
            cv2.imwrite(filename, save_image)
            print "Image saved: {}".format(filename)
        return save_image

    def annotate(img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = ["blur kernel size = {}".format(blur_amount)] # blur applied
        if upper_green is not None: # color mask applied
            lines = lines + [
                 "HSV green lower bound = {}".format(lower_green),
                 "HSV green upper bound = {}".format(upper_green)]
            if array is None and kt is not None: # single morph applied
                lines = lines + [
                 "kernel type = {}".format(kernel_type),
                 "kernel size = {}".format(morph_amount),
                 "morphological transformation = {}".format(morph_type),
                 "number of iterations = {}".format(iterations)]
        h = img.shape[0]; w = img.shape[1]
        textsize = w / 1200.
        lineheight = int(40 * textsize); textweight = int(3.5 * textsize)
        add = lineheight + lineheight * len(lines)
        if array is not None and kt is not None: # multiple morphs applied
            add_1 = add
            add += lineheight + lineheight * len(array)
        try: # color image?
            c = img.shape[2]
            new_shape = (h + add, w, c)
        except IndexError:
            new_shape = (h + add, w)
        annotated_image = np.zeros(new_shape, np.uint8)
        annotated_image[add:, :] = img
        for o, line in enumerate(lines):
            cv2.putText(annotated_image, line, 
                (10, lineheight + o * lineheight), 
                font, textsize, (255,255,255), textweight)
        if array is not None and kt is not None: # multiple morphs applied
            for o, line in enumerate(array):
                cv2.putText(annotated_image, str(line), 
                    (10, add_1 + o * lineheight), 
                    font, textsize, (255,255,255), textweight)
        return annotated_image
    
    print "\nProcessing image: {}".format(image)
    kt = None; upper_green = None

    # Load image and create blurred image
    img = cv2.imread(image, 1)
    if blur_amount is None: blur_amount = 5
    blur = cv2.medianBlur(img, blur_amount)
    if debug:
        img2 = img.copy()
        save_image(blur, 0, 'blurred')

    # Create HSV image and select HSV color bounds for mask
    # Hue range: [0,179], Saturation range: [0,255], Value range: [0,255]
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 10, 10])
    upper_green = np.array([90, 255, 255])

    # Create plant mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    if debug:
        save_image(mask, 1, 'mask')
        res = cv2.bitwise_and(img, img, mask=mask)
        save_image(res, 2, 'masked')
    
    # Create dictionaries of morph types
    kt = {} # morph kernel type
    kt['ellipse'] = cv2.MORPH_ELLIPSE
    kt['rect'] = cv2.MORPH_RECT
    kt['cross'] = cv2.MORPH_CROSS
    mt = {} # morph type
    mt['close'] = cv2.MORPH_CLOSE
    mt['open'] = cv2.MORPH_OPEN

    # Process mask to try to make plants more coherent
    if array is None:
        # Single morphological transformation
        if morph_amount is None: morph_amount = 5
        kernel_type = 'ellipse'
        kernel = cv2.getStructuringElement(kt[kernel_type], (morph_amount, morph_amount))
        if iterations is None: iterations = 1
        morph_type = 'close'
        proc = cv2.morphologyEx(mask, mt[morph_type], kernel, iterations=iterations)
    else:
        # List of morphological transformations
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
            save_image(proc, '3p{}'.format(p), 'processed-mask')
        array = processes
    if debug:
        save_image(proc, 4, 'processed-mask')
        res2 = cv2.bitwise_and(img, img, mask=proc)
        save_image(res2, 5, 'processed-masked')

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
        save_image(proc, 6, 'contours')
        final_image = save_image(img2, 7, 'img-contours')

    # Save soil image with plants marked
    save_image(img, None, 'marked')

    if not save:
        return final_image

if __name__ == "__main__":
    single_image = True
    if single_image:
        image = "soil_image.jpg"
        detect_plants(image)
    else: # multiple images to process
        images = ["soil_image_{}.jpg".format(i) for i in range(0,7)]
        for image in images:
            detect_plants(image, morph=3, iterations=10, debug=True)

