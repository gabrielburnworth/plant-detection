"""Plant Detection.

Detects green plants on a dirt background
 and marks them with red circles.
"""
import numpy as np
import cv2
from pixel_to_coordinate.pixel2coord import Pixel2coord

def detect_plants(image, **kwargs):
    """Detect plants in image and saves an image with plants marked.

       Args:
           image (str): filename of image to process

       Kwargs:
           calibration_img (filename): calibration image filename used to
               output coordinates instead of pixel locations (default = None)
           known_plants (list): [x, y, radius] of known (intentional) plants
                                (default = None)
           debug (boolean): output debug images (default = False)
           blur (int): blur kernel size (must be odd, default = 5)
           morph (int): amount of filtering (default = 5)
           iterations (int): number of morphological iterations (default = 1)
           array (list): list of morphs to run
               [[morph kernel size, morph kernel type, morph type, iterations]]
               example: array=[[3, 'cross', 'dilate', 2],
                               [5, 'rect',  'erode',  1]]
           save (boolean): save images (default = True)
           clump_buster (boolean): attempt to break
                                   plant clusters (default = False)
           HSV_min (list): green lower bound Hue(0-179), Saturation(0-255),
                           and Value(0-255) (default = [30, 20, 20])
           HSV_max (list): green upper bound Hue(0-179), Saturation(0-255),
                           and Value(0-255) (default = [90, 255, 255])

       Examples:
           detect_plants('soil_image.jpg')
           detect_plants('soil_image.jpg', morph=3, iterations=10, debug=True)
           detect_plants("soil_image.jpg", blur=9, morph=7, iterations=4,
              calibration_img="pixel_to_coordinate/p2c_test_calibration.jpg")
           detect_plants('soil_image.jpg', blur=15,
              array=[[5, 'ellipse', 'erode',  2],
                     [3, 'ellipse', 'dilate', 8]], debug=True, save=False,
              clump_buster=True, HSV_min=[15, 15, 15], HSV_max=[85, 245, 245])
    """
    calibration_img = None # default
    known_plants = None # default
    debug = False # default
    blur_amount = None   # To allow values to be defined as kwargs
    morph_amount = None  #  and keep defaults in the relevant
    iterations = None    #  sections of code.
    array = None  # default
    save = True   # default
    clump_buster = False # default
    HSV_min = None       # default in relevant code section
    HSV_max = None       # default in relevant code section
    for key in kwargs:
        if key == 'calibration_img': calibration_img = kwargs[key]
        if key == 'known_plants': known_plants = kwargs[key]
        if key == 'debug': debug = kwargs[key]
        if key == 'blur': blur_amount = kwargs[key]
        if key == 'morph': morph_amount = kwargs[key]
        if key == 'iterations': iterations = kwargs[key]
        if key == 'array': array = kwargs[key]
        if key == 'save': save = kwargs[key]
        if key == 'clump_buster': clump_buster = kwargs[key]
        if key == 'HSV_min': HSV_min = kwargs[key]
        if key == 'HSV_max': HSV_max = kwargs[key]

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
    original_image = cv2.imread(image, 1)
    img = original_image.copy()
    if blur_amount is None: blur_amount = 5
    blur = cv2.medianBlur(img, blur_amount)
    if debug:
        img2 = img.copy()
        save_image(blur, 0, 'blurred')

    # Create HSV image and select HSV color bounds for mask
    # Hue range: [0,179], Saturation range: [0,255], Value range: [0,255]
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    if HSV_min is None: HSV_min = [30, 20, 20]
    if HSV_max is None: HSV_max = [90, 255, 255]
    lower_green = np.array(HSV_min)
    upper_green = np.array(HSV_max)

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
        kernel = cv2.getStructuringElement(kt[kernel_type],
                     (morph_amount, morph_amount))
        if iterations is None: iterations = 1
        morph_type = 'close'
        proc = cv2.morphologyEx(mask,
                   mt[morph_type], kernel, iterations=iterations)
    else:
        # List of morphological transformations
        processes = array; array = None
        proc = mask
        for p, process in enumerate(processes):
            morph_amount = process[0]; kernel_type = process[1]
            morph_type = process[2]; iterations = process[3]
            kernel = cv2.getStructuringElement(kt[kernel_type],
                         (morph_amount, morph_amount))
            if morph_type == 'erode':
                proc = cv2.erode(proc, kernel, iterations=iterations)
            elif morph_type == 'dilate':
                proc = cv2.dilate(proc, kernel, iterations=iterations)
            else:
                proc = cv2.morphologyEx(proc,
                           mt[morph_type], kernel, iterations=iterations)
            save_image(proc, '3p{}'.format(p), 'processed-mask')
        array = processes
    if debug:
        save_image(proc, 4, 'processed-mask')
        res2 = cv2.bitwise_and(img, img, mask=proc)
        save_image(res2, 5, 'processed-masked')

    if clump_buster:
        contours, hierarchy = cv2.findContours(proc,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cnt = contours[i]
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            cv2.line(proc, (rx + rw / 2, ry), (rx + rw / 2, ry + rh),
                     (0,0,0), rw / 25)
            cv2.line(proc, (rx, ry + rh / 2), (rx + rw, ry + rh / 2),
                     (0,0,0), rh / 25)
        proc = cv2.dilate(proc, kernel, iterations=1)

    def find(proc):
        # Find contours (hopefully of outside edges of plants)
        contours, hierarchy = cv2.findContours(proc,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print "{} plants detected in image.".format(len(contours))

        # Loop through contours
        for i, cnt in enumerate(contours):
            # Calculate plant location by using centroid of contour
            M = cv2.moments(cnt)
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                (_, _), radius = cv2.minEnclosingCircle(cnt)
            except ZeroDivisionError:
                continue
            if calibration_img is None:
                if i == 0:
                    print "Detected plant center pixel locations ( X Y ):"
                print "    ( {:5.0f}px {:5.0f}px )".format(cx, cy)

            # Mark plant with red circle
            cv2.circle(img, (cx,cy), 20, (0,0,255), 4)

            if debug:
                cv2.drawContours(proc, [cnt], 0, (255,255,255), 3)
                cv2.circle(img2, (cx,cy), 20, (0,0,255), 4)
                cv2.drawContours(img2, [cnt], 0, (255,255,255), 3)

            object_pixel_locations.append([cx, cy, radius])
        return object_pixel_locations

    object_pixel_locations = []
    if calibration_img is None:
        object_pixel_locations = find(proc)

    # Return coordinates if requested
    if calibration_img is not None:
        P2C = Pixel2coord(calibration_img)
        P2C.calibration()
        def rotateimage(image, rotationangle):
            try:
                rows, cols, _ = image.shape
            except ValueError:
                rows, cols = image.shape
            mtrx = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotationangle, 1)
            return cv2.warpAffine(image, mtrx, (cols, rows))
        inputimage = rotateimage(original_image, P2C.total_rotation_angle)
        proc = rotateimage(proc, P2C.total_rotation_angle)
        object_pixel_locations = find(proc)
        P2C.testimage = True
        plant_coordinates, plant_pixel_locations = P2C.p2c(
                                object_pixel_locations)

        if debug: save_image(img2, None, 'coordinates_found')
        marked_img = inputimage.copy()

        # Known plant exclusion:
        if known_plants is not None:
            # Print known
            print "\n{} known plants inputted.".format(len(known_plants))
            if len(known_plants) > 0:
                print "Plants at the following machine coordinates " + \
                      "( X Y ) with R = radius are to be saved:"
            for known_plant in known_plants:
                print "    ( {:5.0f} {:5.0f} ) R = {:.0f}".format(*known_plant)

            # Find unknown
            marked, unmarked = [], []
            kplants = np.array(known_plants)
            for plant_coord in plant_coordinates:
                x, y, r = plant_coord[0], plant_coord[1], plant_coord[2]
                cxs, cys, crs = kplants[:, 0], kplants[:, 1], kplants[:, 2]
                if all((x - cx)**2 + (y - cy)**2 > cr**2
                       for cx, cy, cr in zip(cxs, cys, crs)):
                    marked.append([x, y, r])
                else:
                    unmarked.append([x, y, r])

            # Print removal candidates
            print "\n{} plants marked for removal.".format(len(marked))
            if len(marked) > 0:
                print "Plants at the following machine coordinates " + \
                      "( X Y ) with R = radius are to be removed:"
            for mark in marked:
                print "    ( {:5.0f} {:5.0f} ) R = {:.0f}".format(*mark)

            # Print saved
            print "\n{} detected plants are known or have escaped "\
                  "removal.".format(len(unmarked))
            if len(unmarked) > 0:
                print "Plants at the following machine coordinates " + \
                      "( X Y ) with R = radius have been saved:"
            for unmark in unmarked:
                print "    ( {:5.0f} {:5.0f} ) R = {:.0f}".format(*unmark)

            # Create annotated image
            known_PL = P2C.c2p(known_plants)
            marked_PL = P2C.c2p(marked)
            unmarked_PL = P2C.c2p(unmarked)
            for mark in marked_PL:
                cv2.circle(marked_img, (int(mark[0]), int(mark[1])),
                           int(mark[2]), (0, 0, 255), 4)
            for known in known_PL:
                cv2.circle(marked_img, (int(known[0]), int(known[1])),
                           int(known[2]), (0, 255, 0), 4)
            for unmarked in unmarked_PL:
                cv2.circle(marked_img, (int(unmarked[0]), int(unmarked[1])),
                           int(unmarked[2]), (255, 0, 0), 4)
        else:
            for ppl in plant_pixel_locations[1:]:
                cv2.circle(marked_img, (int(ppl[0]), int(ppl[1])),
                           int(ppl[2]), (0, 0, 0), 4)

        # Grid
        # TODO: put grid in correct location
        grid_range = np.array([[x] for x in range(0, 2000, 100)])
        large_grid = np.hstack((grid_range, grid_range, grid_range))
        large_grid_pl = np.array(P2C.c2p(large_grid))
        for x, xc in zip(large_grid_pl[:, 0], large_grid[:, 0]):
            if x > marked_img.shape[1] or x < 0:
                continue
            marked_img[:, int(x):int(x + 1)] = (255, 255, 255)
            cv2.putText(marked_img, str(xc), (int(x), 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        for y, yc in zip(large_grid_pl[:, 1], large_grid[:, 1]):
            if y > marked_img.shape[0] or y < 0:
                continue
            marked_img[int(y):int(y + 1), :] = (255, 255, 255)
            cv2.putText(marked_img, str(yc), (100, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        save_image(marked_img, None, 'marked')

    if debug:
        save_image(proc, 6, 'contours')
        final_debug_image = save_image(img2, 7, 'img-contours')

    # Save soil image with plants marked
    if calibration_img is None:
        save_image(img, None, 'marked')

    if debug and not save:
        return final_debug_image

if __name__ == "__main__":
    coordinate_output = False
    single_image = True
    if single_image:
        image = "soil_image.jpg"
        if coordinate_output:
            detect_plants(image, blur=15, morph=6, iterations=4,
             calibration_img="pixel_to_coordinate/p2c_test_calibration.jpg",
             known_plants=[[600, 300, 100], [850, 700, 120]])
        else:
            detect_plants(image)
    else: # multiple images to process
        images = ["soil_image_{:02d}.jpg".format(i) for i in range(0,11)]
        for image in images:
            detect_plants(image, blur=15, morph=6, iterations=10, debug=True)
