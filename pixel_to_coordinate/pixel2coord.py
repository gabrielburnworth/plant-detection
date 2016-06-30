"""Image Pixel Location to Machine Coordinate Conversion.

Calibrates the conversion of pixel locations to machine coordinates in images.
Finds object coordinates in image.
"""
## Input values and options
calibration_circles_xaxis = True # else, calibration circles spaced along y-axis
image_bot_origin_location = [0, 1] # bot axes locations in image
calibration_circle_separation = 1000 # distance between red dots
bot_coordinates = [600, 400] # for testing, current location
camera_offset_coordinates = [200, 100] # camera offset from current location
test_rotation = 5 # for testing, add some image rotation
iterations = 3 # min 2 if image is rotated or if rotation is unknown
viewoutputimage = False # overridden as True if running script
fromfile = True # otherwise, take photos

import cv2
import numpy as np
from time import sleep

def getcoordinates():
    """Get machine coordinates from bot."""
    return bot_coordinates

def getimage():
    """Take a photo."""
    camera = cv2.VideoCapture(0)
    sleep(0.1)
    return_value, image = camera.read()
    camera.release()
    return image

def readimage(filename):
    """Read an image from a file."""
    image = cv2.imread(filename)
    return image

def showimage(image):
    """Show an image."""
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotationdetermination(image, object_pixel_locations):
    """Determine angle of rotation if necessary."""
    rotationangle = 0 # Leave at zero
    threshold = 0
    obj_1_x, obj_1_y, r1 = object_pixel_locations[1]
    obj_2_x, obj_2_y, r2 = object_pixel_locations[2]
    if not calibration_circles_xaxis:
        if obj_1_x > obj_2_x:
            obj_1_x, obj_2_x = obj_2_x, obj_1_x
            obj_1_y, obj_2_y = obj_2_y, obj_1_y
    dx = (obj_1_x - obj_2_x)
    dy = (obj_1_y - obj_2_y)
    if calibration_circles_xaxis:
        difference = abs(dy)
        trig = difference / dx
    else:
        difference = abs(dx)
        trig = difference / dy
    if difference > threshold:
        rotation_angle_radians = np.tan(trig)
        rotationangle = 180. / np.pi * rotation_angle_radians
    return rotationangle

def rotateimage(image, rotationangle):
    """Rotate image number of degrees."""
    try:
        rows, cols, o = image.shape
    except ValueError:
        rows, cols = image.shape
    Mtrx = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotationangle, 1)
    image = cv2.warpAffine(image, Mtrx, (cols, rows))
    return image

def process(image):
    """Prepare image for contour detection."""
    blur = cv2.medianBlur(image, 5)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask_L = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([20, 255, 255]))
    mask_U = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))
    mask = cv2.addWeighted(mask_L, 1.0, mask_U, 1.0, 0.0)
    proc = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return proc

def findobjects(image, proc, **kwargs):
    """Create contours and find locations of objects."""
    small_c = False # default
    circle = True # default
    draw_contours = True # default
    for key in kwargs:
        if key == 'small_c': small_c = kwargs[key]
        if key == 'circle': circle = kwargs[key]
        if key == 'draw_contours': draw_contours = kwargs[key]
    contours, hierarchy = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circled = image.copy()
    object_pixel_locations = np.append(np.array(image.shape[:2][::-1]) / 2, 0)
    for i in range(len(contours)):
        cnt = contours[i]
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        object_pixel_locations = np.vstack((object_pixel_locations, [cx, cy, radius]))
        center = (int(cx), int(cy))
        if small_c:
            radius = 20
        if circle:
            cv2.circle(circled, center, int(radius), (255, 0, 0), 4)
        if draw_contours:
            cv2.drawContours(proc, [cnt], 0, (255, 255, 255), 3)
            cv2.drawContours(circled, [cnt], 0, (0, 255, 0), 3)
    return object_pixel_locations, circled

def calibrate(object_pixel_locations):
    """Determine coordinate conversion parameters."""
    if len(object_pixel_locations) > 2:
        calibration_circle_sep = float(calibration_circle_separation)
        i = 1
        if calibration_circles_xaxis: i = 0
        object_sep = abs(object_pixel_locations[1][i] - object_pixel_locations[2][i])
        coord_scale = calibration_circle_sep / object_sep
        return coord_scale

def p2c(object_pixel_locations, coord_scale):
    """Convert pixel locations to machine coordinates from image center."""
    object_pixel_locations = np.array(object_pixel_locations)
    coord = np.array(getcoordinates(), dtype=float)
    camera_offset = np.array(camera_offset_coordinates, dtype=float)
    camera_coordinates = coord + camera_offset # image center machine coordinates
    center_pixel_location = object_pixel_locations[0, :2] # image center pixel location
    sign = [1 if s == 1 else -1 for s in image_bot_origin_location]
    coord_scale = np.array([coord_scale, coord_scale])
    object_coordinates = []
    print "Detected object machine coordinates:"
    for o, object_pixel_location in enumerate(object_pixel_locations[1:, :2]):
        moc = ( camera_coordinates +
                sign * coord_scale *
                (center_pixel_location - object_pixel_location) )
        print "    x={:.0f} y={:.0f}".format(*moc)
        object_coordinates.append([moc[0], moc[1], coord_scale[0] * object_pixel_locations[1:][o][2]])
    return object_coordinates, object_pixel_locations

def c2p(center_pixel_location, object_coordinates, coord_scale):
    """Convert machine coordinates to pixel locations using image center."""
    object_coordinates = np.array(object_coordinates)
    coord = np.array(getcoordinates(), dtype=float)
    camera_offset = np.array(camera_offset_coordinates, dtype=float)
    camera_coordinates = coord + camera_offset # image center machine coordinates
    center_pixel_location = center_pixel_location[:2]
    sign = [1 if s == 1 else -1 for s in image_bot_origin_location]
    coord_scale = np.array([coord_scale, coord_scale])
    object_pixel_locations = []
    #print "Detected object pixel locations:"
    for o, object_coordinate in enumerate(object_coordinates[:, :2]):
        opl = ( center_pixel_location -
                ( (object_coordinate - camera_coordinates) / (sign * coord_scale) ) )
        #print "    x={:.0f} y={:.0f}".format(*opl)
        object_pixel_locations.append([opl[0], opl[1],  object_coordinates[o][2] / coord_scale[0]])
    return object_pixel_locations

def calibration(inputimage):
    """Determine pixel to coordinate conversion scale and image rotation angle."""
    if isinstance(inputimage, str):
        inputimage = readimage(inputimage)
    total_rotation_angle = 0 # Leave at zero
    for i in range(0, iterations):
        object_pixel_locations, circled = findobjects(inputimage, process(inputimage))
        if i != (iterations - 1):
            rotation_angle = rotationdetermination(inputimage, object_pixel_locations)
            inputimage = rotateimage(inputimage, rotation_angle)
            total_rotation_angle += rotation_angle
    if total_rotation_angle != 0:
        print "Rotation Required = {:.2f} degrees".format(total_rotation_angle)
    coord_scale = calibrate(object_pixel_locations)
    if viewoutputimage: showimage(circled)
    return coord_scale, total_rotation_angle

def determine_coordinates(inputimage, coord_scale, rotation_angle):
    """Use calibration parameters to determine locations of objects."""
    if isinstance(inputimage, str):
        inputimage = readimage(inputimage)
    inputimage = rotateimage(inputimage, rotation_angle)
    object_pixel_locations, circled = findobjects(inputimage, process(inputimage))
    object_coordinates, object_pixel_locations = p2c(object_pixel_locations, coord_scale)
    if viewoutputimage: showimage(circled)

if __name__ == "__main__":
    viewoutputimage = True
    if fromfile:
        ### From files
        ## Calibration
        image = readimage("p2c_test_calibration.jpg")
        image = rotateimage(image, test_rotation)
        coord_scale, rotation_angle = calibration(image)
        ## Tests
        # Object detection
        testimage = readimage("p2c_test_objects.jpg")
        testimage = rotateimage(testimage, test_rotation)
        determine_coordinates(testimage, coord_scale, rotation_angle)
        # Color range
        testimage = readimage("p2c_test_color.jpg")
        _, outputimage = findobjects(testimage, process(testimage), circle=False)
        if viewoutputimage: showimage(outputimage)

    else:
        ### Use camera
        ## Calibration
        image = getimage()
        coord_scale, rotation_angle = calibration(image)
        ## Test
        image = getimage()
        image = rotateimage(image, test_rotation)
        determine_coordinates(image, coord_scale, rotation_angle)
