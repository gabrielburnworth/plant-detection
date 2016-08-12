"""Image Pixel Location to Machine Coordinate Conversion."""

from time import sleep
import cv2
import numpy as np


class Pixel2coord():
    """Calibrates the conversion of pixel locations to machine coordinates
    in images. Finds object coordinates in image.
    """
    def __init__(self, calibration_image):
        self.image = None
        if isinstance(calibration_image, str):
            self.readimage(calibration_image)
        else:
            self.image = calibration_image
        self.proc = None
        self.circled = None
        self.calibration_object_pixel_locations = []
        self.center_pixel_location = np.array(self.image.shape[:2][::-1]) / 2
        self.rotationangle = 0
        self.calibration_circles_xaxis = True  # else, calibration circles spaced along y-axis
        self.image_bot_origin_location = [0, 1]  # bot axes locations in image
        self.calibration_circle_separation = 1000  # distance between red dots
        self.camera_offset_coordinates = [200, 100]  # camera offset from current location
        self.test_rotation = 5  # for testing, add some image rotation
        self.iterations = 3  # min 2 if image is rotated or if rotation is unknown
        self.viewoutputimage = False  # overridden as True if running script
        self.fromfile = True  # otherwise, take photos
        self.coord_scale = None
        self.total_rotation_angle = 0
        self.testimage = 0

    def getcoordinates(self, test):
        """Get machine coordinates from bot."""
        # For now, testing coordintes:
        bot_coordinates = [600, 400]  # for calib image, current location
        if test:
            bot_coordinates = [600, 400]  # for testing image, current location
        return bot_coordinates

    def getimage(self):
        """Take a photo."""
        camera = cv2.VideoCapture(0)
        sleep(0.1)
        _, self.image = camera.read()
        camera.release()

    def readimage(self, filename):
        """Read an image from a file."""
        self.image = cv2.imread(filename)

    def showimage(self, image_to_show):
        """Show an image."""
        cv2.imshow("image", image_to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rotationdetermination(self):
        """Determine angle of rotation if necessary."""
        threshold = 0
        obj_1_x, obj_1_y, _ = self.calibration_object_pixel_locations[0]
        obj_2_x, obj_2_y, _ = self.calibration_object_pixel_locations[1]
        if not self.calibration_circles_xaxis:
            if obj_1_x > obj_2_x:
                obj_1_x, obj_2_x = obj_2_x, obj_1_x
                obj_1_y, obj_2_y = obj_2_y, obj_1_y
        dx = (obj_1_x - obj_2_x)
        dy = (obj_1_y - obj_2_y)
        if self.calibration_circles_xaxis:
            difference = abs(dy)
            trig = difference / dx
        else:
            difference = abs(dx)
            trig = difference / dy
        if difference > threshold:
            rotation_angle_radians = np.tan(trig)
            self.rotationangle = 180. / np.pi * rotation_angle_radians

    def rotateimage(self, rotationangle):
        """Rotate image number of degrees."""
        try:
            rows, cols, _ = self.image.shape
        except ValueError:
            rows, cols = self.image.shape
        mtrx = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotationangle, 1)
        self.image = cv2.warpAffine(self.image, mtrx, (cols, rows))

    def process(self):
        """Prepare image for contour detection."""
        blur = cv2.medianBlur(self.image, 5)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask_L = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([20, 255, 255]))
        mask_U = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))
        mask = cv2.addWeighted(mask_L, 1.0, mask_U, 1.0, 0.0)
        self.proc = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    def findobjects(self, **kwargs):
        """Create contours and find locations of objects."""
        small_c = False  # default
        circle = True  # default
        draw_contours = True  # default
        for key in kwargs:
            if key == 'small_c': small_c = kwargs[key]
            if key == 'circle': circle = kwargs[key]
            if key == 'draw_contours': draw_contours = kwargs[key]
        contours, _ = cv2.findContours(
            self.proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.circled = self.image.copy()
        for i, cnt in enumerate(contours):
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if i == 0:
                self.calibration_object_pixel_locations = [cx, cy, radius]
            else:
                self.calibration_object_pixel_locations = np.vstack(
                    (self.calibration_object_pixel_locations,
                     [cx, cy, radius]))
            center = (int(cx), int(cy))
            if small_c:
                radius = 20
            if circle:
                cv2.circle(self.circled, center, int(radius), (255, 0, 0), 4)
            if draw_contours:
                cv2.drawContours(self.proc, [cnt], 0, (255, 255, 255), 3)
                cv2.drawContours(self.circled, [cnt], 0, (0, 255, 0), 3)

    def calibrate(self):
        """Determine coordinate conversion parameters."""
        if len(self.calibration_object_pixel_locations) > 1:
            calibration_circle_sep = float(self.calibration_circle_separation)
            if self.calibration_circles_xaxis:
                i = 0
            else:
                i = 1
            object_sep = abs(self.calibration_object_pixel_locations[0][i] -
                             self.calibration_object_pixel_locations[1][i])
            self.coord_scale = calibration_circle_sep / object_sep

    def p2c(self, object_pixel_locations):
        """Convert pixel locations to machine coordinates from image center."""
        # TODO: ability to process only one coordinate (e.g. x instead of [x, y, r])
        object_pixel_locations = np.array(object_pixel_locations)
        coord = np.array(self.getcoordinates(self.testimage), dtype=float)
        camera_offset = np.array(self.camera_offset_coordinates, dtype=float)
        camera_coordinates = coord + camera_offset  # image center coordinates
        sign = [1 if s == 1 else -1 for s in self.image_bot_origin_location]
        coord_scale = np.array([self.coord_scale, self.coord_scale])
        object_coordinates = []
        print "Detected object machine coordinates ( X Y ) with R = radius:"
        for o, object_pixel_location in enumerate(object_pixel_locations[:, :2]):
            radius = object_pixel_locations[:][o][2]
            moc = (camera_coordinates +
                   sign * coord_scale *
                   (self.center_pixel_location - object_pixel_location))
            print "    ( {:5.0f} {:5.0f} ) R = {R:.0f}".format(*moc, R=radius)
            object_coordinates.append(
                [moc[0], moc[1], coord_scale[0] * radius])
        return object_coordinates, object_pixel_locations

    def c2p(self, object_coordinates):
        """Convert machine coordinates to pixel locations
        using image center."""
        object_coordinates = np.array(object_coordinates)
        coord = np.array(self.getcoordinates(0), dtype=float)
        camera_offset = np.array(self.camera_offset_coordinates, dtype=float)
        camera_coordinates = coord + camera_offset  # image center coordinates
        center_pixel_location = self.center_pixel_location[:2]
        sign = [1 if s == 1 else -1 for s in self.image_bot_origin_location]
        coord_scale = np.array([self.coord_scale, self.coord_scale])
        object_pixel_locations = []
        for o, object_coordinate in enumerate(object_coordinates[:, :2]):
            opl = (center_pixel_location -
                   ((object_coordinate - camera_coordinates) / (sign * coord_scale)))
            object_pixel_locations.append([opl[0], opl[1],
                                           object_coordinates[o][2] / coord_scale[0]])
        return object_pixel_locations

    def calibration(self):
        """Determine pixel to coordinate conversion scale
        and image rotation angle."""
        for i in range(0, self.iterations):
            self.process()
            self.findobjects()
            if i != (self.iterations - 1):
                self.rotationdetermination()
                self.rotateimage(self.rotationangle)
                self.total_rotation_angle += self.rotationangle
        if self.total_rotation_angle != 0:
            print " Note: required rotation executed = {:.2f} degrees".format(
                self.total_rotation_angle)
        self.calibrate()
        if self.viewoutputimage:
            self.showimage(self.circled)

    def determine_coordinates(self):
        """Use calibration parameters to determine locations of objects."""
        self.rotateimage(self.total_rotation_angle)
        self.process()
        self.findobjects()
        _, _ = self.p2c(self.calibration_object_pixel_locations)
        if self.viewoutputimage:
            self.showimage(self.circled)

if __name__ == "__main__":
    P2C = Pixel2coord("p2c_test_calibration.jpg")
    P2C.viewoutputimage = True
    if P2C.fromfile:
        # From files
        # Calibration
        P2C.rotateimage(P2C.test_rotation)
        P2C.calibration()
        # Tests
        # Object detection
        P2C.readimage("p2c_test_objects.jpg")
        P2C.rotateimage(P2C.test_rotation)
        P2C.determine_coordinates()
        # Color range
        P2C.readimage("p2c_test_color.jpg")
        P2C.process()
        P2C.findobjects(circle=False)
        if P2C.viewoutputimage:
            P2C.showimage(P2C.circled)

    else:
        # Use camera
        P2C.getimage()
        P2C.calibration()
        P2C.determine_coordinates()
