#!/usr/bin/env python
"""DB for Plant Detection.

For Plant Detection.
"""
import sys, os
import numpy as np

class DB():
    def __init__(self, **kwargs):
        self.output_text = True
        self.output_json = False
        self.known_plants = None
        self.object_count = None
        self.marked = None
        self.unmarked = None
        self.pixel_locations = []
        self.coordinate_locations = []
        self.calibration_pixel_locations = []
        self.dir = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep
        self.known_plants_file = "plant-detection_known-plants.txt"
        self.tmp_dir = None

    def save(self, directory, filename):
        with open(directory + filename, 'w') as f:
            f.write('X Y Radius\n')
            if self.known_plants is not None:
                for plant in self.known_plants:
                    f.write('{} {} {}\n'.format(*plant))

    def save_detected_plants(save, remove):
        if self.tmp_dir is None:
            csv_dir = self.dir
        else:
            csv_dir = self.tmp_dir
        try:
            np.savetxt(csv_dir + "detected-plants_saved.csv", save,
                       fmt='%.1f', delimiter=',', header='X,Y,Radius')
            np.savetxt(csv_dir + "detected-plants_to-remove.csv", remove,
                       fmt='%.1f', delimiter=',', header='X,Y,Radius')
        except IOError:
            self.tmp_dir = "/tmp/"
            save_detected_plants(save, remove)

    def load(self, directory, filename):
        try:  # Load input parameters from file
            with open(self.dir + self.known_plants_file, 'r') as f:
                lines = f.readlines()
                known_plants = []
                for line in lines[1:]:
                    line = line.strip().split(' ')
                    known_plants.append([float(line[0]),
                                         float(line[1]),
                                         float(line[2])])
                if len(known_plants) > 0:
                    self.known_plants = known_plants
        except IOError:
            pass

    def identify(self):
        # Find unknown
        self.marked = []
        self.unmarked = []
        if self.known_plants is None:
            self.known_plants = [[0, 0, 0]]
        kplants = np.array(self.known_plants)
        for plant_coord in self.coordinate_locations:
            x, y, r = plant_coord[0], plant_coord[1], plant_coord[2]
            cxs, cys, crs = kplants[:, 0], kplants[:, 1], kplants[:, 2]
            if all((x - cx)**2 + (y - cy)**2 > cr**2
                   for cx, cy, cr in zip(cxs, cys, crs)):
                self.marked.append([x, y, r])
            else:
                self.unmarked.append([x, y, r])

    def print_count(self, calibration=False):
        if self.output_text:
            if calibration:
                object_name = 'calibration objects'
            else:
                object_name = 'plants'
            print("{} {} detected in image.".format(self.object_count,
                                                    object_name))

    def print_(self):
        # Known plant exclusion:
        if self.known_plants is not None:
            # Print known
            print("\n{} known plants inputted.".format(
                len(self.known_plants)))
            if len(self.known_plants) > 0:
                print("Plants at the following machine coordinates "
                      "( X Y ) with R = radius are to be saved:")
            for known_plant in self.known_plants:
                print("    ( {:5.0f} {:5.0f} ) R = {:.0f}".format(
                    *known_plant))
        else:
            print("\n No known plants inputted.")

        # Print removal candidates
        if self.marked is not None:
            print("\n{} plants marked for removal.".format(len(self.marked)))
            if len(self.marked) > 0:
                print("Plants at the following machine coordinates "
                      "( X Y ) with R = radius are to be removed:")
                for mark in self.marked:
                        print("    ( {:5.0f} {:5.0f} ) R = {:.0f}".format(*mark))

        # Print saved
        if self.unmarked is not None:
            print("\n{} detected plants are known or have escaped "
                  "removal.".format(len(self.unmarked)))
            if len(self.unmarked) > 0:
                print("Plants at the following machine coordinates "
                      "( X Y ) with R = radius have been saved:")
            for unmark in self.unmarked:
                print("    ( {:5.0f} {:5.0f} ) R = {:.0f}".format(*unmark))

    def print_coordinates(self):
        if len(self.coordinate_locations) > 0:
            print("Detected object machine coordinates ( X Y ) with R = radius:")
            for coordinate_location in self.coordinate_locations:
                print("    ( {:5.0f} {:5.0f} ) R = {:.0f}".format(
                                                        coordinate_location[0],
                                                        coordinate_location[1],
                                                        coordinate_location[2]))

    def print_pixel(self):
        if len(self.pixel_locations) > 0:
            print("Detected object center pixel locations ( X Y ):")
            for pixel_location in self.pixel_locations:
                print("    ( {:5.0f}px {:5.0f}px )".format(pixel_location[0],
                                                           pixel_location[1]))

    def json_(self):
        # Encode to CS
        FarmBot = FarmBotJSON()
        for mark in self.marked:
            x, y = round(mark[0], 2), round(mark[1], 2)
            FarmBot.add_point(x, y, 0)
        for unmark in self.unmarked:
            x, y = round(unmark[0], 2), round(unmark[1], 2)
            r = round(unmark[2], 2)
            FarmBot.add_plant(0, [x, y, 0], r)

        # Save plant coordinates to file
        self.save_detected_plants(self.unmarked, self.marked)


if __name__ == "__main__":
    db = DB()
    db.load(db.dir, db.known_plants_file)
    db.print_()
    print('-' * 60)
    db.known_plants = [[4.0, 3.0, 4.0]]
    db.marked = [[4.0, 3.0, 4.0]]
    db.unmarked = [[4.0, 3.0, 4.0]]
    db.print_()
    db.save(db.dir, db.known_plants_file)
