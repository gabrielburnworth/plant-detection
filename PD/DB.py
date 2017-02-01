#!/usr/bin/env python
"""Parameters for Plant Detection.

For Plant Detection.
"""
import sys, os

class DB():
    def __init__(self, **kwargs):
        self.output_text = True
        self.output_json = False
        self.known_plants = None
        self.marked = None
        self.unmarked = None
        self.dir = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep
        self.known_plants_file = "plant-detection_known-plants.txt"
        self.filename = self.dir + self.known_plants_file

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write('X Y Radius\n')
            if self.known_plants is not None:
                for plant in self.known_plants:
                    f.write('{} {} {}\n'.format(*plant))

    def save_detected_plants(save, remove):
        np.savetxt(self.dir + "detected-plants_saved.csv", save,
                   fmt='%.1f', delimiter=',', header='X,Y,Radius')
        np.savetxt(self.dir + "detected-plants_to-remove.csv", remove,
                   fmt='%.1f', delimiter=',', header='X,Y,Radius')

    def load(self, filename):
        try:  # Load input parameters from file
            with open(self.known_plants_file, 'r') as f:
                lines = f.readlines()
                known_plants = []
                for line in lines[1:]:
                    line = line.strip().split(' ')
                    known_plants.append([float(line[0]),
                                         float(line[1]),
                                         float(line[2])])
                if len(known_plants) > 0:
                    self.known_plants = known_plants
        except IOError:  # Use defaults and save to file
            self.save(filename)

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
    db.load(db.filename)
    db.print_()
    print('-' * 60)
    db.known_plants = [[4.0, 3.0, 4.0]]
    db.marked = [[4.0, 3.0, 4.0]]
    db.unmarked = [[4.0, 3.0, 4.0]]
    db.print_()
    db.save(db.filename)
