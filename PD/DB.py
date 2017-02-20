#!/usr/bin/env python
"""DB for Plant Detection.

For Plant Detection.
"""
import os
import json
import numpy as np
try:
    from .CeleryPy import CeleryPy
except:
    from CeleryPy import CeleryPy


class DB():
    """Known and detected plant data for Plant Detection"""

    def __init__(self):
        self.plants = {'known': [], 'save': [],
                       'remove': [], 'safe_remove': []}
        self.output_text = True
        self.output_json = False
        self.object_count = None
        self.pixel_locations = []
        self.coordinate_locations = []
        self.calibration_pixel_locations = []
        self.dir = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep
        self.plants_file = "plant-detection_plants.json"
        self.tmp_dir = None
        self.weeder_destrut_r = 50

    def save_plants(self):
        """Save plant detection plants to file:
                'known', 'remove', 'safe_remove', and 'save'
        """
        if self.tmp_dir is None:
            json_dir = self.dir
        else:
            json_dir = self.tmp_dir
        try:
            with open(json_dir + self.plants_file, 'w') as f:
                json.dump(self.plants, f)
        except IOError:
            self.tmp_dir = "/tmp/"
            self.save_plants()

    def load_plants_from_file(self):
        """Load plants from file"""
        try:
            with open(self.dir + self.plants_file, 'r') as f:
                self.plants = json.load(f)
        except IOError:
            pass

    def load_known_plants_from_env_var(self):
        """Load known plant inputs 'x' 'y' and 'radius' from json"""
        db_json = json.loads(os.environ['DB'])
        self.plants['known'] = db_json['plants']
        # Remove unnecessary data, if it exists
        for plant in self.plants['known']:
            plant.pop('name', None)
            plant.pop('device_id', None)

    def identify(self):
        """Compare detected plants to known to separate plants from weeds"""
        self.plants['remove'] = []
        self.plants['save'] = []
        self.plants['safe_remove'] = []
        if self.plants['known'] is None or self.plants['known'] == []:
            self.plants['known'] = [{'x': 0, 'y': 0, 'radius': 0}]
        kplants = np.array(
            [[_['x'], _['y'], _['radius']] for _ in self.plants['known']])
        for plant_coord in self.coordinate_locations:
            x, y, r = plant_coord[0], plant_coord[1], plant_coord[2]
            x, y, r = round(x, 2), round(y, 2), round(r, 2)
            cxs, cys, crs = kplants[:, 0], kplants[:, 1], kplants[:, 2]
            if all((x - cx)**2 + (y - cy)**2 > (cr + self.weeder_destrut_r)**2
                   for cx, cy, cr in zip(cxs, cys, crs)):
                # Detected plant is outside of known plant safe zone
                self.plants['remove'].append({'x': x, 'y': y, 'radius': r})
            elif all((x - cx)**2 + (y - cy)**2 > cr**2
                     for cx, cy, cr in zip(cxs, cys, crs)):
                # Detected plant is inside known plant safe zone
                self.plants['safe_remove'].append(
                    {'x': x, 'y': y, 'radius': r})
            else:  # Detected plant is within known plant area
                self.plants['save'].append({'x': x, 'y': y, 'radius': r})
        if self.plants['known'] == [{'x': 0, 'y': 0, 'radius': 0}]:
            self.plants['known'] = []

    def print_count(self, calibration=False):
        """output text indicating the number of plants/objects detected"""
        if calibration:
            object_name = 'calibration objects'
        else:
            object_name = 'plants'
        print("{} {} detected in image.".format(self.object_count,
                                                object_name))

    def print_identified(self):
        """output text including data about identified detected plants"""
        def identified_plant_text_output(title, action, plants):
            print("\n{} {}.".format(
                len(self.plants[plants]), title))
            if len(self.plants[plants]) > 0:
                print("Plants at the following machine coordinates "
                      "( X Y ) with R = radius {}:".format(action))
            for known_plant in self.plants[plants]:
                print("    ( {x:5.0f} {y:5.0f} ) R = {r:.0f}".format(
                    x=known_plant['x'],
                    y=known_plant['y'],
                    r=known_plant['radius']))

        # Print known
        identified_plant_text_output(
            title='known plants inputted',
            action='are to be saved',
            plants='known')
        # Print removal candidates
        identified_plant_text_output(
            title='plants marked for removal',
            action='are to be removed',
            plants='remove')
        # Print safe_remove plants
        identified_plant_text_output(
            title='plants marked for safe removal',
            action='were too close to the known plant to remove completely',
            plants='safe_remove')
        # Print saved
        identified_plant_text_output(
            title='detected plants are known or have escaped removal',
            action='have been saved',
            plants='save')

    def print_coordinates(self):
        """output text data (coordinates) about
           detected (but not identified) plants"""
        if len(self.coordinate_locations) > 0:
            print("Detected object machine coordinates ( X Y ) with R = radius:")
            for coordinate_location in self.coordinate_locations:
                print("    ( {:5.0f} {:5.0f} ) R = {:.0f}".format(
                    coordinate_location[0],
                    coordinate_location[1],
                    coordinate_location[2]))

    def print_pixel(self):
        """output text data (pixels) about
           detected (but not identified) plants"""
        if len(self.pixel_locations) > 0:
            print("Detected object center pixel locations ( X Y ):")
            for pixel_location in self.pixel_locations:
                print("    ( {:5.0f}px {:5.0f}px )".format(pixel_location[0],
                                                           pixel_location[1]))

    def output_CS(self):
        """output JSON with identified plant coordinates and radii"""
        # Encode to CS
        FarmBot = CeleryPy()
        for mark in self.plants['remove']:
            x, y = round(mark['x'], 2), round(mark['y'], 2)
            r = round(mark['radius'], 2)
            FarmBot.add_point(x, y, 0, r)
        for unmark in self.plants['save']:
            x, y = round(unmark['x'], 2), round(unmark['y'], 2)
            r = round(unmark['radius'], 2)
            # FarmBot.add_plant(0, [x, y, 0], r)

        # Save plant coordinates to file
        self.save_plants()


if __name__ == "__main__":
    db = DB()
    db.load_plants_from_file()
    db.print_identified()
    print('-' * 60)
    db.plants['known'] = [{'x': 3.0, 'y': 4.0, 'radius': 5.0}]
    db.plants['save'] = [{'x': 6.0, 'y': 7.0, 'radius': 8.0}]
    db.plants['remove'] = [{'x': 9.0, 'y': 10.0, 'radius': 11.0}]
    db.plants['safe_remove'] = []
    db.print_identified()
    db.save_plants()
