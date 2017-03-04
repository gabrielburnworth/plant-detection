#!/usr/bin/env python
"""DB for Plant Detection.

For Plant Detection.
"""
import os
import json
import requests
import numpy as np
try:
    from .CeleryPy import CeleryPy
except:
    from CeleryPy import CeleryPy


class DB(object):
    """Known and detected plant data for Plant Detection."""

    def __init__(self):
        """Set initial attributes."""
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

        # API requests setup
        try:
            api_token = os.environ['API_TOKEN']
        except KeyError:
            api_token = 'none'
        self.headers = {'Authorization': 'Bearer {}'.format(api_token),
                        'content-type': "application/json"}
        self.errors = {}

    def api_response_error_collector(self, response):
        """Catch and log errors from API requests."""
        self.errors = {}  # reset
        if response.status_code != 200:
            try:
                self.errors[str(response.status_code)] += 1
            except KeyError:
                self.errors[str(response.status_code)] = 1

    def api_response_error_printer(self):
        """Print API response error output."""
        error_string = ''
        for key, value in self.errors.items():
            error_string += '{} {} errors '.format(value, key)
        print(error_string)

    def save_plants(self):
        """Save plant detection plants to file.

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
        """Load plants from file."""
        try:
            with open(self.dir + self.plants_file, 'r') as f:
                self.plants = json.load(f)
        except IOError:
            pass

    def load_known_plants_from_env_var(self):
        """Load known plant inputs 'x' 'y' and 'radius' from json."""
        db_json = json.loads(os.environ['DB'])
        self.plants['known'] = db_json['plants']
        # Remove unnecessary data, if it exists
        for plant in self.plants['known']:
            plant.pop('name', None)
            plant.pop('device_id', None)

    def load_plants_from_web_app(self):
        """Download known plants from the FarmBot Web App API."""
        response = requests.get('https://staging.farmbot.io/api/plants',
                                headers=self.headers)
        self.api_response_error_collector(response)
        self.api_response_error_printer()
        plants = response.json()
        if response.status_code == 200:
            self.plants['known'] = plants

    def identify(self):
        """Compare detected plants to known to separate plants from weeds."""
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
        """Output text indicating the number of plants/objects detected."""
        if calibration:
            object_name = 'calibration objects'
        else:
            object_name = 'plants'
        print("{} {} detected in image.".format(self.object_count,
                                                object_name))

    def print_identified(self):
        """Output text including data about identified detected plants."""
        def identified_plant_text_output(title, action, plants):
            print("\n{} {}.".format(
                len(self.plants[plants]), title))
            if len(self.plants[plants]) > 0:
                print("Plants at the following machine coordinates "
                      "( X Y ) with R = radius {}:".format(action))
            for plant in self.plants[plants]:
                print("    ( {x:5.0f} {y:5.0f} ) R = {r:.0f}".format(
                    x=plant['x'],
                    y=plant['y'],
                    r=plant['radius']))

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
        """Output coordinate data for detected (but not identified) plants."""
        if len(self.coordinate_locations) > 0:
            print("Detected object machine coordinates "
                  "( X Y ) with R = radius:")
            for coordinate_location in self.coordinate_locations:
                print("    ( {:5.0f} {:5.0f} ) R = {:.0f}".format(
                    coordinate_location[0],
                    coordinate_location[1],
                    coordinate_location[2]))

    def print_pixel(self):
        """Output text pixel data for detected (but not identified) plants."""
        if len(self.pixel_locations) > 0:
            print("Detected object center pixel locations ( X Y ):")
            for pixel_location in self.pixel_locations:
                print("    ( {:5.0f}px {:5.0f}px )".format(pixel_location[0],
                                                           pixel_location[1]))

    def output_celery_script(self):
        """Output JSON with identified plant coordinates and radii."""
        # Encode to CS
        farmbot = CeleryPy()
        for mark in self.plants['remove']:
            x, y = round(mark['x'], 2), round(mark['y'], 2)
            r = round(mark['radius'], 2)
            farmbot.add_point(x, y, 0, r)

    def upload_weeds(self):
        """Add plants marked for removal to FarmBot Web App Farm Designer."""
        for mark in self.plants['remove']:
            # payload
            x, y = round(mark['x'], 2), round(mark['y'], 2)
            r = round(mark['radius'], 2)
            plant = {'x': str(x), 'y': str(y),
                     'radius': str(r)}
            payload = json.dumps(plant)
            # API Request
            response = requests.post('https://staging.farmbot.io/api/points',
                                     data=payload, headers=self.headers)
            self.api_response_error_collector(response)
        self.api_response_error_printer()

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
