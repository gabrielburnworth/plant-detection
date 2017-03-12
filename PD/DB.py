#!/usr/bin/env python
"""DB for Plant Detection.

For Plant Detection.
"""
import os
import json
import base64
import requests
import numpy as np
from PD import CeleryPy
from PD import ENV


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
        self.test_coordinates = [600, 400, 0]
        self.coordinates = None

        # API requests setup
        try:
            api_token = os.environ['API_TOKEN']
        except KeyError:
            api_token = 'x.eyJpc3MiOiAiLy9zdGFnaW5nLmZhcm1ib3QuaW86NDQzIn0.x'
        try:
            encoded_payload = api_token.split('.')[1]
            encoded_payload += '=' * (4 - len(encoded_payload) % 4)
            json_payload = base64.b64decode(encoded_payload).decode('utf-8')
            server = json.loads(json_payload)['iss']
        except:  # noqa pylint:disable=W0702
            server = '//my.farmbot.io'
        self.api_url = 'https:' + server + '/api/'
        self.headers = {'Authorization': 'Bearer {}'.format(api_token),
                        'content-type': "application/json"}
        self.errors = {}

    def api_get(self, endpoint):
        """GET from an API endpoint."""
        response = requests.get(self.api_url + endpoint, headers=self.headers)
        self.api_response_error_collector(response)
        self.api_response_error_printer()
        return response

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

    def _download_image_from_url(self, img_filename, url):
        response = requests.get(url, stream=True)
        self.api_response_error_collector(response)
        self.api_response_error_printer()
        if response.status_code == 200:
            with open(img_filename, 'wb') as img_file:
                for chunk in response:
                    img_file.write(chunk)

    def get_image(self, image_id):
        """Download an image from the FarmBot Web App API."""
        response = self.api_get('images/' + str(image_id))
        if response.status_code == 200:
            image_json = response.json()
            image_url = image_json['attachment_url']
            try:
                testfilename = self.dir + 'test_write.try_to_write'
                testfile = open(testfilename, "w")
                testfile.close()
                os.remove(testfilename)
            except IOError:
                directory = '/tmp/'
            else:
                directory = self.dir
            image_filename = directory + str(image_id) + '.jpg'
            self._download_image_from_url(image_filename, image_url)
            self.coordinates = list([int(image_json['meta']['x']),
                                     int(image_json['meta']['y']),
                                     int(image_json['meta']['z'])])
            return image_filename
        else:
            return None

    def getcoordinates(self, test_coordinates=False):
        """Get machine coordinates from bot."""
        location = ENV.redis_load('location')
        if location is None or test_coordinates:
            self.coordinates = self.test_coordinates  # testing coordintes
        else:
            self.coordinates = location  # current bot coordinates

    def save_plants(self):
        """Save plant detection plants to file.

        'known', 'remove', 'safe_remove', and 'save'
        """
        if self.tmp_dir is None:
            json_dir = self.dir
        else:
            json_dir = self.tmp_dir
        try:
            with open(json_dir + self.plants_file, 'w') as plant_file:
                json.dump(self.plants, plant_file)
        except IOError:
            self.tmp_dir = "/tmp/"
            self.save_plants()

    def load_plants_from_file(self):
        """Load plants from file."""
        try:
            with open(self.dir + self.plants_file, 'r') as plant_file:
                self.plants = json.load(plant_file)
        except IOError:
            pass

    def load_plants_from_web_app(self):
        """Download known plants from the FarmBot Web App API."""
        response = self.api_get('plants')
        app_plants = response.json()
        if response.status_code == 200:
            plants = []
            for plant in app_plants:
                plants.append({
                    'x': plant['x'],
                    'y': plant['y'],
                    'radius': plant['radius']})
            self.plants['known'] = plants

    def identify_plant(self, plant_x, plant_y, known):
        """Identify a provided plant based on its location.

        Args:
            known: [x, y, r] array of known plants
            plant_x, plant_y: x and y coordinates of plant to identify
        Coordinate is:
            within a known plant area: a plant to 'save' (it's the known plant)
            within a known plant safe zone: a 'safe_remove' weed
            outside a known plant area or safe zone: a 'remove' weed
        """
        cxs, cys, crs = known[:, 0], known[:, 1], known[:, 2]
        if all((plant_x - cx)**2 + (plant_y - cy)**2
               > (cr + self.weeder_destrut_r)**2
               for cx, cy, cr in zip(cxs, cys, crs)):
            # Plant is outside of known plant safe zone
            return 'remove'
        elif all((plant_x - cx)**2 + (plant_y - cy)**2 > cr**2
                 for cx, cy, cr in zip(cxs, cys, crs)):
            # Plant is inside known plant safe zone
            return 'safe_remove'
        else:  # Plant is within known plant area
            return 'save'

    def identify(self, second_pass=False):
        """Compare detected plants to known to separate plants from weeds."""
        if not second_pass:
            self.plants['remove'] = []
            self.plants['save'] = []
            self.plants['safe_remove'] = []
        if self.plants['known'] is None or self.plants['known'] == []:
            self.plants['known'] = [{'x': 0, 'y': 0, 'radius': 0}]
        kplants = np.array(
            [[_['x'], _['y'], _['radius']] for _ in self.plants['known']])
        for plant_coord in self.coordinate_locations:
            plant_x = round(plant_coord[0], 2)
            plant_y = round(plant_coord[1], 2)
            plant_r = round(plant_coord[2], 2)
            plant_is = self.identify_plant(plant_x, plant_y, kplants)
            if plant_is == 'remove':
                self.plants['remove'].append(
                    {'x': plant_x, 'y': plant_y, 'radius': plant_r})
            elif plant_is == 'safe_remove' and not second_pass:
                self.plants['safe_remove'].append(
                    {'x': plant_x, 'y': plant_y, 'radius': plant_r})
            else:
                if not second_pass:
                    self.plants['save'].append(
                        {'x': plant_x, 'y': plant_y, 'radius': plant_r})
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
        def _identified_plant_text_output(title, action, plants):
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
        _identified_plant_text_output(
            title='known plants inputted',
            action='are to be saved',
            plants='known')
        # Print removal candidates
        _identified_plant_text_output(
            title='plants marked for removal',
            action='are to be removed',
            plants='remove')
        # Print safe_remove plants
        _identified_plant_text_output(
            title='plants marked for safe removal',
            action='were too close to the known plant to remove completely',
            plants='safe_remove')
        # Print saved
        _identified_plant_text_output(
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
        unsent_cs = []
        # Encode to CS
        for mark in self.plants['remove']:
            plant_x, plant_y = round(mark['x'], 2), round(mark['y'], 2)
            plant_r = round(mark['radius'], 2)
            unsent = CeleryPy.add_point(plant_x, plant_y, 0, plant_r)
            unsent_cs.append(unsent)
        return unsent_cs

    def upload_point(self, point, color, id_list):
        """Upload a point to the FarmBot Web App."""
        # payload
        plant_x, plant_y = round(point['x'], 2), round(point['y'], 2)
        plant_r = round(point['radius'], 2)
        plant = {'x': str(plant_x), 'y': str(plant_y), 'z': 0,
                 'radius': str(plant_r),
                 'meta': {'created_by': 'plant-detection',
                          'color': color}}
        payload = json.dumps(plant)
        # API Request
        response = requests.post(self.api_url + 'points',
                                 data=payload, headers=self.headers)
        point_id = None
        if response.status_code == 200:
            point_id = response.json()['id']
            id_list.append(point_id)
        self.api_response_error_collector(response)
        return id_list

    def upload_plants(self):
        """Add plants to FarmBot Web App Farm Designer."""
        point_ids = []
        for plant in self.plants['remove']:
            point_ids = self.upload_point(plant, 'red', point_ids)
        for plant in self.plants['save']:
            point_ids = self.upload_point(plant, 'blue', point_ids)
        for plant in self.plants['known']:
            point_ids = self.upload_point(plant, 'green', point_ids)
        for plant in self.plants['safe_remove']:
            point_ids = self.upload_point(plant, 'cyan', point_ids)
        self.api_response_error_printer()
        if point_ids:
            # Points have been added to the web app
            # Indicate that a sync is required for the points
            CeleryPy.data_update('points', point_ids)
