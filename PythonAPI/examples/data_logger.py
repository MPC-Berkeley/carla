#!/usr/bin/env python

"""
    Data Logging functionality with incremental writing to file.
    See test_data_logger for an example of usage of the main DataLogger class.
"""

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import numpy as np 
import time
import threading
from PIL import Image
from collections import deque # thread-safe queue/stack used as a ring buffer
# import json 
# import pickle
import csv 
import copy

# Helper utility functions for data extraction.
def extract_xyz(xyz_type):
    return xyz_type.x, xyz_type.y, xyz_type.z

def extract_rpy(rpy_type):
    return rpy_type.roll, rpy_type.pitch, rpy_type.yaw

# Class implementation: VehicleEntry, Measurement, and DataLogger.
class VehicleEntry(object):
    """ A simple struct of vehicle actor properties."""

    def __init__(self, actor_id, actor_name, position, orientation, velocity, angular_velocity, acceleration, \
                 throttle=np.nan, brake=np.nan, steer_angle=np.nan):
        # TODO: finalize the entries needed (e.g. maybe bounding box).
        self.id = actor_id
        self.name = actor_name
        # Assuming the below fields are 3D vectors (represented as lists) in the map frame.
        self.position = position
        self.orientation = orientation # (roll, pitch, yaw)
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.acceleration = acceleration
        self.throttle = throttle
        self.brake = brake   
        self.steer_angle = steer_angle

class Measurement(object):
    """ A single timestamped measurement containing images and a list of VehicleEntry objects."""

    def __init__(self, episode_no, frame_no, timestamp, image_list, image_name_list, vehicle_list):
        self._episode_no = episode_no
        self._frame_no = frame_no
        self._image_name_list = image_name_list
        self._image_list = image_list

        self._measurement_dict = {}
        self._measurement_dict['frame'] = frame_no
        self._measurement_dict['time'] = timestamp
        self._measurement_dict['vehicles'] = [vehicle.__dict__ for vehicle in vehicle_list]
        
    def save(self, rootdir):
        """ Saves contained images and vehicle entries to a specified rootdir. """

        for name, img in zip(self._image_name_list, self._image_list):
            savename = '%s/%s_%08d.png' % (rootdir, name, self._frame_no)
            if isinstance(img, carla.Image):
                img.save_to_disk(savename)
            elif isinstance(img, np.ndarray):
                im_pil = Image.fromarray(img)
                im_pil.save(savename)
            else:
                raise ValueError('Image format: ', type(img), ' not handled currently!')

        # TODO: finalize a format (e.g. mat, hdf5, pickle, etc.):
        filename = '%s/%s.csv' % (rootdir, self._episode_no)
        with open(filename, 'a') as f:
            wr = csv.writer(f)
            for vehicle in self._measurement_dict['vehicles']:
                wr.writerow([self._measurement_dict['frame'], \
                             self._measurement_dict['time'], \
                             vehicle
                             ])
            f.write('\n')
        # with open(filename, 'a') as f:
        #     json_str = json.dumps(self._measurement_dict, indent=2)
        #     f.write('[' + json_str + '],\n')
        # with open('%s/%s.pkl' % (rootdir, self._episode_no), 'ab') as f:
        #     pickle.dump(self._measurement_dict, f)

class DataLogger(object):
    """ A logging class that uses a deque and saving thread to incrementally save measurements. """

    def __init__(self, savedir, maxlen=None, max_save_fps=10):
        # deque used to implement a FIFO queue
        # if maxlen is specified, will drop oldest entries to add new entries (ringbuffer)
        # else the FIFO queue can grow unbounded if push rate > pop rate (TODO: check this)

        os.makedirs(savedir, exist_ok=True)
        self._savedir = savedir
        self._fifo_queue = deque(maxlen=maxlen)
        self._is_running = False
        self._max_save_fps = int(max_save_fps) if max_save_fps > 0 else 10 
    
    def start(self):
        """ Start the saver thread.  Must run before filling in data. """

        self._thread = threading.Thread(target=self._saving_thread_fncn)
        self._thread.start()
        

    def update(self, entry):
        """ Adds a new data entry to the FIFO queue. """

        if not self._is_running:
            raise RuntimeError("The saving thread should be started first.")

        self._fifo_queue.append(entry)

    def stop(self, finish_writing=True):
        """ Stop the saver thread.

            If finish_writing is True, then the queue will be emptied out first.
        """

        if finish_writing:
            while self._is_running and len(self._fifo_queue) > 0:
                time.sleep(0.1) # maybe a more elegant way to do this    

        self._is_running = False
        self._thread.join()

    def _saving_thread_fncn(self):
        prev_thread_time = 0
        min_save_period = 1.0 / float(self._max_save_fps)
        self._is_running = True
        while self._is_running:
            current_time = time.time()
            if(current_time - prev_thread_time) < min_save_period:
                # throttling the thread rate
                time.sleep(current_time - prev_thread_time)

            try:
                entry_to_write = self._fifo_queue.popleft()
                entry_to_write.save(self._savedir) 
            except IndexError:
                pass
            except Exception as e:
                # For example, no save method provided.
                self._is_running = False
                print('Error in save thread:', e)

            prev_thread_time = current_time

# Test main functions: one with fake data and one with a Carla client.
def test_data_logger():
    """ Test functionality of main DataLogger class.  Also serves as an example of usage. """
    im_rgb = np.asarray(Image.open('_out/00011156.png')) # change to whatever dummy images you have
    im_seg = np.asarray(Image.open('_out/00014366.png'))
    dl = DataLogger('test_out', max_save_fps=10)
    dl.start()
    
    ego_vehicle_1 = VehicleEntry(0, 'ego', [5.0, 10.0, 15.0], [0.0, 0.0, 90.0], \
                                               [10.0, 0.0, 0.0],  [0.0, 0.0, 0.0], \
                                               [1.0, 0.0, 0.0])
    lead_vehicle_1 = VehicleEntry(1, 'lead', [30.0, 10.0, 15.0], [0.0, 0.0, 91.0], \
                                                 [15.0, 0.0, 0.0],  [0.0, 0.0, 0.0], \
                                                 [0.0, 0.0, 0.0])
    data_entry_1 = Measurement(0, 0, time.time(), [im_rgb, im_seg], ['rgb', 'seg'], \
                             [ego_vehicle_1, lead_vehicle_1])
    dl.update(data_entry_1)

    ego_vehicle_2 = VehicleEntry(0, 'ego', [-5.0, -10.0, -15.0], [0.0, 0.0, -90.0], \
                                               [-10.0, 0.0, 0.0],  [0.0, 0.0, 0.0], \
                                               [-1.0, 0.0, 0.0])
    lead_vehicle_2 = VehicleEntry(1, 'lead', [-30.0, -10.0, -15.0], [0.0, 0.0, -91.0], \
                                                [-15.0, 0.0, 0.0],  [0.0, 0.0, 0.0], \
                                                [0.0, 0.0, 0.0])
    data_entry_2 = Measurement(0, 1, time.time(), [im_rgb, im_seg], ['rgb', 'seg'], \
                             [ego_vehicle_2, lead_vehicle_2])
    dl.update(data_entry_2)

    dl.stop()

def test_carla_logging():
    # Assumes that you have already
    # (1) Started the Carla server (bash ...CarlaUE4.sh)
    # (2) spawned the relevant vehicle actors.
    #     e.g. python spawn_npc.py -n 10 -w 0; python manual_control_steeringwheel.py
    
    # Callback used for Carla logging example only.
    current_image = None
    image_lock = threading.Lock()
    def image_callback(image):
        with image_lock:
            current_image = image

    try:
        vehicle_actor_ids = []
        vehicle_actor_names = []

        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()

        vehicle_actor_list = world.get_actors().filter('vehicle.*')
        for actor in vehicle_actor_list:
            vehicle_actor_ids.append(actor.id)
            vehicle_actor_names.append('%s_%s' % (actor.type_id, actor.attributes['role_name']))

        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_x', '1080')
        camera_bp.set_attribute('fov', '110')
        camera_bp.set_attribute('sensor_tick', '0.05')
        camera_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=10.0))
        camera = world.spawn_actor(camera_bp, camera_transform)
        
        cc = carla.ColorConverter.Raw
        #camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame, cc))
        camera.listen(image_callback)

        dl = DataLogger('test_carla_log', max_save_fps=10)
        dl.start()

        episode_id = 0
        for frame_no in range(100):
            snapshot = world.get_snapshot()
            vehicle_entries = []
            for veh_id, veh_name in zip(vehicle_actor_ids, vehicle_actor_names):
                try:
                    vehicle = snapshot.find(veh_id)
                    transform = vehicle.get_transform()
                    x, y, z = extract_xyz(transform.location)
                    roll, pitch, yaw = extract_rpy(transform.rotation)
                    vx, vy, vz = extract_xyz(vehicle.get_velocity())
                    wx, wy, wz = extract_xyz(vehicle.get_angular_velocity())
                    ax, ay, az = extract_xyz(vehicle.get_acceleration())
                    vehicle_entries.append(
                        VehicleEntry(veh_id, veh_name, \
                                    [x,y,z], \
                                    [roll, pitch, yaw], \
                                    [vx, vy, vz], \
                                    [wx, wy, wz], \
                                    [ax, ay, az])
                    )
                except Exception as e:
                    print(e)
                    
            latest_image = None
            with image_lock:
                # TODO: May be a better way to handle this extracting of the global image variable.
                # If we ensure current_image is a numpy array, can use np.copy instead.
                latest_image = current_image

            if latest_image is None:
                print('No image yet!')
                time.sleep(0.1)
                continue

            measurement = Measurement(episode_id, frame_no, snapshot.timestamp.platform_timestamp, [latest_image], ['spectator'], \
                vehicle_entries)
            dl.update(measurement)

            time.sleep(0.1)

        dl.stop()
        camera.destroy()
    except:
        if 'camera' in vars():
            camera.destroy()

if __name__ == '__main__':
    # Can choose which test to run here, or add argparse later.
    #test_data_logger()
    test_carla_logging()
