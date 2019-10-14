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

class VehicleEntry(object):
    """ A simple struct of vehicle actor properties."""

    def __init__(self, actor_id, actor_name, position, orientation, velocity, angular_velocity, acceleration):
        # TODO: finalize the entries needed (e.g. maybe bounding box).
        self.id = actor_id
        self.name = actor_name
        # Assuming the below fields are 3D vectors (represented as lists) in the map frame.
        self.position = position
        self.orientation = orientation # (roll, pitch, yaw)
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.acceleration = acceleration

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
                im.save_to_disk(savename)
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

if __name__ == '__main__':
    test_data_logger()
