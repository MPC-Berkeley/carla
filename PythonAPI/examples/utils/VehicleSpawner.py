from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


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


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
import pdb
from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')



# ==============================================================================
# -- Vehicle spawner -----------------------------------------------------------
# ==============================================================================
class VehicleSpawner(object):
  def __init__(self,client=None,auto_pilot=False, rowsToSpawn=[0,1,2,3],free_spots=[0,0,0,0],auto_spawn=True):
    
    self.client = client
    self.auto_pilot = False

    # To keep track of spawned vehicles
    self.vehicles_list = []

    # Set up the indices for spawning
    self.park_indices = np.array([],dtype=int)
    for row,spots in zip(rowsToSpawn,free_spots):
      ids = np.arange(row*16+1,1+row*16+16)
      
      # Randomly remove spots from the list
      for spot in range(spots):
        if(len(ids) > 0):
          id_to_delete = random.randrange(len(ids))
          ids = np.delete(ids,id_to_delete)
        else:
          break
      # Stack the parking ids for the row
      self.park_indices = np.hstack([self.park_indices,ids])


    if auto_spawn:
      self.spawn()

  # Method to set custom list
  def set_parking_indices(self,spawn_indices):
    self.park_indices = spawn_indices
  
  # Method remove specific indices
  def remove_parking_indices(self,remove_spots):
    indx = np.ravel([np.where(self.park_indices == i) for i in remove_spots])
    self.park_indices = np.delete(self.park_indices,indx)

  # Method to spawn the vehicles from the parking_indices list
  def spawn(self):
    # Try spawning
    if self.client is not None:
      # Get world and blueprints of the vehicles - only 4 wheelers
      world = self.client.get_world()
      blueprints = world.get_blueprint_library().filter('vehicle.*')
      blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
      SpawnActor = carla.command.SpawnActor
      SetAutopilot = carla.command.SetAutopilot
      FutureActor = carla.command.FutureActor

      # Loop through parking spots
      batch = []
      spawn_points = world.get_map().get_spawn_points()
      for spot in self.park_indices:
        # Select random blueprint
        blueprint = random.choice(blueprints)
        # Randomize color
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Randomize id
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)

        # Disable autopilot
        blueprint.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(blueprint, spawn_points[spot]).then(SetAutopilot(FutureActor, self.auto_pilot)))
      
      # Try synchronous application I assume
      for response in self.client.apply_batch_sync(batch):
        if response.error:
          logging.error(response.error)
        else:
          self.vehicles_list.append(response.actor_id)
   
  # Method to remove spawned vehicles
  def remove(self):
    print('\ndestroying %d vehicles' % len(self.vehicles_list))
    self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
    self.vehicles_list = []


