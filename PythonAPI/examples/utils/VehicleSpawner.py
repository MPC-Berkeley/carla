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
	def __init__(self,carla_world,auto_pilot=False, rowsToSpawn=[0,1,2,3],free_spots=0,auto_spawn=True):
		self.world = carla_world
		self.park_indices = np.array([])
		# Set up the indices for spawning
		for row in rowsToSpawn:
			self.park_indices = np.hstack([self.park_indices,np.arange(row*16+1,1+row*16+16)])
		

		# Removing free_spots from the spawning list
		for spot in range(free_spots):
			if (len(self.park_indices)) > 0:
				id_to_delete = random.randrange(len(self.park_indices))
				self.park_indices = np.delete(self.park_indices,id_to_delete)
			else:
				break

		if auto_spawn:
			self.spawn()

	# Method to set custom list
	def set_parking_indices(self,spawn_indices):
		self.parking_indices = spawn_indices

	# Method to spawn the vehicles from the parking_indices list
	def spawn():
		#TODO: Spawn vehicle code from spawn_npc.py
		print('Spawn vehicles')


	# Method to remove spawned vehicles
	def remove():
		#TODO: Destroy vehicles from spawn_npc.py
		print('Remove vehcles')



