#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control with steering wheel Logitech G27.

To drive start by preshing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
from utils.VehicleSpawner import VehicleSpawner
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
import numpy as np
import random

# Calibrate the resistence of wheeel
import evdev
from evdev import ecodes, InputDevice

from utils.carla_utils import *
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
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    drone_camera = None

    try:
        for scen in range(2):
            np.random.seed(1)
            random.seed(1)
            
            client = carla.Client(args.host, args.port)
            client.set_timeout(2.0)
            display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

            hud = HUD(args.width, args.height)
            world = World(client.get_world(), hud, args.filter)
            controller = DualControl(world, args.autopilot)

            # # "Drone" camera view
            blueprint_library = world.world.get_blueprint_library()
            drone_camera_bp = blueprint_library.find('sensor.camera.rgb')
            drone_camera_bp.set_attribute('image_size_x', str(args.width))
            drone_camera_bp.set_attribute('image_size_x', str(args.height))
            drone_camera_bp.set_attribute('fov', '120')
            drone_camera_bp.set_attribute('sensor_tick', '0.05')
            drone_camera_transform = carla.Transform(carla.Location(x=285.0, y=-210.0, z=20.0), carla.Rotation(yaw=90.0, pitch=-90))
            drone_camera = world.world.spawn_actor(drone_camera_bp, drone_camera_transform)

            for ep in range(3):
                print('Episode: ', ep)
                

                spwnr = VehicleSpawner(client,False,[0,1,2,3],[0,4,4,0])
                clock = pygame.time.Clock()

                while True:
                    clock.tick_busy_loop(60)
                    if controller.parse_events(world, clock):
                        break
                    world.tick(clock)
                    world.render(display)
                    pygame.display.flip()

                spwnr.remove()
                world.restart()
                # world.destroy()
    except Exception as e:
        print(e)

    finally:
        if drone_camera:
            drone_camera.destroy()
        if world is not None:
            spwnr.remove()
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.*',
        help='actor filter (default: "vehicle.*")')

    argparser.add_argument(
        '-t', '--test', 
        help="test drive mode",
        action="store_true")

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    # Calibrate the steer wheel
    device = evdev.list_devices()[0]
    evtdev = InputDevice(device)
    val = 15000
    evtdev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)

    print(__doc__)

    try:
        game_loop(args)
        

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
