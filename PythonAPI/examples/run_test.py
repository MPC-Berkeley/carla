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

# Rosbag record
import subprocess, shlex
import time

# ROS topic intention
import rospy
from std_msgs.msg import String

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

    rosbag_proc = None

    trail = args.trail

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        # client.load_world('exp')
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args.filter)
        controller = DualControl(world, args.autopilot)

        # # "Drone" camera view
        blueprint_library = world.world.get_blueprint_library()
        drone_camera_bp = blueprint_library.find('sensor.camera.rgb')
        # drone_camera_bp.set_attribute('image_size_x', str(args.width))
        # drone_camera_bp.set_attribute('image_size_y', str(args.height))
        drone_camera_bp.set_attribute('image_size_x', str(600))
        drone_camera_bp.set_attribute('image_size_y', str(800))
        drone_camera_bp.set_attribute('fov', '100')
        drone_camera_bp.set_attribute('sensor_tick', '0.1')
        drone_camera_transform = carla.Transform(carla.Location(x=285.0, y=-210.0, z=20.0), carla.Rotation(yaw=90.0, pitch=-90))
        drone_camera = world.world.spawn_actor(drone_camera_bp, drone_camera_transform)

        rospy.init_node('intention_node', anonymous=True)
        intention_pub = rospy.Publisher('intention', String, queue_size=10)

        np.random.seed(0)
        random.seed(0)

        spwnr = None

        for ep in range(10):
            print('trail: %d , Episode: %d' % (trail, ep))

            # ROSBAG RECORD
            command = "roslaunch carla_ros_bridge carla_ros_bridge.launch"
            command = shlex.split(command)              
            # roslaunch_proc = subprocess.Popen(command)
            #time.sleep(0.1)

            # ROSBAG RECORD
            command = "rosbag record -a -o bags/parking_p%d_t%d_e%d.bag __name:=carla_rosbag" % (args.s_id, trail, ep)
            command = shlex.split(command)
            rosbag_proc = None
            if args.record:
                rosbag_proc = subprocess.Popen(command)

                now = datetime.datetime.now()
                now = now.strftime("%Y-%m-%d-%H-%M-%S")
                # client.start_recorder("/home/carla/PythonAPI/examples/bags/parking_p%s_s%d_e%d_%s.log" % (args.s_id, trail, ep, now))
                client.start_recorder("parking_p%d_t%d_e%d_%s.log" % (args.s_id, trail, ep, now))

            

            spwnr = VehicleSpawner(client,False,[0,1,2,3],[0,4,4,0])
            clock = pygame.time.Clock()

            while True:
                clock.tick_busy_loop(60)

                controller_return = controller.parse_events(world, clock)

                if controller_return == 10:
                    print("Terminated by user")
                    raise KeyboardInterrupt

                if controller_return == 11:
                    # Next episode
                    break

                if controller_return == 6:
                    # publish intention
                    pub_str = "Intention Determined"
                    intention_pub.publish(pub_str)

                world.tick(clock)
                world.render(display)
                pygame.display.flip()
            
            # Stop the rosbag recording
            if rosbag_proc:
                command = "rosnode kill /carla_rosbag"
                command = shlex.split(command)
                subprocess.Popen(command)
                rosbag_proc.send_signal(subprocess.signal.SIGINT)


                client.stop_recorder()

            spwnr.remove()
            # roslaunch_proc.send_signal(subprocess.signal.SIGINT)
            world.restart()
            print('Done with ep loop')
                # world.destroy()

    except Exception as e:
        print('got an exception', e)

    finally:
        if rosbag_proc:
            command = "rosnode kill /carla_rosbag"
            command = shlex.split(command)
            subprocess.Popen(command)
            rosbag_proc.send_signal(subprocess.signal.SIGINT)

            client.stop_recorder()
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
        default='1920x1280',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.*',
        help='actor filter (default: "vehicle.*")')

    argparser.add_argument(
        '-i', '--s_id', 
        help="id of the subject",
        required=True,
        type=int)

    argparser.add_argument(
        '-t', '--trail', 
        help="trail number",
        required=True,
        type=int)

    argparser.add_argument(
        '-r', '--record', 
        help="Record rosbag and log",
        default=0,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    # Calibrate the steer wheel
    device = evdev.list_devices()[0]
    evtdev = InputDevice(device)
    val = 20000
    evtdev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)

    print(__doc__)

    try:
        game_loop(args)
        

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
