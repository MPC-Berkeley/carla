#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# Modified by Vijay Govindarajan to collect time-synced image data.

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

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

import matplotlib.pyplot as plt
import numpy as np 
import pdb
from tqdm import tqdm
import argparse
from collections import deque
import threading
from PIL import Image
import time

from synchronous_mode import CarlaSyncMode, draw_image, get_font, should_quit
from data_logger import CarlaImageSaver

# run spawn_npc.py externally
def run_autopilot_with_npc(args, camera_config, max_frames=1000):
    actor_list = []
    pygame.init()

    # display = pygame.display.set_mode(
    #     (args.width, args.height),
    #     pygame.HWSURFACE | pygame.DOUBLEBUF)
    # font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)

    world = client.get_world()

    try:        
        m = world.get_map()

        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        bp_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(bp_library.filter(args.filter)),
            start_pose)
        vehicle.set_autopilot(True)
        ego_id = vehicle.id
        actor_list.append(vehicle)


        # Get RGB, Depth, and Semantic Segmentation Camera with camera_config params.
        sensor_location = carla.Location(x=camera_config['x'], y=camera_config['y'],
                                         z=camera_config['z'])
        sensor_rotation = carla.Rotation(pitch=camera_config['pitch'],
                                         roll=camera_config['roll'],
                                         yaw=camera_config['yaw'])
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        bp_rgb = bp_library.find('sensor.camera.rgb')
        bp_depth = bp_library.find('sensor.camera.depth')
        bp_seg = bp_library.find('sensor.camera.semantic_segmentation')

        for bp in [bp_rgb, bp_depth, bp_seg]:
            bp.set_attribute('image_size_x', str(camera_config['width']))
            bp.set_attribute('image_size_y', str(camera_config['height']))
            bp.set_attribute('fov', str(camera_config['fov']))


        camera_rgb = world.spawn_actor(bp_rgb, sensor_transform, attach_to=vehicle)
        actor_list.append(camera_rgb)
        camera_depth = world.spawn_actor(bp_depth, sensor_transform, attach_to=vehicle)
        actor_list.append(camera_depth)
        camera_seg = world.spawn_actor(bp_seg, sensor_transform, attach_to=vehicle)
        actor_list.append(camera_seg)

        # num_successful = 0
        # for i in range(num_npc_vehicles):
        #     npc = world.try_spawn_actor(bp, random.choice(m.get_spawn_points()))
        #     if npc is not None:
        #         actor_list.append(npc)
        #         npc.set_autopilot()
        #         num_successful += 1
        # print('num_successful npc: ', num_successful)

        # rgb_saver   = CarlaImageSaver('%s/rgb' % args.logdir)
        # depth_saver = CarlaImageSaver('%s/depth' % args.logdir) 
        # seg_saver   = CarlaImageSaver('%s/seg' % args.logdir)

        # Create a synchronous mode context.
        num_frames_saved = 0
        with CarlaSyncMode(world, camera_rgb, camera_depth, camera_seg, fps=args.fps) as sync_mode:
            # for saver in [rgb_saver, depth_saver, seg_saver]:
            #     saver.start()

            while num_frames_saved < max_frames:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_depth, image_semseg = sync_mode.tick(timeout=2.0)

                # # Choose the next waypoint and update the car location.
                # waypoint = random.choice(waypoint.next(1.5))
                # vehicle.set_transform(waypoint.transform)
                # curr_waypoint_ind += 1
                # waypoint = waypoint_list[curr_waypoint_ind]
                # vehicle.set_transform(waypoint.transform)


                # TODO: there is a memory leak, not handling queue well.
                # rgb_saver.add_image(i, image_rgb)
                # depth_saver.add_image(i, image_depth)
                # seg_saver.add_image(i, image_semseg)

                ego_snap = snapshot.find(ego_id)
                vel_ego = ego_snap.get_velocity()
                vel_thresh = 1.0
                if vel_ego.x**2 + vel_ego.y**2 > vel_thresh:
                    image_rgb.save_to_disk('%s/rgb/%08d' % (args.logdir, num_frames_saved))
                    image_depth.save_to_disk('%s/depth/%08d' % (args.logdir, num_frames_saved))
                    image_semseg.save_to_disk('%s/seg/%08d' % (args.logdir, num_frames_saved))
                    num_frames_saved +=1
                    print('Frames Saved: %d of %d' % (num_frames_saved, max_frames))

                # image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                # fps = round(1.0 / snapshot.timestamp.delta_seconds)
                # # Draw the display.
                # draw_image(display, image_rgb)
                # draw_image(display, image_semseg, blend=True)
                # display.blit(
                #     font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                #     (8, 10))
                # display.blit(
                #     font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                #     (8, 28))
                # pygame.display.flip()

    finally:
        # print('Stopping saving threads.')
        # for saver in [rgb_saver, depth_saver, seg_saver]:
        #     saver.stop()

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')

# Main function to let the car teleport around an empty town.           
def run_empty_town(args, camera_config):
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (args.width, args.height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)

    world = client.get_world()

    try:        
        m = world.get_map()

        waypoint_list = m.generate_waypoints(float(args.waypoint_resolution)) # generate all waypoints with 2.0 m discretization
        curr_waypoint_ind = 0

        start_pose = waypoint_list[curr_waypoint_ind].transform #random.choice(m.get_spawn_points())
        start_pose.location.z += 2.0 # to prevent collisions
        waypoint = m.get_waypoint(start_pose.location)

        bp_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(bp_library.filter(args.filter)),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        # Get RGB, Depth, and Semantic Segmentation Camera with camera_config params.
        sensor_location = carla.Location(x=camera_config['x'], y=camera_config['y'],
                                         z=camera_config['z'])
        sensor_rotation = carla.Rotation(pitch=camera_config['pitch'],
                                         roll=camera_config['roll'],
                                         yaw=camera_config['yaw'])
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        bp_rgb = bp_library.find('sensor.camera.rgb')
        bp_depth = bp_library.find('sensor.camera.depth')
        bp_seg = bp_library.find('sensor.camera.semantic_segmentation')

        for bp in [bp_rgb, bp_depth, bp_seg]:
            bp.set_attribute('image_size_x', str(camera_config['width']))
            bp.set_attribute('image_size_y', str(camera_config['height']))
            bp.set_attribute('fov', str(camera_config['fov']))


        camera_rgb = world.spawn_actor(bp_rgb, sensor_transform, attach_to=vehicle)
        actor_list.append(camera_rgb)
        camera_depth = world.spawn_actor(bp_depth, sensor_transform, attach_to=vehicle)
        actor_list.append(camera_depth)
        camera_seg = world.spawn_actor(bp_seg, sensor_transform, attach_to=vehicle)
        actor_list.append(camera_seg)

        # rgb_saver   = CarlaImageSaver('%s/rgb' % args.logdir)
        # depth_saver = CarlaImageSaver('%s/depth' % args.logdir) 
        # seg_saver   = CarlaImageSaver('%s/seg' % args.logdir)

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_depth, camera_seg, fps=args.fps) as sync_mode:
            # for saver in [rgb_saver, depth_saver, seg_saver]:
            #     saver.start()

            for i in tqdm(range(1, len(waypoint_list))):
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_depth, image_semseg = sync_mode.tick(timeout=2.0)

                # # Choose the next waypoint and update the car location.
                # waypoint = random.choice(waypoint.next(1.5))
                # vehicle.set_transform(waypoint.transform)
                curr_waypoint_ind += 1
                waypoint = waypoint_list[curr_waypoint_ind]
                vehicle.set_transform(waypoint.transform)


                # TODO: there is a memory leak, not handling queue well.
                # rgb_saver.add_image(i, image_rgb)
                # depth_saver.add_image(i, image_depth)
                # seg_saver.add_image(i, image_semseg)

                image_rgb.save_to_disk('%s/rgb/%08d' % (args.logdir, i))
                image_depth.save_to_disk('%s/depth/%08d' % (args.logdir, i))
                image_semseg.save_to_disk('%s/seg/%08d' % (args.logdir, i))

                #image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                # Draw the display.
                draw_image(display, image_rgb)
                #draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()

    finally:
        # print('Stopping saving threads.')
        # for saver in [rgb_saver, depth_saver, seg_saver]:
        #     saver.stop()

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='CARLA Synchronous Camera Data Collector')
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
        '--res',
        metavar='WIDTHxHEIGHT',
        default='800x600',
        help='window resolution (default: 800x600)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.lincoln.*',
        help='actor filter (default: "vehicle.lincoln.*")')
    argparser.add_argument( 
        '--waypoint_resolution',
        default=50.0,
        type=float,
        help='Sampling resolution (m) for map waypoints')
    argparser.add_argument( 
        '--logdir',
        default='data_synced',
        help='Image logging directory for saved rgb,depth,and semantic segmentation images.')
    argparser.add_argument( 
        '--fps',
        default=10,
        type=int)
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    ''' Modified from scenario runner example.
        Todo: use the interface in https://github.com/carla-simulator/scenario_runner/blob/master/srunner/challenge/autoagents/agent_wrapper.py
        Removed sensors:
        {'type': 'sensor.camera.rgb', 'x':0.7, 'y':-0.4, 'z': 1.60,   'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0, 'width': 800, 'height': 600, 'fov': 100, 'id': 'Left'},
               {'type': 'sensor.camera.rgb', 'x':0.7, 'y':0.4, 'z':1.60, 'roll':0.0, 'pitch':0.0, 'yaw':45.0, 'width':800, 'height':600, 'fov':100, 'id': 'Right'},
               {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0, 'id': 'LIDAR'},
               {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
               {'type': 'sensor.speedometer','reading_frequency': 25, 'id': 'speed'}
    '''

    camera_config = {'x':0.7, 'y':0.0, 'z':1.60, \
                     'roll':0.0, 'pitch':0.0, 'yaw':0.0, \
                     'width':800, 'height': 600, 'fov':100} 

    try:
        #run_empty_town(args, camera_config)
        run_autopilot_with_npc(args, camera_config)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')