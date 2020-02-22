import rosbag
import matplotlib.pyplot as plt
from transformations import euler_from_quaternion
import numpy as np

# Code Status (as of 2/6/20): this works but is hard-coded for our parking lot experiment.  
# Some things like Euler angles can be fixed, as well as applying outlier information to every field in the res_dict.

def extract_parking_lines():
  # Returns hard-coded lane markings for the parking lot map (called exp).
  # Top = horizontal line for the "top" row
  # Bot = horizontal line for the "bottom" row
  # remaining lines are the vertical lines separating parking spots
  # Contains the lines in format: x, y, dx, dy, theta,
  # where (x,y) is the line center, (dx,dy) is the thickness, and theta is the orientation

  lines = []
  # Length and width for the long lines
  dX, dY = round(0.185915, 3), round(51.3499985, 3)

  # Top big line
  xTop, yTop = round(29365.3296875 / 100, 3), round(20952.0 / 100, 3)
  lines.append([xTop, yTop, dX, dY, 0])

  # Bottom big line
  xBot, yBot  = round(27615.4296875 / 100, 3), round(20952.0 / 100, 3)
  lines.append([xBot,yBot,dX, dY, 0])

  # Length and width for the short lines
  dx, dy = 5., round(0.170981, 3)

  # Left-most top short line
  xt, yt = round(29111.3886719 / 100, 3), round(23510.8730469 / 100, 3)
  for k in range(17):
    lines.append([xt,yt-k*3.2,dx,dy,0])

  # Left-most bottom short line
  xb, yb = round(27871.3886719 / 100, 3), round(23510.8730469 / 100, 3)
  for k in range(17):
    lines.append([xb,yb-k*3.2,dx,dy,0])

  return lines

def extract_carla_time(rostime):
    return rostime.secs + 1e-9 * rostime.nsecs

def extract_item_xyz(item):
    if hasattr(item, 'w'):
        raise TypeError("Item should only have xyz, not w (quaternion).")
    return [item.x, item.y, item.z]

def extract_roll_pitch_yaw(q):
    # TODO: It seems this returns heading first, with what is called "r" here.
    
    # Reference this: https://github.com/carla-simulator/ros-bridge/blob/28d4ef607e07d217b873e35aee768d0a11b1bfa5/carla_ros_bridge/src/carla_ros_bridge/transforms.py#L96
    
    r, p, y = euler_from_quaternion((q.x, q.y, q.z, q.w))
    return [r,p,y]

def outlier_removal(res_dict, position_thresh=1.):
    # Imperfect method to get rid of jumps in data using ego's odometry.
    # First check we are near the ego spawn point.
    # Then look for jumps with respect to expected position using velocity info.
    # position_thresh is the threshold in meters below which we say a jump did not happen.

    start_point = np.array([2.85000000e+02, 2.39998810e+02]) # ego spawn location
    start_time = None; start_ind = None; final_time = None; final_ind = None
    
    for i in range( 1, len(res_dict['ego_odometry_list'])):
        prev_odom = res_dict['ego_odometry_list'][i-1]
        new_odom  = res_dict['ego_odometry_list'][i]
        
        # TODO: don't need estimated new pose.  This just cares about proximity
        # to the start point.
        dt = new_odom['time'] - prev_odom['time']
        prev_position = np.array(prev_odom['position'])
        prev_velocity = np.array(prev_odom['linear_velocity'])
        
        est_new_position = prev_position + dt * prev_velocity
        
        new_position = np.array(new_odom['position'])
        
        if np.linalg.norm(prev_position[:2] - start_point) < position_thresh:
            start_time = prev_odom['time']
            start_ind = i-1
            break
            
    if start_time is not None:
        print('\t File: %s, Start point processed. Setting start time %.3f vs. first entry at time %.3f for ego odom.' % \
              (res_dict['name'], start_time, res_dict['ego_odometry_list'][0]['time'])) 
        # Remove the corresponding outliers in ego-odometry, GPS and all other time-varying entries
        res_dict['ego_odometry_list'] = res_dict['ego_odometry_list'][start_ind:]
    
    i = 1
    filter_odom = [res_dict['ego_odometry_list'][0]]
    while i < len(res_dict['ego_odometry_list']):
        prev_odom = filter_odom[-1]
        new_odom  = res_dict['ego_odometry_list'][i]
        
        dt = new_odom['time'] - prev_odom['time']
        prev_position = np.array(prev_odom['position'])
        prev_velocity = np.array(prev_odom['linear_velocity'])
        
        est_new_position = prev_position + dt * prev_velocity
        
        new_position = np.array(new_odom['position'])
        
        if np.linalg.norm(new_position[:2] - est_new_position[:2]) < position_thresh:
            filter_odom.append(res_dict['ego_odometry_list'][i])
        
        i += 1

    res_dict['ego_odometry_list'] = filter_odom
           
def process_bag(bag):
    b = rosbag.Bag(bag)
    topics = b.get_type_and_topic_info().topics.keys()

    if len(topics) == 0:
        raise ValueError("\tNothing recorded, so skipping this instance.")

    #print('This bag contains: ')
    #[print(x) for x in topics]
    
    res_dict = {}
    res_dict['name'] = bag.split("/")[-1].split(".bag")[0]
    
    ''' Time-independent data '''
    # (Id, name) dictionary of other vehicles
    vehicle_dict = {}
    for topic, msg, t in b.read_messages('/carla/actor_list'):
        for actor in msg.actors:
            if 'vehicle' in actor.type:
                vehicle_dict[actor.id] = '%s_%s' % (actor.type, actor.rolename)
    res_dict['vehicle_dict'] = vehicle_dict
    
    # Intention signal
    intention_time_list = []
    for topic, msg, t in b.read_messages('/intention'):
        intention_time_list.append(extract_carla_time(t))

    if len(intention_time_list) == 0:
        raise ValueError("\tNo intention time, so skipping this instance.")
    else:
        res_dict['intention_time_list'] = intention_time_list
    
    
    # Ego Vehicle Params
    ego_info_dict = {}
    for topic, msg, t in b.read_messages('/carla/hero/vehicle_info'):
        simple_items_to_extract = ['id', 'type', 'rolename', 'max_rpm', 'moi', \
                                   'damping_rate_full_throttle', \
                                   'damping_rate_zero_throttle_clutch_engaged', \
                                   'damping_rate_zero_throttle_clutch_disengaged', \
                                   'use_gear_autobox', \
                                   'clutch_strength', \
                                   'mass', \
                                   'drag_coefficient']
        for item in simple_items_to_extract:
            val = getattr(msg, item)
            
            # Shorten really long names for sio.savemat
            if item == 'damping_rate_full_throttle':
                name = 'damping_rate_ft'
            elif item == 'damping_rate_zero_throttle_clutch_engaged':
                name = 'damping_rate_ztce'
            elif item == 'damping_rate_zero_throttle_clutch_disengaged':
                name = 'damping_rate_ztcd'
            else:
                name = item
            ego_info_dict[name] = val
        
        ego_info_dict['center_of_mass'] = extract_item_xyz(msg.center_of_mass)
        
        wheel_params = []
        for wheel in msg.wheels:
            wheel_params.append([wheel.tire_friction, wheel.damping_rate, wheel.max_steer_angle])
        wheel_params = np.array(wheel_params)
        ego_info_dict['wheels'] = wheel_params
    res_dict['ego_info_dict'] = ego_info_dict
    
    # World settings
    world_dict = {}
    for topic, msg, t in b.read_messages('/carla/status'):
        world_dict['dt'] = msg.fixed_delta_seconds 
        world_dict['sync_mode'] = msg.synchronous_mode
        world_dict['sync_mode_active'] = msg.synchronous_mode_running
        break # assuming this is held constant in our data.
        
    for topic, msg, t in b.read_messages('/carla/world_info'):
        world_dict['map'] = msg.map_name
        #TODO: opendrive xml storage + processing
        break # assuming this is held constant in our data.
    res_dict['world_dict'] = world_dict
       
    ''' Time-varying data '''
    
    # Ego collision list
    ego_collision_list = []
    for topic, msg, t in b.read_messages('/carla/hero/collision'):
        ego_collision_entry = {}
        ego_collision_entry['time'] = extract_carla_time(msg.header.stamp)
        ego_collision_entry['other_id'] = msg.other_actor_id
        ego_collision_entry['normal_impulse'] = extract_item_xyz(msg.normal_impulse)
        ego_collision_list.append(ego_collision_entry)
    res_dict['ego_collision_list'] = ego_collision_list
    
    # Ego Vehicle Control Status
    ego_control_list = []
    for topic, msg, t in b.read_messages('/carla/hero/vehicle_status'):
        ego_control_entry = {}
        ego_control_entry['time'] = extract_carla_time(msg.header.stamp)
        ego_control_entry['velocity'] = msg.velocity
        ego_control_entry['acceleration'] = extract_item_xyz(msg.acceleration.linear) # ang. accel. = 0
        ego_control_entry['orientation'] = extract_roll_pitch_yaw(msg.orientation) # TODO: fix this
        
        items_in_control_field = ['throttle', 'steer', 'brake', 'hand_brake', \
                                  'reverse', 'gear', 'manual_gear_shift']
        for item in items_in_control_field:
            ego_control_entry[item] = getattr(msg.control, item)
        
        ego_control_list.append(ego_control_entry)
    res_dict['ego_control_list'] = ego_control_list
        
    # Ego GPS Fix List
    ego_gps_list = []
    for topic, msg, t in b.read_messages('/carla/hero/gnss/front/fix'):
        ego_gps_entry = {}
        ego_gps_entry['time'] = extract_carla_time(msg.header.stamp)
        for item in ['latitude', 'longitude', 'altitude']:
            ego_gps_entry[item] = getattr(msg, item)
        ego_gps_list.append(ego_gps_entry)
    res_dict['ego_gps_list'] = ego_gps_list
    
    # Ego Odometry
    ego_odometry_list = []
    for topic, msg, t in b.read_messages('/carla/hero/odometry'):
        ego_odometry_entry = {}
        ego_odometry_entry['time'] = extract_carla_time(msg.header.stamp)
        ego_odometry_entry['position'] = extract_item_xyz(msg.pose.pose.position)
        ego_odometry_entry['orientation'] = extract_roll_pitch_yaw(msg.pose.pose.orientation)
        ego_odometry_entry['linear_velocity'] = extract_item_xyz(msg.twist.twist.linear)
        ego_odometry_entry['angular_velocity'] = extract_item_xyz(msg.twist.twist.angular)
        ego_odometry_list.append(ego_odometry_entry)
    res_dict['ego_odometry_list'] = ego_odometry_list
            
    # Other Vehicle Odometry
    vehicle_odometry_dict = {}
    for topic, msg, t in b.read_messages('/carla/odometry'):
        
        veh_id = msg.child_frame_id.split('/')[-1] # e.g. "vehicle/847" -> "847"
        if veh_id not in vehicle_odometry_dict.keys():
            vehicle_odometry_dict[veh_id] = []
        
        odometry_entry = {}
        odometry_entry['time'] = extract_carla_time(msg.header.stamp)
        odometry_entry['position'] = extract_item_xyz(msg.pose.pose.position)
        odometry_entry['orientation'] = extract_roll_pitch_yaw(msg.pose.pose.orientation)
        odometry_entry['linear_velocity'] = extract_item_xyz(msg.twist.twist.linear)
        odometry_entry['angular_velocity'] = extract_item_xyz(msg.twist.twist.angular)
        
        vehicle_odometry_dict[veh_id].append(odometry_entry)
        
    res_dict['vehicle_odometry_dict'] = vehicle_odometry_dict
            
    # Other Vehicle Object List
    vehicle_object_lists = []
    for topic, msg, t in b.read_messages('/carla/hero/objects'):
        veh_obj_list = []
        for obj in msg.objects:   
            veh_obj_entry = {}
            veh_obj_entry['time'] = extract_carla_time(obj.header.stamp)
            for item in ['id', 'detection_level', 'object_classified', \
                          'classification', 'classification_certainty', \
                          'classification_age']:
                veh_obj_entry[item] = getattr(obj, item)
            
            veh_obj_entry['position'] = extract_item_xyz(obj.pose.position)
            veh_obj_entry['orientation'] = extract_roll_pitch_yaw(obj.pose.orientation)
            veh_obj_entry['linear_velocity'] = extract_item_xyz(obj.twist.linear)
            veh_obj_entry['angular_velocity'] = extract_item_xyz(obj.twist.angular)
            veh_obj_entry['acceleration'] = extract_item_xyz(obj.accel.linear) # angular accel = 0.
            veh_obj_entry['shape_type'] = obj.shape.type # should all be rectangles (1) for vehicles 
            veh_obj_entry['dimensions'] = obj.shape.dimensions
            veh_obj_list.append(veh_obj_entry)
        vehicle_object_lists.append(veh_obj_list)
    res_dict['vehicle_object_lists'] = vehicle_object_lists
   
    ''' Items needed for rasterized image representation are below '''
    res_dict['parking_lot'] = extract_parking_lines()
    res_dict['ego_dimensions'] = {}
    for topic, msg, t in b.read_messages('/carla/objects'):
        ego_ind = -1
        for ind, obj in enumerate(msg.objects):
            if obj.id == ego_info_dict['id']:
                ego_ind = ind
                break

        assert ego_ind >= 0, "Ego vehicle not found in object list!"

        res_dict['ego_dimensions']['length'] = msg.objects[ego_ind].shape.dimensions[0]
        res_dict['ego_dimensions']['width']  = msg.objects[ego_ind].shape.dimensions[1]
        break

    res_dict['static_object_list'] = []
    # Contains the bounding boxes in format: x, y, dx, dy, theta,
    # where (x,y) is the bb center, (dx,dy) is the bb thickness, and theta is the orientation
    static_object_ind = -1 
    for ind, obj_list in enumerate(res_dict['vehicle_object_lists']):
        if abs(obj_list[0]['acceleration'][-1]) < 0.2:
            # Hack to find when the cars stop falling after being spawned.
            static_object_ind = ind
            break
    assert static_object_ind >= 0, "Could not find when the vehicles stop moving!"
    for obj in res_dict['vehicle_object_lists'][static_object_ind]:
        x, y, z = obj['position']
        theta = round(obj['orientation'][0], 2) # TODO, confirm this again.
        assert theta==0., "Parked vehicle has non-zero heading!"
        dx, dy, dz    = obj['dimensions']
        obj_entry = [x, y, dx, dy, theta]
        res_dict['static_object_list'].append(obj_entry)

    outlier_removal(res_dict)
    return res_dict