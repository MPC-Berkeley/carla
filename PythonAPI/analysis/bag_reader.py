import rosbag
import matplotlib.pyplot as plt
from transformations import euler_from_quaternion
import numpy as np
#from cv_bridge import CvBridge

def extract_carla_time(rostime):
    return rostime.secs + 1e-9 * rostime.nsecs

def extract_item_xyz(item):
    if hasattr(item, 'w'):
        raise TypeError("Item should only have xyz, not w (quaternion).")
    return [item.x, item.y, item.z]

def extract_roll_pitch_yaw(q):
    # TODO: It seems this returns heading first.
    
    # Reference this: https://github.com/carla-simulator/ros-bridge/blob/28d4ef607e07d217b873e35aee768d0a11b1bfa5/carla_ros_bridge/src/carla_ros_bridge/transforms.py#L96
    
    r, p, y = euler_from_quaternion((q.x, q.y, q.z, q.w))
    return [r,p,y]

def process_bag(bag):
    b = rosbag.Bag(bag)
    topics = b.get_type_and_topic_info().topics.keys()
    #print('This bag contains: ')
    #[print(x) for x in topics]
    
    res_dict = {}
    
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
        # Disgard the outlier at the end of the bag
        if ego_odometry_entry['position'][0] > 1e-3:
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
    
    '''
    # View just the first image.  This can be done in Python2, issues with Python3 here.
    image_viewed = False
    for topic, msg, t in b.read_messages('/carla/camera/rgb/front/image_color'):
        if image_viewed:
            break
        else:
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            plt.imshow(cv_image)
            image_viewed = True
    '''
    
    return res_dict