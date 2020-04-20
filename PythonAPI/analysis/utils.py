import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from keras.utils import to_categorical
from PIL import Image
from PIL import ImageDraw

def fix_angle(diff_ang):
    while diff_ang > np.pi:
        diff_ang -= 2 * np.pi
    while diff_ang < -np.pi:
        diff_ang += 2 * np.pi
    assert(-np.pi <= diff_ang and diff_ang <= np.pi)
    return diff_ang

# Misc. utils below for the following:
# (1) generating semantic birds eye view images (get_rect, generate_image, generate_image_ego)
# (2) trajectory visualization (get_parking_lot_image_hist, sup_plot, extract_data, generate_movie)

# TODO: clean up in the future.  Left alone for now as lower priority fix.

def get_parking_lot_image_hist(parking_lot, static_objs, feature, ego_dims, resize_factor=1.0):
    # code for the plotting of the parking lot
    ''' 
    Scene Image construction. Resolution/Image Params hardcoded for now.
    '''
    # Get center point (x,y) of the parking lot
    x0 = np.mean([min(x[0] for x in parking_lot),max(x[0] for x in parking_lot)])
    y0 = np.mean([min(x[1] for x in parking_lot),max(x[1] for x in parking_lot)])

    # Parking dimensions we want to consider
    parking_size = [20,65] # dX and dY
    res = 0.1 # in metres
    img_center = [x0,y0] # parking lot center

    h = int(parking_size[1] / res)
    w = int(parking_size[0] / res)

    num_imgs = len(feature)
    img_hist = np.zeros((num_imgs, h, w, 3), dtype=np.uint8)
    img_hist[:,:,:,0] = generate_image(parking_size,res,img_center,parking_lot)
    img_hist[:,:,:,1] = generate_image(parking_size,res,img_center,static_objs)

    for ind_p, ego_pose in enumerate(feature):
        ego_bb = [ego_pose[0],         # x
                  ego_pose[1],         # y
                  ego_dims['length'],  # dx
                  ego_dims['width'],   # dy
                  ego_pose[2]]         # theta

        img_hist[ind_p,:,:,2] = generate_image_ego(parking_size,res,img_center,ego_bb)
    
    # Return place holder
    img_return = img_hist    
    if resize_factor > 1.0:
        raise NotImplemented()
    elif resize_factor < 0.0:
        raise ValueError("Invalid resize_factor")
    else:
        # If we want resize, refill the slices
        h_resize = int(resize_factor * h)
        w_resize = int(resize_factor * w)
        img_return  = np.zeros((num_imgs, h_resize, w_resize, 3), dtype=np.uint8)
        # Go through each image
        for k in range(num_imgs):
            img_pil = np.asarray( Image.fromarray(img_hist[k,:,:,:]).resize((w_resize, h_resize)) )
            img_return[k,:,:,:] = img_pil

    img_return = np.flip(img_return, axis=1) # flip based on pixel axis to align with map frame (h)
    return img_return

def extract_data(pklfile, full_traj=False, crop_traj=False):
    with open(pklfile, 'rb') as f:
        dataset_all = pickle.load(f)
        
    # All the history trajectoreis (x, y, heading), with shape (batch_size, sequence_length, feature_dims)
    if crop_traj:
        history_traj_data = np.array(dataset_all['features'])[:, :, :3]
    else:
        history_traj_data = np.array(dataset_all['features'])

    # All the future trajectoreis (x, y), with shape (batch_size, sequence_length, feature_dims)
    if crop_traj:
        future_traj_data = np.array(dataset_all['labels'])[:, :, :2]
    else:
        future_traj_data = np.array(dataset_all['labels'])[:,:,:5]

    # All the goal positins and occupancy (x, y, occup), with shape (batch_size, (goal_nums * feature_dims))
    goals_position = np.array(dataset_all['goals'])
    goals_position = goals_position.reshape((goals_position.shape[0], goals_position.shape[1] * goals_position.shape[2]))

    # All intention labels, with shape (batch_size, goal_nums)
    goal_idx = np.array(dataset_all['labels'])[:, 0, -1]
    # Convert to one-hot and the last one is undecided (-1)
    one_hot_goal = to_categorical(goal_idx, num_classes=33)
    
    if full_traj:
        traj_idx = np.array(dataset_all['traj_idx'])
        return history_traj_data, future_traj_data, goals_position, one_hot_goal, traj_idx
    else:  
        return history_traj_data, future_traj_data, goals_position, one_hot_goal

def sup_plot(model_name, plot_set, traj_idx, goal_pred, traj_pred_dict, limit=None):
    goal_ind = np.arange(33)
    bar_width = 0.35
    # Recover the goal coordinates
    plot_goals_coords = plot_set['goal_position'].reshape((plot_set['goal_position'].shape[0], 32, 3))
    plot_hist_traj    = plot_set['history_traj_data']
    plot_future_traj  = plot_set['future_traj_data']
    plot_one_hot_goal = plot_set['one_hot_goal']

    if limit == None:
        num_itr = len(traj_idx) - 1
    else:
        num_itr = limit

    for num_traj in range(num_itr):
        
        print(model_name +": Start processing trajectory # %03d ....." % num_traj)
        start_idx = traj_idx[num_traj]
        end_idx   = traj_idx[num_traj+1]
        directory = './figures/' + model_name +'_%03d' % num_traj
        if not os.path.exists(directory):
            os.mkdir(directory)

        for t in range(start_idx, end_idx):

            fig = plt.figure(dpi=200)
            plt.suptitle(model_name, va='center')
            plt.subplot(211)

            vector = plot_goals_coords[t][-3,:2] - plot_goals_coords[t][-1,:2]
            th = np.arctan2(vector[1], vector[0])
            R = np.array([[ np.cos(th), np.sin(th)], \
                          [-np.sin(th), np.cos(th)]])

            # Plot the vehicle trajectory in the snippet
            plot_hist_traj_rot = plot_hist_traj[t][:,:2] @ R.T
            plot_future_traj_rot = plot_future_traj[t][:,:2] @ R.T
            plt.plot(plot_hist_traj_rot[:,0], plot_hist_traj_rot[:,1], 'k')
            plt.plot(plot_future_traj_rot[:,0], plot_future_traj_rot[:,1], color = '#1f77b4')
            
            probs = goal_pred[t].copy()
            # prob_undetermined = probs[-1]
            probs.sort()
            for top_k, traj_pred in traj_pred_dict.items():
                traj_pred_rot      = traj_pred[t][:, :2] @ R.T
                prob = probs[-1-top_k]
                plt.plot(traj_pred_rot[:,0], traj_pred_rot[:,1], '.', markersize = 3, color = '#ff770e', alpha= prob)

            # Plot the occupancy in the snippet
            plot_goals_coords_rot = plot_goals_coords[t][:,:2] @ R.T
            for goal, occup in zip(plot_goals_coords_rot, plot_goals_coords[t]):
                if occup[2] > 0:
                    plt.plot(goal[0], goal[1], 'ko', fillstyle='none', markersize = 9)
                else:
                    plt.plot(goal[0], goal[1], 'ko', markersize = 9)

            # Get the ground truth intention
            gt_idx = np.argmax(plot_one_hot_goal[t])
            if gt_idx == 32: # If it is "-1" -> undetermined 
                plt.plot(0, 0, 'v', fillstyle='bottom', color = '#1f77b4', markersize = 9)
            else:
                plt.plot(plot_goals_coords_rot[gt_idx][0], plot_goals_coords_rot[gt_idx][1], 'o', fillstyle='bottom', color = '#1f77b4', markersize = 9)

            num_goals_to_show = len(traj_pred_dict.keys()) if 'LSTM' in model_name else 3
            for top_k in range(num_goals_to_show):
                j = np.argsort(goal_pred[t])[-1-top_k]
                prob = probs[-1-top_k]
            # for j in best_k_idx:
                if j == 32:
                    # plt.plot(0, 0, 'v', fillstyle='none', color = '#ff770e', markersize = 9, alpha=prob_undetermined)
                    plt.plot(0, 0, 'v', fillstyle='none', color = '#ff770e', markersize = 9, alpha=prob)
                else:
                    prob = 1. if prob > 0.1 else 0.
                    plt.plot(plot_goals_coords_rot[j][0], plot_goals_coords_rot[j][1], 'o', fillstyle='none', color = '#ff770e', markersize = 9, alpha = prob)

            plt.title('Trajectory and Spots in Ego Frame')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
        #     plt.axis('equal')

            plt.subplot(212)
            p1 = plt.bar(goal_ind - bar_width/2, plot_one_hot_goal[t], bar_width, label='GT')
            p2 = plt.bar(goal_ind + bar_width/2, goal_pred[t], bar_width, label='Pred')
            plt.xlabel('Goal Index')
            plt.ylabel('Probability')
            plt.title('Likelihood of Selecting Different Goals')
            plt.legend()
            plt.tight_layout()
            
            
            fig.savefig('./figures/' + model_name + '_%03d/frame_%03d.png' % (num_traj, t-start_idx))
            plt.close(fig)
            
        fps = 2
        mv = os.system("ffmpeg -r {0:d} -i ./figures/{1:s}_{2:03d}/frame_%03d.png -vcodec mpeg4 -y ./figures/{1:s}_{2:03d}_movie.mp4".format(fps, model_name, num_traj) )
        if mv == 0:
            print( model_name + ": Trajectory # %03d movie saved successfully." % num_traj)
        else:
            print( model_name + ": Meet problem saving Trajectory # %03d movie." % num_traj)
            
def generate_movie(case_name, parking_lot, static_object_list, traj_pred_dict, features_global, labels_global, goal_pred, goal_gt, goal_snpts, top_k_goal, movie=True):
    print(case_name +": Start processing trajectory .....")
    directory = './figures/' + case_name
    if not os.path.exists(directory):
        os.mkdir(directory)


    for idx, feature in enumerate(features_global):
        fig = plt.figure(figsize=(4, 10), dpi=200, facecolor='w', edgecolor='k')
        ax = plt.gca()

        plt.rcParams['font.weight'] = 'normal'
        plt.rcParams['font.size'] = 14

        # Line
        for line_info in parking_lot:
            rect = patches.Rectangle((line_info[0]-line_info[2]/2, line_info[1]-line_info[3]/2),line_info[2],line_info[3],line_info[4], facecolor='k')
            ax.add_patch(rect)

        # Static objects
        for static_object in static_object_list:
            if static_object[0] < 275 or static_object[0] > 295:
                continue
            rect = patches.Rectangle((static_object[0]-static_object[2]/2, static_object[1]-static_object[3]/2),static_object[2],static_object[3],static_object[4], facecolor='#C6B7B3')
            ax.add_patch(rect)

        # Transformation
        ego_global = feature[-1]
        th = ego_global[2]
        R = np.array([[ np.cos(th),-np.sin(th)], \
                      [ np.sin(th), np.cos(th)]])
        curr = ego_global.copy()[:2]
        
        # Predicted goals
        probs = goal_pred[idx].copy()
        probs.sort()
        goal_snpts_global = goal_snpts[idx].copy()
        goal_snpts_global[:, :2] = goal_snpts_global[:, :2] @ R.T + curr

        # Plot the probability distribution
        for goal_idx in range(len(goal_pred[idx])-1):
            x_center = goal_snpts_global[goal_idx, 0]
            y_center = goal_snpts_global[goal_idx, 1]

            spot_w = 5.
            spot_h = 3.

            if x_center <= 285:
                rect_w = spot_w * goal_pred[idx, goal_idx]
                rect_h = spot_h
                x_left_corner = x_center - spot_w / 2.
                y_left_corner = y_center - spot_h / 2.
            else:
                rect_w = spot_w * goal_pred[idx, goal_idx]
                rect_h = spot_h
                x_left_corner = x_center + spot_w / 2. - rect_w
                y_left_corner = y_center - spot_h / 2.

            rect = patches.Rectangle((x_left_corner, y_left_corner), rect_w, rect_h, 0, facecolor='C9', alpha=0.5)
            ax.add_patch(rect)
            
        colors = ['C1', 'C2', 'C4']
        for top_k in top_k_goal:
            j = np.argsort(goal_pred[idx])[-1-top_k]
            prob = probs[-1-top_k]

            color = colors[top_k]
            alpha = 1 - 1/(len(top_k_goal)+1) * (top_k + 1)
            # Predicted Goal
            if j == 32:
                plt.plot(curr[0], curr[1], 'p', color = color, markersize = 20, alpha=alpha)
            else:
                plt.plot(goal_snpts_global[j, 0], goal_snpts_global[j, 1], 'p', color = color, markersize = 20, alpha=alpha)
        
        # Ground Truth Goal
        gt_idx = np.argmax(goal_gt[idx])
        if gt_idx == 32: # If it is "-1" -> undetermined 
            plt.plot(curr[0], curr[1], 'p', fillstyle='none', color = 'b', markeredgewidth = 5, markersize = 20)
        else:
            plt.plot(goal_snpts_global[gt_idx, 0], goal_snpts_global[gt_idx, 1], 'p', fillstyle='none', color = 'b', markeredgewidth = 5, markersize = 20)

        # Traj History
        plt.plot(feature[:,0], feature[:,1], linewidth = 3, color = 'k', alpha= 0.75)

        # Predicted Trajectory
        for top_k, traj_pred in sorted(traj_pred_dict.items(), reverse=True):
            prob = probs[-1-top_k]
        
            traj_pred_global = traj_pred[idx] @ R.T + curr

            color = colors[top_k]
            alpha = 1 - 1/(len(traj_pred_dict.items())+1) * (top_k + 1)
            plt.plot(traj_pred_global[:,0], traj_pred_global[:,1],'.', markersize = 15, color = color, alpha= alpha)
        
        # Ground Truth Future Traj
        label_global = labels_global[idx].copy()
        label_global_trans = label_global[:,:2]
        plt.plot(label_global_trans[:,0], label_global_trans[:,1], linewidth = 5, color = 'b', alpha= 0.5)


        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
            
        plt.plot()
        plt.axis('equal')

        plt.xlim(275, 295)
        plt.ylim(180, 240)

        fig.savefig('./figures/' + case_name + '/frame_%03d.png' % idx)
        plt.close(fig)

    if movie:
        fps = 2
        mv = os.system("ffmpeg -r {0:d} -i ./figures/{1:s}/frame_%03d.png -vcodec mpeg4 -y ./figures/{1:s}_movie.mp4".format(fps, case_name) )
        if mv == 0:
            print( case_name + ": Trajectory movie saved successfully.")
        else:
            print( case_name + ": Meet problem saving Trajectorymovie.")
    else:
        print(case_name + ": Processed, no movie generated")

def generate_image(parking_size,resolution,img_center,lines):
    
    # Total parking lot dimensions to consider
    dX, dY = parking_size
    
    h, w = int(dY/resolution), int(dX/resolution)
    x0, y0 = img_center

    # Lambda functions for transforming x,y into pixel space
    xp = lambda x : int(np.round(w / dX * ( x - x0) + w / 2))
    yp = lambda y : int(np.round(h / dY * ( y - y0) + h / 2 ))
    
    # Base image
    img = np.zeros((h,w),np.uint8)
    
    # Loop over parking lines
    for x_center,y_center,dx,dy,th in lines:
        
        # pixel width in x and y dimension
        dxp,dyp = int(np.round(dx/resolution)),int(np.round(dy/resolution))
        x1,y1 = int(np.round(xp(x_center)-dxp/2)),int(np.round(yp(y_center)-dyp/2))
        x2,y2 = int(np.round(xp(x_center)+dxp/2)),int(np.round(yp(y_center)+dyp/2))

        if x1 < 0 or x2 < 0:
            continue
        elif x1 > w or x2 > w:
            continue

        img[y1:y2,x1:x2] = 255
        
    return img

''' Rotates the rectangle with the angle around center of origin x,y'''
def get_rect(x, y, width, height, angle):
    rect = np.array([(-width/2, height/2), (width/2, height/2), (width/2, -height/2), (-width/2, -height/2), (-width/2, height/2)])
    theta = angle #(np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.matmul(rect,R) + offset
    return transformed_rect

def generate_image_ego(parking_size,resolution,img_center,lines):
    
    # Total parking lot dimensions to consider
    dX, dY = parking_size
    
    h, w = int(dY/resolution), int(dX/resolution)
    x0, y0 = img_center

    # Base image
    img = np.zeros((h,w),np.uint8)

    # Lambda functions for transforming x,y into pixel space
    xp = lambda x : int(np.round(w / dX * ( x - x0) + w / 2))
    yp = lambda y : int(np.round(h / dY * ( y - y0) + h / 2 ))
    
    # Read the ego object
    x_c,y_c,dx,dy,th = lines
    
    # Width and height in pixel space
    dxp, dyp = dx/resolution, dy/resolution
    
    img = Image.fromarray(img)

    # Draw a rotated rectangle on the image.
    draw = ImageDraw.Draw(img)
    rect = get_rect(x=xp(x_c), y=yp(y_c), width=dxp, height=dyp, angle=th)
    draw.polygon([tuple(p) for p in rect], fill=255)
    # Convert the Image data to a numpy array.
    new_data = np.asarray(img)
    return new_data