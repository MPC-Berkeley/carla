import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from keras.utils import to_categorical

import pdb

def fix_angle(diff_ang):
    while diff_ang > np.pi:
        diff_ang -= 2 * np.pi
    while diff_ang < -np.pi:
        diff_ang += 2 * np.pi
    assert(-np.pi <= diff_ang and diff_ang <= np.pi)
    return diff_ang

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