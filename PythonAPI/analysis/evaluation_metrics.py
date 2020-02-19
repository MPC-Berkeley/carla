import numpy as np
from keras.metrics import top_k_categorical_accuracy
from scipy.special import entr # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.entr.html
import sys
import pdb
import matplotlib.pyplot as plt

# Helper functions to split test/train and evaluate prediction accuracy.

##########################################
def build_train_test_splits(tf_files_list, num_tf_folds=5):
    # Given num_tf_folds and tf_files_list corresponding to this number,
    # this function assembles all combinations of test/train for num_tf_folds-fold
    # cross validation.  Essentially looping over the "hold-out" set.

    train_sets = []
    test_sets  = []
     
    for test_fold_ind in range(num_tf_folds):
        train_set = [x for x in tf_files_list if int(x.split('/')[-1].split('_')[2]) != test_fold_ind]
        test_set = [x for x in tf_files_list if int(x.split('/')[-1].split('_')[2]) == test_fold_ind]
        
        train_sets.append(train_set)
        test_sets.append(test_set)
        
    return train_sets, test_sets

##########################################
def min_dist_by_timestep(traj_pred_dict, traj_actual):
    # Multimodal metric finding the closest trajectory among all k rollouts
    # and then reporting dist_by_timestep for this trajectory.
    # Specifically, the trajectory selected is the one that is the argmin
    # of the min average displacement error objective (see Waymo MultiPath paper).
    # d_k(t) = distance error for the kth rollout at timestep t

    num_pred_traj = len(traj_pred_dict.keys())
    # M = # of dataset instances, N = time horizon, 2 (xy) 
    M = traj_pred_dict[0].shape[0]
    N = traj_pred_dict[0].shape[1]
    
    min_ade = np.ones((M)) * sys.float_info.max                  # minimum average displacement error as defined in MultiPath paper by Waymo
    min_dist_by_timestep = np.ones((M,N)) * sys.float_info.max   # minimum distance to single trajectory minimizing ade among k rollouts

    for k in range(num_pred_traj):
        traj_pred_k = traj_pred_dict[k] # M by N by 2, kth rollout
        diff = traj_pred_k - traj_actual # M by N by 2, kth rollout xy position difference
        diff_xy_norm = np.linalg.norm(diff, axis=2) # M by N, kth rollout 2-norm error by timestep (i.e. d_k(.) defined above)
        ade = np.mean(diff_xy_norm, axis=1) # M, average displacement error

        ade_lower_inds = np.ravel( np.argwhere(ade < min_ade) ) # find where ade is strictly improved on an instance index level
        min_ade[ade_lower_inds] = ade[ade_lower_inds]           # update min_ade if we found a better candidate by instance index

        min_dist_by_timestep[ade_lower_inds,:] = diff_xy_norm[ade_lower_inds,:] # update min_dist if we found a better candidate by instance index

    return np.mean(min_dist_by_timestep, axis=0), np.mean(min_ade)

def weighted_dist_by_timestep(goal_pred, traj_pred_dict, traj_actual, normalize_top_k=True):
    # Multimodal metric computing a weighted average of the distance error for the top-k predicted trajectories.
    # Essentially, computes the distance error at timestep t in the following way:
    # wd(t) = d_1(t) * p_1(t) + d_2(t) * p_2(t) + ... d_k(t) * p_k(t)
    # where subscript = kth rollout in traj_pred_dict, 
    # d_k(t) = distance error for the kth rollout at timestep t
    # p_k(t) = probability of the kth rollout

    # If normalize_top_k is True, then we normalize the probability distribution to only cover
    # the top k entries rather than all N_goals + 1 options.

    # M = # of dataset instances, N = time horizon, 2 (xy) 
    M = traj_pred_dict[0].shape[0]
    N = traj_pred_dict[0].shape[1]
    
    weighted_sum = np.zeros((M, N)) # final result is a weighted average of distance errors

    num_pred_traj = len(traj_pred_dict.keys()) # number of rollouts (should be k)
    
    top_k_probs = -np.sort(-goal_pred, axis=1)[:,:num_pred_traj] # get the probabilities corresponding to top k in order
    
    if normalize_top_k:
        top_k_probs = top_k_probs / top_k_probs.sum(axis=1, keepdims=True) # normalize so prob dist adds up to 1.
    
    for k in range(num_pred_traj):
        traj_pred_k = traj_pred_dict[k] # M by N by 2, kth rollout
        diff = traj_pred_k - traj_actual # M by N by 2, kth rollout xy position difference
        diff_xy_norm = np.linalg.norm(diff, axis=2) # M by N, kth rollout 2-norm error by timestep (i.e. d_k(.) defined above)

        for i in range(N):
            diff_xy_norm[:,i] *= top_k_probs[:,k] # weight d_k by p_k as above
        
        weighted_sum += diff_xy_norm # add the weighted component to the average
    return np.mean(weighted_sum, axis=0)

def dist_by_timestep(traj_pred_dict, traj_actual):
    # Unimodal metric returning avg, min, max distance error across each timestep.
    # We assume traj_pred_dict is unimodal, hence we only look at key 0.
    # {weighted, min}_dist_by_timestep is similar but meant for multimodal cases.
    assert(len(traj_pred_dict.keys()) == 1)
    diff = traj_pred_dict[0] - traj_actual # M by N_pred by 2
    diff_xy_norm = np.linalg.norm(diff, axis=2)
    return np.mean(diff_xy_norm, axis=0), np.min(diff_xy_norm, axis = 0), np.max(diff_xy_norm, axis=0)

##########################################
def top_k_accuracy(goal_pred, goal_actual, k=1):
    # Returns empirical probability of the real goal being contained
    # in the top k most likely goal set from goal_pred.
    return np.mean(top_k_categorical_accuracy(goal_actual, goal_pred, k=k))

def mean_entropy(goal_pred):
    # Returns the mean entropy of the goal prediction dist.
    # Higher entropy indicates more uncertain predictions
    N = goal_pred.shape[0]
    
    entr_matrix = entr(goal_pred)
    entr_by_instance = np.sum(entr_matrix, axis=1) #entropy by snippet
    return np.mean(entr_by_instance)

##########################################
if __name__=='__main__':
    # Tests for key metrics.
    def generate_traj_pred_dict():
        t = np.arange(0.0, np.pi/2, np.pi/40)
        # Trajectory 1:
        x = np.sin(t)
        y1 = 1 - np.cos(t)
        traj_1 = np.column_stack((x,y1))
        traj_1 = np.expand_dims(traj_1, axis=0) # 1 by 20 by 2

        # Trajectory 2:
        y2 = -y1
        traj_2 = np.column_stack((x,y2))
        traj_2 = np.expand_dims(traj_2, axis=0) # 1 by 20 by 2

        # Trajectory 3:
        y3 = np.zeros_like(y1)
        traj_3 = np.column_stack((x,y3))
        traj_3 = np.expand_dims(traj_3, axis=0) # 1 by 20 by 2

        traj_pred_dict = {}
        traj_pred_dict[0] = np.concatenate((traj_1, traj_1), axis=0)
        traj_pred_dict[1] = np.concatenate((traj_2, traj_2), axis=0)
        traj_pred_dict[2] = np.concatenate((traj_3, traj_3), axis=0)
        return traj_pred_dict

    def generate_traj_gt():
        t = np.arange(0.0, np.pi/2, np.pi/40)
        x = np.sin(t)
        y1 = np.sin(t)
        y2 = -0.5 * np.sin(t)
        
        traj_gt_1 = np.column_stack((x,y1))
        traj_gt_1 = np.expand_dims(traj_gt_1, axis=0)

        traj_gt_2 = np.column_stack((x,y2))
        traj_gt_2 = np.expand_dims(traj_gt_2, axis=0)

        traj_gt = np.concatenate((traj_gt_1, traj_gt_2))
        return traj_gt

    traj_pred_dict = generate_traj_pred_dict()
    traj_gt = generate_traj_gt()

    min_de_timestep, min_ade, argmin_ade = min_dist_by_timestep(traj_pred_dict, traj_gt)
    print(min_de_timestep)
    print(min_ade)
    print(argmin_ade)

    for k in traj_pred_dict.keys():
        plt.plot(traj_pred_dict[k][0,:,0], traj_pred_dict[k][0,:,1])

    plt.plot(traj_gt[0,:,0], traj_gt[0,:,1], 'rx')
    plt.plot(traj_gt[1,:,0], traj_gt[1,:,1], 'bx')

    plt.show()