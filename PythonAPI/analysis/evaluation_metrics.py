import numpy as np
from keras.metrics import top_k_categorical_accuracy
from scipy.special import entr # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.entr.html

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
def weighted_dist_by_timestep(goal_pred, traj_pred_dict, traj_actual, normalize_top_k=True):
    # Computes a weighted average of the distance error for the top-k predicted trajectories.
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
    # Returns avg, min, max distance error across each timestep
    # assumption here is traj_pred_dict is unimodal, hence we only look at key 0.
    # weighted_dist_by_timestep is similar but meant for multimodal cases.
    diff = traj_pred_dict[0] - traj_actual # N by N_pred by 2
    diff_xy_norm = np.linalg.norm(diff, axis=2)
    return np.mean(diff_xy_norm, axis=0), np.min(diff_xy_norm, axis = 0), np.max(diff_xy_norm, axis=0)

def top_k_accuracy(goal_pred, goal_actual, k=1):
    # Returns empirical probability of the real goal being contained
    # in the top k most likely goal set from goal_pred.
    return np.mean(top_k_categorical_accuracy(goal_actual, goal_pred, k=k))

def mean_entropy(goal_pred):
    # Eeturns the mean entropy of the goal prediction dist.
    # Higher entropy indicates more uncertain predictions
    N = goal_pred.shape[0]
    
    entr_matrix = entr(goal_pred)
    entr_by_instance = np.sum(entr_matrix, axis=1) #entropy by snippet
    return np.mean(entr_by_instance)