#!/usr/bin/env python

import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import seaborn as sns
import pdb

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= "6"      # choose which GPU to run on.

# These are the models we want to evaluate (EKF, LSTM, CNN+LSTM)
from kf_impl import EKF_CV_MODEL
from lstm_impl import CombinedLSTM
from cnn_lstm_impl import CombinedCNNLSTM

from evaluation_metrics import build_train_test_splits

''' Parameters to adjust each time. '''
MODE = 'TRAIN' # 'TRAIN'   : train models and save them in model_dir
               # 'PREDICT' : load trained models in model_dir and save predictions in results_dir

# Dataset (tfrecords) specified via a search string.
tfrecord_search_str = '../examples/bags/dataset_01_2020/dataset*.tfrecord'

# Model directory where trained models are saved (if MODE is 'TRAIN')
model_dir = './models'

# Results directory where predictions are saved (if MODE is 'PREDICT')
results_dir = './results'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


''' Train and Predict Functions '''
def train_models(names, models, train_sets, model_dir, num_epochs=200, batch_size=32, verbose=1):
    for name, model in zip(names, models):
        for i_fold, train_set in enumerate(train_sets):
            print('Training %s, Fold %d at ' % (name, i_fold), time.time())
            if 'LSTM' in name or 'CNN' in name:
                model.fit(train_set, 
                          num_epochs=num_epochs, 
                          batch_size=batch_size, 
                          verbose=verbose)
                          #save_name='%s_fold%d' % (name, i_fold))
            elif 'EKF' in name:
                model.fit(train_set)
            else:
                raise ValueError("invalid model for testing")

            model.save('%s/%s_fold%d' % (model_dir, name, i_fold)) 

        print('Finished model: ', name)

def predict_models(names, models, train_sets, test_sets, results_dir, top_k_goal=None):
    for name, model in zip(names, models):
        for i_fold, (train_set, test_set) in enumerate(zip(train_sets, test_sets)):
            print('Loading %s, Fold %d at ' % (name, i_fold), time.time())
            filename = '%s/%s_fold%d' % (model_dir, name, i_fold)
            if 'EKF' in name:
                filename += '.pkl'
            model.load(filename)
        
            # Make the predictions
            pred_dict = {}
            for tkey, tset in zip(['train', 'test'], [train_set, test_set]):
                print('\tStarted prediction for tkey: ', tkey, ' at ', time.time())
                goal_pred, goal_gt, traj_pred_dict, traj_gt = model.predict(tset) # either no goal or ground truth
                
                res_dict = {}
                for res_key in ['goal_pred', 'goal_gt', 'traj_pred_dict', 'traj_gt']:
                    res_dict[res_key] = eval(res_key)
                pred_dict[tkey] = res_dict
                
                if 'EKF' in name or 'no_goal' in name or not top_k_goal:
                    pass
                else:
                    _, _, traj_pred_dict_multimodal, _ = model.predict(tset, top_k_goal)
                    res_dict['traj_pred_dict_mm'] = traj_pred_dict_multimodal
                    
            # Save them to file.
            pickle.dump(pred_dict, open('%s/%s_fold%d_pred.pkl' % (results_dir, name, i_fold), 'wb'))
 
        print('Finished model: ', name)


''' Dataset Building, Model Selection, and Analysis '''
tffiles_to_process = glob.glob(tfrecord_search_str)
train_sets, test_sets = build_train_test_splits(tffiles_to_process, num_tf_folds=5) # k-fold cross validation sets

# Build the model bank.  If GPU memory is problematic, can generate models in a loop instead.
models = []
names = []

# Common LSTM/CNN+LSTM parameters
# TODO: Hard coded for now, else need to read the tfrecord and extract it from there.
# This is tuned for our dataset.
history_shape       = (5, 3)         # pose history: history horizon of 5, pose dim of 3
image_input_shape   = (5,325,100,3)  # image history: history horizon of 5, image is 325x100x3
goal_position_shape = (32*3,)        # occupancy info: 32 spots x (x,y,is_free) flattened
one_hot_goal_shape  = (32+1,)        # intent prediction dim: 32 spots + 1 "undetermined" category
future_shape        = (20, 2)        # position future: future horizon of 20, xy position dim of 2

image_feature_dim = 16               # output dimension of CNN image encoder
hidden_dim = 100                     # hidden dimension for LSTM 
top_k_goal = [0,1,2]                 # specify which trajectory rollouts to predict for multimodal predictions
                                     # (i.e. 0 = most probable intent, 1 = second-most probable intent, etc.)

''' Start of models for evaluation. '''
# EKF baseline.
models.append(EKF_CV_MODEL(x_init=np.zeros(5), P_init=np.eye(5), R=np.diag([1e-3]*3), dt=0.1))
names.append('EKF_CV')

# Multimodal CNN+LSTM and CNN models.
gamma = 1.0                  # occupancy penalty weight for intent prediction 
for beta in [0.1, 0.5, 1.0]: # max-entropy weight for intent prediction
    models.append(
        CombinedCNNLSTM(history_shape,
                        goal_position_shape,
                        image_input_shape,
                        one_hot_goal_shape,
                        future_shape,
                        hidden_dim,
                        beta=beta,
                        gamma=gamma,
                        image_feature_dim=image_feature_dim,
                        use_goal_info=True))
    names.append('CNN_b%.3f_g%.3f' % (beta,gamma))

    models.append(
        CombinedLSTM(history_shape,
                     goal_position_shape,
                     one_hot_goal_shape,
                     future_shape,
                     hidden_dim,
                     beta=beta,
                     gamma=gamma,
                     use_goal_info=True))
    names.append('LSTM_b%.3f_g%.3f' % (beta,gamma))

# Unimodal variants of CNN+LSTM/LSTM with no intent label,
# provided for trajectory regression (only history features).
models.append(
    CombinedCNNLSTM(history_shape,
                    goal_position_shape,
                    image_input_shape,
                    one_hot_goal_shape,
                    future_shape,
                    hidden_dim,
                    beta=beta,
                    gamma=gamma,
                    image_feature_dim=image_feature_dim,
                    use_goal_info=False))
names.append('CNN_no_goal')

models.append(
    CombinedLSTM(history_shape,
                 goal_position_shape,
                 one_hot_goal_shape,
                 future_shape,
                 hidden_dim,
                 beta=beta,
                 gamma=gamma,
                 use_goal_info=False))
names.append('LSTM_no_goal')
''' End of models for evaluation. '''

if MODE is 'TRAIN':
    train_models(names, models, train_sets, model_dir, num_epochs=200, batch_size=32, verbose=1)
elif MODE is 'PREDICT':
    predict_models(names, models, train_sets, test_sets, results_dir, top_k_goal=top_k_goal)
else:
    raise ValueError("Invalid mode: %s" % MODE)