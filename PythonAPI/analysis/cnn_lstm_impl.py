import re
import numpy as np
import unidecode
from IPython import get_ipython;
import matplotlib.pyplot as plt
import pandas
import math
from keras import metrics, optimizers
from keras import Input, Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Softmax, Flatten, concatenate, Conv2D, MaxPooling2D
from keras.layers import Activation, TimeDistributed, RepeatVector, Embedding, LeakyReLU, BatchNormalization
from keras.layers.recurrent import LSTM
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import keras.applications.mobilenet_v2 as mbnet
import keras.backend as K # for custom loss function
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tfrecord_utils import _parse_function
import tensorflow as tf
import glob
from datetime import datetime
from tqdm import tqdm
import time

# This file provides the implementation of the CNN-LSTM used for intent/goal and trajectory prediction.

def cnn_base_network(input_shape, img_feature_dim):
	# Base network taking in the semantic birds view and returning image features for the LSTM.
	cnn_model = Sequential()
	
	cnn_model.add( Conv2D(16, kernel_size=(3,3), strides=1, activation='relu', input_shape=input_shape,kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3)) )
	cnn_model.add( MaxPooling2D(pool_size=(2,2)) )
	cnn_model.add( Dropout(0.1) )

	cnn_model.add( Conv2D(32, kernel_size=(3,3), strides=2, activation='relu',kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3)) )
	cnn_model.add( MaxPooling2D(pool_size=(2,2)) )
	cnn_model.add( Dropout(0.1) )
	
	cnn_model.add( Flatten() )
	cnn_model.add( Dense(128,kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3)) )
	cnn_model.add( Dropout(0.2) )
	cnn_model.add( Dense(img_feature_dim,kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3)) )
	cnn_model.add( BatchNormalization())
	return cnn_model

class CombinedCNNLSTM(object):
	def __init__(self, history_shape, goals_position_shape, image_input_shape, one_hot_goal_shape, future_shape, hidden_dim, beta=0.1, gamma=0.1, image_feature_dim=32, use_goal_info=True):
		traj_input_shape    = (history_shape[1], history_shape[2]) # motion history: should be N_hist by state_dim
		goal_input_shape    = (goals_position_shape[1],)           # occupancy: num_spots by (x_spot, y_spot, is_free) -> flattened to 3 * num_spots
		n_outputs           = one_hot_goal_shape[1]                # goal/intent classification: num_spots + 1
		intent_input_shape  = (n_outputs,)                         # intent input to trajectory prediction model: intent input is the same size as the intent classification above.
		future_horizon      = future_shape[1]                      # future prediction horizon: N_pred
		future_dim          = future_shape[2]                      # future prediction state dimension: (in our usage, we drop the so state_dim=2 for output)

		self.use_goal_info = use_goal_info # if True, use ground truth intent information while training.  Else zero out the intent label and only learn from motion info.
		self.goal_model = GoalCNNLSTM(traj_input_shape, goal_input_shape, image_input_shape, n_outputs, beta, gamma, hidden_dim=hidden_dim, img_feature_dim=image_feature_dim)
		self.traj_model = TrajCNNLSTM(traj_input_shape, intent_input_shape, image_input_shape, future_horizon, future_dim, use_goal_info=self.use_goal_info, hidden_dim=hidden_dim, img_feature_dim=image_feature_dim)	

	def fit(self, train_set, val_set, num_epochs=100, batch_size=64, verbose=0):
		self.goal_model.fit_model(train_set, val_set, num_epochs=num_epochs, batch_size=batch_size, \
		                          verbose=verbose)
		self.traj_model.fit_model(train_set, val_set, num_epochs=num_epochs, batch_size=batch_size, \
			                      verbose=verbose)

	def predict(self, test_set, top_k_goal=[]):
		# top_k_goal, if filled in, can allow for multimodal predictions based on estimated intent.
		# See predict method of TrajCNNLSTM for details.
		goal_pred, goal_gt = self.goal_model.predict(test_set)
		top_idxs = np.argsort(goal_pred,axis=1)
		traj_pred_dict, traj_gt = self.traj_model.predict(test_set,top_idxs,top_k_goal=top_k_goal)
		
		return goal_pred, goal_gt, traj_pred_dict, traj_gt

	def save(self, filename):
		try:
			self.goal_model.model.save_weights('%s_goalcnnw.h5' % filename)
			self.traj_model.model.save_weights('%s_trajcnnw.h5' % filename)
		except Exception as e:
			print(e)

	def load(self, filename):
		try:
			self.goal_model.model.load_weights('%s_goalcnnw.h5' % filename)
			self.traj_model.model.load_weights('%s_trajcnnw.h5' % filename)
		except Exception as e:
			print(e)

class GoalCNNLSTM(object):
	"""This LSTM predicts the goal given trajectory/image inputs."""
	def __init__(self, traj_input_shape, goal_input_shape,  image_input_shape, n_outputs, beta, gamma, hidden_dim=100, img_feature_dim=32):
		self.beta       = beta   # hyperparameter for maximum entropy
		self.gamma      = gamma  # hyperparameter for penalty for predicting occupied spots
		self.model  = self._create_model(traj_input_shape, goal_input_shape, image_input_shape, hidden_dim, n_outputs, img_feature_dim)
		self.trained = False

		''' Debug '''
		#plot_model(self.model, to_file='goal_model.png', show_shapes=True)
		#print(self.model.summary())

	def goal_loss(self, occupancy):
		beta = self.beta
		gamma = self.gamma

		def loss(y_true, y_pred):
			# standard cross-entropy
			loss1 = K.categorical_crossentropy(y_true, y_pred)
			# max_ent (-loss2 used)
			loss2 = K.categorical_crossentropy(y_pred, y_pred)
			# occupancy penalty (note 0 = occupied)
			loss3 = K.sum(  K.relu( y_pred[:,:32] - K.reshape(occupancy, (K.shape(occupancy)[0], 32, 3))[:,:,2] ), axis = 1  )

			return loss1 - beta * loss2 + gamma * loss3

		return loss

	def top_k_acc(self, y_true, y_pred, k=3):
		# We call this top-n accuracy in the paper, basically accuracy if you get k tries to pick the label based on highest probability guesses.
		return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

	def _create_model(self, traj_input_shape, goal_input_shape, image_input_shape, hidden_dim, n_outputs, img_feature_dim):
		# Input for previous trajectory information.
		traj_hist_input = Input(shape=(traj_input_shape),name="input_trajectory")

		# Input for goals (occupancy).
		goals_input = Input(shape=(goal_input_shape),name="goal_input")

		# Image input.
		img_hist_input  = Input(shape=(image_input_shape),name="image_history")
		cnn_base = cnn_base_network(image_input_shape[1:], img_feature_dim)
		img_hist_features = TimeDistributed(cnn_base)(img_hist_input)
		lstm_cnn_inp = concatenate([traj_hist_input,img_hist_features])
		# -----------------------------
		
		# LSTM unit
		lstm = LSTM(hidden_dim,return_state=True,name="lstm_unit")

		# LSTM outputs
		lstm_outputs, state_h, state_c = lstm(lstm_cnn_inp)

		# Merge inputs with LSTM features
		concat_input = concatenate([goals_input, lstm_outputs],name="stacked_input")

		concat_output = Dense(100, activation="relu", name="concat_relu")(concat_input)

		# Final FC layer with a softmax activation
		goal_pred_output = Dense(n_outputs,activation="softmax",name="goal_output")(concat_output)

		# Create final model
		model = Model([traj_hist_input, goals_input, img_hist_input], goal_pred_output)
		
		# Compile model using loss
		model.compile(loss=self.goal_loss(goals_input), 
			optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=10.), metrics=[self.top_k_acc])
		self.init_weights = model.get_weights()

		return model

	def _reset(self):
		self.model.set_weights(self.init_weights)

	def fit_model(self, train_set, val_set, num_epochs=100, batch_size = 64, verbose=0):
		# NOTE: model weights are reset to same initialization each time fit is called.
		self._reset() 
		dataset = tf.data.TFRecordDataset(train_set)
		dataset = dataset.map(_parse_function)
		dataset = dataset.shuffle(10*batch_size, reshuffle_each_iteration=True)
		dataset = dataset.batch(batch_size)	
		dataset = dataset.prefetch(2)	

		for epoch in range(num_epochs):
			losses = []
			for image, feature, label, goal in dataset:
				feature = feature[:,:,:3] # only pose information from motion history used.

				goal = tf.reshape(goal,(-1,goal.shape[1]*goal.shape[2])) # occupancy

			    # All intention labels, with shape (batch_size, goal_nums)
				goal_idx = label[:, 0, -1]
	
			    # Convert to one-hot and the last one is undecided (-1)
				one_hot_goal = to_categorical(goal_idx, num_classes=33) # ground truth goal label
		
				train_data = [feature, goal, image/255] # image normalization done here.

				batch_loss = self.model.train_on_batch(
					      train_data,
					      one_hot_goal,
					      reset_metrics=True)

				losses.append(batch_loss)
			if verbose:
				print('\tGoal Epoch %d, Loss %f' % (epoch, np.mean(losses)))

		self.trained = True

	def save_model(self, file_name):
		if not self.trained:
			raise AttributeError("Run fit_model first.")
		self.model.save(file_name)
		print('Saved model %s' % file_name)

	def load(self, file_name):
		self.model = load_model(file_name, custom_objects={'goal_loss': self.goal_loss, 'top_k_acc': self.top_k_acc})
		print('Loaded model from %s' % file_name)
		
	def predict(self, test_set, batch_size=1):
		dataset = tf.data.TFRecordDataset(test_set)
		dataset = dataset.map(_parse_function)
		dataset = dataset.batch(batch_size)

		goal_pred = None
		goal_gt =  None # goal index ground truth

		for image, feature, label, goal in dataset:
			feature = feature[:,:,:3]

			goal = tf.reshape(goal,(-1,goal.shape[1]*goal.shape[2]))
	
			test_data = [feature, goal, image/255]
			
			goal_idx = label[:,0, -1]
    	
			if goal_pred is None:
				goal_pred = self.model.predict(test_data,steps = 1)
				goal_gt = to_categorical(goal_idx, num_classes=33)
			else:
				goal_pred = np.concatenate((goal_pred,self.model.predict(test_data,steps=1)),axis=0)
				goal_gt = np.concatenate((goal_gt,to_categorical(goal_idx, num_classes=33)),axis=0)

		return goal_pred, goal_gt

class TrajCNNLSTM(object):
	"""This LSTM generates trajectory predictions conditioned on a goal location."""
	def __init__(self, traj_input_shape, intent_input_shape, image_input_shape, future_horizon, future_dim, use_goal_info=True, hidden_dim=100, img_feature_dim=32):
		self.use_goal_info = use_goal_info # if true, use intent label in training/prediction.  Else learn on zeroed intent label (i.e. motion information only).
		self.model = self._create_model(traj_input_shape, intent_input_shape, image_input_shape, hidden_dim, future_horizon, future_dim, img_feature_dim)
		self.trained = False
		
		''' Debug '''
		#plot_model(self.model,to_file='traj_model.png', show_shapes=True)
		#print(self.model.summary())

	def _create_model(self, traj_input_shape, intent_input_shape, image_input_shape, hidden_dim, future_horizon, future_dim, img_feature_dim):
		# Input for previous trajectory information.
		traj_hist_input = Input(shape=(traj_input_shape),name="trajectory_input")

		# Input for goal intention
		# This can be ground_truth (gt) or predicted (multimodal top-k).
		intent_input = Input(shape=(intent_input_shape),name="intent_input")

		# Image input.
		img_hist_input  = Input(shape=(image_input_shape),name="image_history")
		cnn_base = cnn_base_network(image_input_shape[1:], img_feature_dim)
		img_hist_features = TimeDistributed(cnn_base)(img_hist_input)
		lstm_cnn_inp = concatenate([traj_hist_input,img_hist_features])
		# -----------------------------

		# LSTM unit
		lstm = LSTM(hidden_dim,return_state=True,name="lstm_unit")

		# LSTM outputs
		lstm_outputs, state_h, state_c = lstm(lstm_cnn_inp)
		encoder_states = [state_h,state_c]

		# Repeat the intent inputs
		intent_repeated= RepeatVector(future_horizon)(intent_input)

		# Define decoder
		decoder = LSTM(hidden_dim,return_sequences=True, return_state=True)

		# Decoder outputs, initialize with previous lstm states
		decoder_outputs,_,_ = decoder(intent_repeated,initial_state=encoder_states)

		# Shape to a time series prediction of future_horizon x features
		decoder_fully_connected = TimeDistributed(Dense(future_dim))(decoder_outputs)

		# Create final model
		model = Model([traj_hist_input,intent_input,img_hist_input], decoder_fully_connected)

		# Compile model using loss
		model.compile(loss='mean_squared_error', 
			optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=10.), metrics=['accuracy'])

		self.init_weights = model.get_weights()

		return model

	def _reset(self):
		self.model.set_weights(self.init_weights)

	def fit_model(self, train_set, val_set, num_epochs=100, batch_size=64,verbose=0):
		# NOTE: model weights are reset to same initialization each time fit is called.
		self._reset()
		dataset = tf.data.TFRecordDataset(train_set)
		dataset = dataset.map(_parse_function)
		dataset = dataset.shuffle(10*batch_size, reshuffle_each_iteration=True)
		dataset = dataset.batch(batch_size)		
		dataset = dataset.prefetch(2)

		for epoch in range(num_epochs):
			losses = []
			for image, feature, label, goal in dataset:
				
				feature = feature[:,:,:3] # only pose information from motion history used.
				
				# All intention labels, with shape (batch_size, goal_nums)
				goal_idx = label[:, 0, -1]

				# Convert to one-hot and the last one is undecided (-1)
				one_hot_goal = to_categorical(goal_idx, num_classes=33) # ground truth goal label

				if not self.use_goal_info:
					one_hot_goal = np.zeros_like(one_hot_goal) # intent distribution not used

				train_data = [feature,one_hot_goal,image/255] # image normalization done here.
				
				batch_loss = self.model.train_on_batch(
					train_data,
					label[:,:,:2],
					reset_metrics=True)

				losses.append(batch_loss)
			if verbose:
				print('\tTraj Epoch %d, Loss %f' % (epoch, np.mean(losses)))

		self.trained = True


	def save_model(self, file_name):
		if not self.trained:
			raise AttributeError("Run fit_model first.")
		self.model.save(file_name)
		print('Saved model %s' % file_name)

	def load(self, file_name):
		self.model = load_model(file_name)
		print('Loaded model from %s' % file_name)
		
	def predict(self, test_set,top_idxs,top_k_goal=[]):
		dataset = tf.data.TFRecordDataset(test_set)
		dataset = dataset.map(_parse_function)
		dataset = dataset.batch(1)

		# Dictionary returned with the key equal to the k in top_k_goal.
		# If unimodal, only a single key = 0 is given.
		# The value is a trajectory rollout predicted by the model.
		traj_predict_dict = dict()

		# Ground truth trajectory for comparison.
		traj_gt = []
		
		if not top_k_goal:
			# Just use the ground truth intent (or zeroed intent) for prediction.
			traj_predict_dict[0] = []

			for image, feature, label, goal in dataset:

				traj_gt.append(label[0,:,:-1].numpy())

				# Convert to one-hot and the last one is undecided (-1)
				goal_idx = label[:, 0, -1]
				one_hot_goal = to_categorical(goal_idx, num_classes=33)
				
				if not self.use_goal_info:
					one_hot_goal = np.zeros_like(one_hot_goal)

				test_data = [feature[:,:,:3], one_hot_goal, image/255]
				
				traj_predict_dict[0].append(self.model.predict(test_data, steps=1)[0,:,:])

			traj_predict_dict[0] = np.array(traj_predict_dict[0])
		
		else:
			# Multimodal with top_k_goal intent selections.
			for k_ind, k in enumerate(top_k_goal):
				traj_predict_dict[k] = []
				
				kth_idx = top_idxs[:, -1-k]
				one_hot_goals = np.zeros_like(top_idxs)
				for row, col in enumerate(kth_idx):
					one_hot_goals[row, col] = 1
				one_hot_goals = np.expand_dims(one_hot_goals,axis=0)
				
				instance_ind = 0
				for image, feature, label, goal in dataset:
					if k_ind == 0:		
						traj_gt.append(label[0,:,:-1].numpy())


					test_data = [feature[:,:,:3], one_hot_goals[:,instance_ind,:], image/255]
					
					traj_predict_dict[k].append(self.model.predict(test_data, steps=1)[0,:,:])
										
					instance_ind += 1
				
				traj_predict_dict[k] = np.array(traj_predict_dict[k])

		return traj_predict_dict, np.array(traj_gt)