import re
import numpy as np
import unidecode
from IPython import get_ipython;
#get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt
import pandas
import math
from keras import metrics
from keras import Input, Model
from keras.models import Sequential
from keras.models import load_model
# from keras.layers import *
from keras.layers import Dense, Dropout, Softmax, Flatten, concatenate, Conv2D
from keras.layers import Activation, TimeDistributed, RepeatVector, Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K # for custom loss function
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tfrecord_utils import read_tfrecord, read_gt_tfrecord,_parse_function
import tensorflow as tf
import glob
from datetime import datetime
from tqdm import tqdm
import time

class CombinedLSTM(object):
	def __init__(self, history_shape, goals_position_shape, image_input_shape, one_hot_goal_shape, future_shape, hidden_dim, beta=0.1, gamma=0.1, use_goal_info=True):
		traj_input_shape    = (history_shape[1], history_shape[2])
		goal_input_shape    = (goals_position_shape[1],)
		n_outputs           = one_hot_goal_shape[1]
		intent_input_shape  = (n_outputs,)
		future_horizon      = future_shape[1]
		future_dim	        = future_shape[2]

		self.use_goal_info = use_goal_info
		self.goal_model = GoalLSTM(traj_input_shape, goal_input_shape, image_input_shape, n_outputs, beta, gamma, hidden_dim=hidden_dim)
		self.traj_model = TrajLSTM(traj_input_shape, intent_input_shape, image_input_shape, future_horizon, future_dim, use_goal_info=self.use_goal_info, hidden_dim=hidden_dim)	

	def fit(self, train_set, val_set, num_epochs=100, batch_size=64, verbose=0, use_image = False):
		print('Fitting GoalLSTM')
		self.goal_model.fit_model(train_set, val_set, num_epochs=num_epochs, batch_size=batch_size, \
		                          verbose=verbose, use_image=use_image)
		print('Fitting TrajLSTM')
		self.traj_model.fit_model(train_set, val_set, num_epochs=num_epochs, batch_size=batch_size, \
			                      verbose=verbose,use_image=use_image)

	def predict(self, test_set, top_k_goal=[],use_image=False):
		print('\t goal_prediction at ', time.time())
		goal_pred, goal_gt = self.goal_model.predict(test_set,use_image=use_image)

		# TODO: how to cleanly do multimodal predictions here.  Maybe we can't cleanly just pass a test set, or need to add
		# a new field to the dictionary with top k goal predictions and loop in the predict function.
		top_idxs = np.argsort(goal_pred,axis=1)
		print('\t traj_prediction', time.time())
		traj_pred_dict, traj_gt = self.traj_model.predict(test_set,top_idxs,top_k_goal=top_k_goal,use_image=use_image)
		
		# Get ground truth here
		#print('\t ground_truth', time.time())
		#goal_gt, traj_gt = read_gt_tfrecord(test_set)

		return goal_pred, goal_gt, traj_pred_dict, traj_gt

	def save(self, filename):
		try:
			self.goal_model.model.save_weights('%s_goalw.h5' % filename)
			self.traj_model.model.save_weights('%s_trajw.h5' % filename)
		except Exception as e:
			print(e)

	def load(self, filename):
		try:
			self.goal_model.model.load_weights('%s_goalw.h5' % filename)
			self.traj_model.model.load_weights('%s_trajw.h5' % filename)
		except Exception as e:
			print(e)


class GoalLSTM(object):
	"""docstring for GoalLSTM"""
	def __init__(self, traj_input_shape, goal_input_shape,  image_input_shape, n_outputs, beta, gamma, hidden_dim=100):
		self.beta       = beta
		self.gamma      = gamma
		self.history    = None
		self.model  = self._create_model(traj_input_shape, goal_input_shape, image_input_shape, hidden_dim, n_outputs)

		''' Debug '''
		#plot_model(self.model, to_file='goal_model.png')
		#print(self.model.summary())

	def goal_loss(self, occupancy):
		beta = self.beta
		gamma = self.gamma

		def loss(y_true, y_pred):
			loss1 = K.categorical_crossentropy(y_true, y_pred)
			loss2 = K.categorical_crossentropy(y_pred, y_pred)
			loss3 = K.sum(  K.relu( y_pred[:,:32] - K.reshape(occupancy, (K.shape(occupancy)[0], 32, 3))[:,:,2] ), axis = 1  )

			return loss1 - beta * loss2 + gamma * loss3

		return loss

	def max_ent_loss(self, y_true, y_pred):
		loss1 = K.categorical_crossentropy(y_true, y_pred)
		loss2 = K.categorical_crossentropy(y_pred, y_pred)
		#entropy = -tf.math.reduce_sum(tf.math.multiply_no_nan(tf.math.log(y_pred), y_pred), axis = 1)

		# loss  = loss1 - self.beta * entropy
		loss = loss1 - self.beta * loss2
		# loss = -loss2
		return loss

	def top_k_acc(self, y_true, y_pred, k=3):
		return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

	def _create_model(self, traj_input_shape, goal_input_shape, image_input_shape, hidden_dim, n_outputs):


		# ---- DEFINE MODEL HERE ------
		cnn_input  = Input(shape=(image_input_shape),name="image_history")
		#cnn_layer = TimeDistributed(Conv2D(32, kernel_size=(3,3),activation='relu'))(cnn_input)
		#fl = TimeDistributed(Flatten())(cnn_layer)
		#cnn_out = TimeDistributed(Dense(10))(fl)
		# -----------------------------

		# Input to lstm
		lstm_input = Input(shape=(traj_input_shape),name="input_trajectory")
		
		#lstm_cnn_inp = concatenate([lstm_input,cnn_out])
		
		# LSTM unit
		lstm = LSTM(hidden_dim,return_state=True,name="lstm_unit")

		# LSTM outputs
		lstm_outputs, state_h, state_c = lstm(lstm_input)

		# Input for goals
		goals_input = Input(shape=(goal_input_shape),name="goal_input")

		# Merge inputs with LSTM features
		concat_input = concatenate([goals_input, lstm_outputs],name="stacked_input")

		concat_output = Dense(100, activation="relu", name="concat_relu")(concat_input)

		# Final FC layer with a softmax activation
		goal_output = Dense(n_outputs,activation="softmax",name="goal_output")(concat_output)

		# Create final model
		model = Model([lstm_input, goals_input,cnn_input], goal_output)

		# Compile model using loss
		#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.compile(loss=self.goal_loss(goals_input), optimizer='adam', metrics=[self.top_k_acc])

		self.init_weights = model.get_weights()

		return model

	def _reset(self):
		self.model.set_weights(self.init_weights)

	def fit_model(self, train_set, val_set, num_epochs=100, batch_size = 64,verbose=0,use_image=False):

		dataset = tf.data.TFRecordDataset(train_set)
		dataset = dataset.map(_parse_function)
		dataset = dataset.batch(batch_size)
		self._reset()

		for epoch in tqdm(range(num_epochs)):
			for image, feature, label, goal in dataset:
				feature = feature[:,:,:3]

				goal = tf.reshape(goal,(-1,goal.shape[1]*goal.shape[2]))
				#goal_val = tf.reshape(goal_val,(-1,goal_val.shape[1]*goal_val.shape[2]))

		    # All intention labels, with shape (batch_size, goal_nums)
				goal_idx = label[:, 0, -1]
				#goal_idx_val = label_val[:,0,-1]

		    # Convert to one-hot and the last one is undecided (-1)
				one_hot_goal = to_categorical(goal_idx, num_classes=33)
				#one_hot_goal_val = to_categorical(goal_idx_val, num_classes=33)

				if use_image:
					image = tf.zeros_like(image)
					#image_val = tf.zeros_like(image_val)
				
				train_data = [feature, goal,image]
				#val_data = ([feature_val, goal_val,image_val], one_hot_goal_val)

				self.model.fit(
						train_data,
	          one_hot_goal,
	          steps_per_epoch = 1,
						epochs=1,
						#validation_data=val_data,
						#validation_steps = count_val // batch_size,
						verbose=verbose)



		# image, feature, label, goal, count = read_tfrecord(train_set,cutting=True,batch_size=batch_size)
		# image_val, feature_val, label_val, goal_val, count_val = read_tfrecord(val_set,cutting=True,batch_size=batch_size)

		# goal = tf.reshape(goal,(-1,goal.shape[1]*goal.shape[2]))
		# goal_val = tf.reshape(goal_val,(-1,goal_val.shape[1]*goal_val.shape[2]))

  #   # All intention labels, with shape (batch_size, goal_nums)
		# goal_idx = label[:, 0, -1]
		# goal_idx_val = label_val[:,0,-1]

  #   # Convert to one-hot and the last one is undecided (-1)
		# one_hot_goal = to_categorical(goal_idx, num_classes=33)
		# one_hot_goal_val = to_categorical(goal_idx_val, num_classes=33)

		# if use_image:
		# 	image = tf.zeros_like(image)
		# 	image_val = tf.zeros_like(image_val)
		
		# train_data = [feature, goal,image]
		# val_data = ([feature_val, goal_val,image_val], one_hot_goal_val)

		# self._reset()
		# self.history = self.model.fit(
		# 			train_data,
  #         one_hot_goal,
  #         steps_per_epoch=count // batch_size,
		# 			epochs=num_epochs,
		# 			validation_data=val_data,
		# 			validation_steps = count_val // batch_size,
		# 			verbose=verbose)
		# if verbose:
		# 	self.plot_history()

	def plot_history(self):
		if not self.history:
			raise AttributeError("No history available.  Run fit_model.")

		plt.plot(self.history.history['top_k_acc'])
		plt.plot(self.history.history['val_top_k_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='lower right')
		plt.show()
		# summarize history for loss
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper right')
		plt.show()

	def save_model(self, file_name):
		if not self.history:
			raise AttributeError("No history available.  Run fit_model.")

		# now = datetime.now()
		# dt_string = now.strftime('%m_%d_%H_%M')
		# file_name = "./model/goal_model_%.4f_%s.h5" % (self.history.history['val__top_k_acc'][-1], dt_string)
		self.model.save(file_name)
		print('Saved model %s' % file_name)

	def load(self, file_name):
		# model_files_on_disk = glob.glob('./model/goal_model_*.h5')
		# model_files_on_disk.sort()
		# print('Goal Model files on disk: %s' % model_files_on_disk)
		# goal_model = load_model(model_files_on_disk[0], custom_objects={'_max_ent_loss': self._max_ent_loss, '_top_k_acc': self._top_k_acc})
		#self.model = load_model(file_name, custom_objects={'max_ent_loss': self.max_ent_loss, 'top_k_acc': self.top_k_acc})
		self.model = load_model(file_name, custom_objects={'goal_loss': self.goal_loss, 'top_k_acc': self.top_k_acc})
		print('Loaded model from %s' % file_name)
		# return goal_model

	def predict(self, test_set, batch_size=64,use_image=False):
		

		dataset = tf.data.TFRecordDataset(test_set)
		dataset = dataset.map(_parse_function)
		dataset = dataset.batch(batch_size)

		goal_pred = None
		goal_gt =  None # goal index ground truth

		for image, feature, label, goal in dataset:
			feature = feature[:,:,:3]

			goal = tf.reshape(goal,(-1,goal.shape[1]*goal.shape[2]))
			if not use_image:
				image = tf.zeros_like(image)
			test_data = [feature, goal, image]


			goal_idx = label[:,0, -1]
    	
			if goal_pred is None:
				goal_pred = self.model.predict(test_data,steps = 1)
				goal_gt = to_categorical(goal_idx, num_classes=33)
			else:
				goal_pred = np.concatenate((goal_pred,self.model.predict(test_data,steps=1)),axis=0)
				goal_gt = np.concatenate((goal_gt,to_categorical(goal_idx, num_classes=33)),axis=0)


		#image, feature, label, goal, count = read_tfrecord(test_set,cutting=True, repeat=False, shuffle=False,batch_size=1)
		#goal = tf.reshape(goal,(-1,goal.shape[1]*goal.shape[2]))

    # Convert to one-hot and the last one is undecided (-1)
		#if not use_image:
		#	image = tf.zeros_like(image)
		#test_data = [feature, goal, image]

		#goal_pred = self.model.predict(test_data, steps=count)
		return goal_pred, goal_gt

class TrajLSTM(object):
	"""docstring for TrajLSTM"""
	def __init__(self, traj_input_shape, intent_input_shape, image_input_shape, future_horizon, future_dim, use_goal_info=True, hidden_dim=100):
		self.history    = None
		self.use_goal_info = use_goal_info
		self.model = self._create_model(traj_input_shape, intent_input_shape, image_input_shape, hidden_dim, future_horizon, future_dim)
		''' Debug '''
		# plot_model(self.model,to_file='traj_model.png')
		# print(self.model.summary())

	def _create_model(self, traj_input_shape, intent_input_shape, image_input_shape, hidden_dim, future_horizon, future_dim):


		# ---- DEFINE MODEL HERE ------
		cnn_input  = Input(shape=(image_input_shape),name="image_history")
		#cnn_layer = TimeDistributed(Conv2D(32, kernel_size=(3,3),activation='relu'))(cnn_input)
		#fl = TimeDistributed(Flatten())(cnn_layer)
		#cnn_out = TimeDistributed(Dense(10))(fl)
		# -----------------------------

		# Input to lstm
		lstm_input = Input(shape=(traj_input_shape),name="trajectory_input")

		#lstm_cnn_input = concatenate([lstm_input,cnn_out])

		# LSTM unit
		lstm = LSTM(hidden_dim,return_state=True,name="lstm_unit")

		# LSTM outputs
		lstm_outputs, state_h, state_c = lstm(lstm_input)
		encoder_states = [state_h,state_c]

		# Input for goals
		goals_input = Input(shape=(intent_input_shape),name="goal_input")

		# Repeat the goal inputs
		goals_repeated= RepeatVector(future_horizon)(goals_input)

		# Define decoder
		decoder = LSTM(hidden_dim,return_sequences=True, return_state=True)

		# Decoder outputs, initialize with previous lstm states
		decoder_outputs,_,_ = decoder(goals_repeated,initial_state=encoder_states)

		# Shape to a time series prediction of future_horizon x features
		decoder_fully_connected = TimeDistributed(Dense(future_dim))(decoder_outputs)

		# Create final model
		model = Model([lstm_input,goals_input,cnn_input], decoder_fully_connected)

		# Compile model using loss
		model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

		self.init_weights = model.get_weights()

		return model

	def _reset(self):
		self.model.set_weights(self.init_weights)

	def fit_model(self, train_set, val_set, num_epochs=100, batch_size=64,verbose=0,use_image=False):

		dataset = tf.data.TFRecordDataset(train_set)
		dataset = dataset.map(_parse_function)
		dataset = dataset.batch(batch_size)
		self._reset()

		for epoch in tqdm(range(num_epochs)):
			for image, feature, label, goal in dataset:
				feature = feature[:,:,:3]

				goal = tf.reshape(goal,(-1,goal.shape[1]*goal.shape[2]))
				goal_idx = label[:, 0, -1]
				train_goal = to_categorical(goal_idx, num_classes=33)
				if not self.use_goal_info:
					train_goal = np.zeros_like(train_goal)
				if not use_image:
					image = tf.zeros_like(image)

				train_data = [feature, train_goal,image]

				self.model.fit(
					train_data,
					label[:,:,:2],
          steps_per_epoch = 1,
					epochs = 1,
					#validation_data=val_data,
					#validation_steps = count_val // batch_size,
					verbose = verbose)

		# image, feature, label, goal, count = read_tfrecord(train_set,cutting=True,batch_size=batch_size)
		# image_val, feature_val, label_val, goal_val, count_val = read_tfrecord(val_set,cutting=True,batch_size=batch_size)

		# goal = tf.reshape(goal,(-1,goal.shape[1]*goal.shape[2]))
		# goal_val = tf.reshape(goal_val,(-1,goal_val.shape[1]*goal_val.shape[2]))

  #   # All intention labels, with shape (batch_size, goal_nums)
		# goal_idx = label[:, 0, -1]
		# goal_idx_val = label_val[:, 0, -1]
  #   # Convert to one-hot and the last one is undecided (-1)
		# train_goal = to_categorical(goal_idx, num_classes=33)
		# train_goal_val = to_categorical(goal_idx_val, num_classes=33)

		# if not self.use_goal_info:
		# 	train_goal = np.zeros_like(train_goal)
		# 	train_goal_val = np.zeros_like(train_goal_val)

		# if not use_image:
		# 	image = tf.zeros_like(image)
		# 	image_val = tf.zeros_like(image_val)

		# train_data = [feature, train_goal,image]
		# val_data = [[feature_val, train_goal_val,image_val], label_val[:,:,:2]]

		# self._reset()
		# self.history = self.model.fit(
		# 			train_data,
		# 			label[:,:,:2],
  #         steps_per_epoch = count // batch_size,
		# 			epochs=num_epochs,
		# 			validation_data=val_data,
		# 			validation_steps = count_val // batch_size,
		# 			verbose = verbose)
		# if verbose:
		# 	self.plot_history()

	def plot_history(self):
		if not self.history:
			raise AttributeError("No history available.  Run fit_model.")

		plt.plot(self.history.history['accuracy'])
		plt.plot(self.history.history['val_accuracy'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='lower right')
		plt.show()
		# summarize history for loss
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper right')
		plt.show()

	def save_model(self, file_name):
		if not self.history:
			raise AttributeError("No history available.  Run fit_model.")

		# now = datetime.now()
		# dt_string = now.strftime('%m_%d_%H_%M')
		# file_name = "./model/traj_model_%.4f_%s.h5" % (self.history.history['val_acc'][-1], dt_string)
		self.model.save(file_name)
		print('Saved model %s' % file_name)

	def load(self, file_name):
		# model_files_on_disk = glob.glob('./model/traj_model_*.h5')
		# model_files_on_disk.sort()
		# print('Traj Model files on disk: %s' % model_files_on_disk)
		# traj_model = load_model(model_files_on_disk[0])
		self.model = load_model(file_name)
		print('Loaded model from %s' % file_name)
		#return traj_model

	def predict(self, test_set,top_idxs,top_k_goal=[],use_image=False):

		dataset = tf.data.TFRecordDataset(test_set)
		dataset = dataset.map(_parse_function)
		dataset = dataset.batch(1)
		traj_predict_dict = dict()
		traj_gt = []
		if not top_k_goal:
			traj_predict_dict[0] = []
			ind = 0

		#  for epoch in n_epochs: 
			for image, feature, label, goal in dataset:
				goal = tf.reshape(goal,(-1,goal.shape[1]*goal.shape[2]))
				# All intention labels, with shape (batch_size, goal_nums)
				goal_idx = label[:, 0, -1]
				traj_gt.append(label[0,:,:-1])
		    # Convert to one-hot and the last one is undecided (-1)
				test_goal = to_categorical(goal_idx, num_classes=33)
				if not use_image:
					image = tf.zeros_like(image)
				if not self.use_goal_info:
					test_goal = np.zeros_like(test_goal)
				
				test_data = [feature[:,:,:3].numpy(), test_goal,image.numpy()]
				traj_predict_dict[0].append(self.model.predict(test_data)[0,:,:])
				ind+=1
			traj_predict_dict[0] = np.array(traj_predict_dict[0])
		else:
			for k_ind, k in enumerate(top_k_goal):
				traj_predict_dict[k] = []
				kth_idx = top_idxs[:, -1-k]
				test_goal = np.zeros_like(top_idxs)
				for row, col in enumerate(kth_idx):
					test_goal[row, col] = 1
				test_goal = np.expand_dims(test_goal,axis=0)
				ind = 0
				for image, feature, label, goal in dataset:
					if not use_image:
						image = tf.zeros_like(image)
					test_data = [feature[:,:,:3].numpy(), test_goal[:,ind,:],image.numpy()]
					traj_predict_dict[k].append(self.model.predict(test_data)[0,:,:])
					if k_ind == 0:
						traj_gt.append(label[0,:,:-1])
					ind += 1
				traj_predict_dict[k] = np.array(traj_predict_dict[k])

		return traj_predict_dict, np.array(traj_gt)













		# image, feature, label, goal, count = read_tfrecord(test_set,cutting=True,batch_size=1)
		# goal = tf.reshape(goal,(-1,goal.shape[1]*goal.shape[2]))
  #   # All intention labels, with shape (batch_size, goal_nums)
		# goal_idx = label[:, 0, -1]
  #   # Convert to one-hot and the last one is undecided (-1)
		# test_goal = to_categorical(goal_idx, num_classes=33)

		# traj_pred_dict = dict()
		# if not use_image:
		# 	image = tf.zeros_like(image)
		# if not use_goal_info:
		# 	test_goal = tf.zeros_like(test_goal)

		# if not top_k_goal:
		# 	test_data = [feature, test_goal,image]
		# 	traj_pred_dict[0] = self.model.predict(test_data, steps = count)
		# else:
		# 	for k in top_k_goal:
		# 		kth_idx = top_idxs[:,-1-k]




		  #TODO: VIJAY
      # If don't want the goal
  		# if top_k_goal == None or self.use_goal_info == False:
  		# 	# Set others to be zeros while keep the argmax to be 1.
  		# 	traj_test_set['one_hot_goal'] = np.zeros_like(traj_test_set['one_hot_goal'])
  		# 	traj_pred = self.traj_model.predict(traj_test_set)
  		# 	traj_pred_dict[0] = traj_pred
  		# # If using the ground truth goal
  		# elif top_k_goal == []:
  		# 	traj_pred = self.traj_model.predict(traj_test_set)
  		# 	traj_pred_dict[0] = traj_pred
  		# else:
  		# 	top_idxs = np.argsort(goal_pred, axis=1)
  		# 	for k in top_k_goal:
  		# 		kth_idx = top_idxs[:, -1-k]
  		# 		# Set others to be zeros while keep the argmax to be 1.
  		# 		traj_test_set['one_hot_goal'] = np.zeros_like(traj_test_set['one_hot_goal'])
  		# 		for row, col in enumerate(kth_idx):
  		# 			traj_test_set['one_hot_goal'][row, col] = 1.
  		#self.model.predict(test_data, steps = count)
		#traj_pred = self.model.predict(test_data, steps = count)
		#return traj_pred
