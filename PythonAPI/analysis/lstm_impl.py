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
from keras.layers import Dense, Dropout, Softmax, Flatten, concatenate
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
# import tensorflow as tf
import glob
from datetime import datetime

class CombinedLSTM(object):
	def __init__(self, history_shape, goals_position_shape, one_hot_goal_shape, future_shape, hidden_dim, beta=0.1, gamma=0.1, use_goal_info=True):
		traj_input_shape    = (history_shape[1], history_shape[2])
		goal_input_shape    = (goals_position_shape[1],)
		n_outputs           = one_hot_goal_shape[1]
		intent_input_shape  = (n_outputs,)
		future_horizon      = future_shape[1]
		future_dim	        = future_shape[2]

		self.goal_model = GoalLSTM(traj_input_shape, goal_input_shape, n_outputs, beta, gamma, hidden_dim=hidden_dim)
		self.traj_model = TrajLSTM(traj_input_shape, intent_input_shape, future_horizon, future_dim, hidden_dim=hidden_dim)

		self.use_goal_info = use_goal_info

	def fit(self, train_set, val_set, verbose=0):
		self.goal_model.fit_model(train_set, val_set, num_epochs=100, verbose=verbose)
		self.traj_model.fit_model(train_set, val_set, num_epochs=100, verbose=verbose, use_goal_info=self.use_goal_info)

	def predict(self, test_set, top_k_goal=[]):
		goal_pred = self.goal_model.predict(test_set)

		# TODO: how to cleanly do multimodal predictions here.  Maybe we can't cleanly just pass a test set, or need to add
		# a new field to the dictionary with top k goal predictions and loop in the predict function.
		traj_pred_dict = dict()
		traj_test_set = test_set.copy()

		# If don't want the goal
		if top_k_goal == None or self.use_goal_info == False:
			# Set others to be zeros while keep the argmax to be 1.
			traj_test_set['one_hot_goal'] = np.zeros_like(traj_test_set['one_hot_goal'])
			traj_pred = self.traj_model.predict(traj_test_set)
			traj_pred_dict[0] = traj_pred
		# If using the ground truth goal
		elif top_k_goal == []:
			traj_pred = self.traj_model.predict(traj_test_set)
			traj_pred_dict[0] = traj_pred
		else:
			top_idxs = np.argsort(goal_pred, axis=1)
			for k in top_k_goal:
				kth_idx = top_idxs[:, -1-k]
				# Set others to be zeros while keep the argmax to be 1.
				traj_test_set['one_hot_goal'] = np.zeros_like(traj_test_set['one_hot_goal'])
				for row, col in enumerate(kth_idx):
					traj_test_set['one_hot_goal'][row, col] = 1.

				traj_pred = self.traj_model.predict(traj_test_set)
				traj_pred_dict[k] = traj_pred

		return goal_pred, traj_pred_dict

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
	def __init__(self, traj_input_shape, goal_input_shape, n_outputs, beta, gamma, hidden_dim=100):
		self.beta       = beta
		self.gamma      = gamma
		self.history    = None
		self.model  = self._create_model(traj_input_shape, goal_input_shape, hidden_dim, n_outputs)

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

	def _create_model(self, traj_input_shape, goal_input_shape, hidden_dim, n_outputs):
		# Input to lstm
		lstm_input = Input(shape=(traj_input_shape),name="input_trajectory")

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
		model = Model([lstm_input, goals_input], goal_output)

		# Compile model using loss
		#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.compile(loss=self.goal_loss(goals_input), optimizer='adam', metrics=[self.top_k_acc])

		self.init_weights = model.get_weights()

		return model

	def _reset(self): 
		self.model.set_weights(self.init_weights)

	def fit_model(self, train_set, val_set, num_epochs=100, verbose=0):
		val_data = ([val_set['history_traj_data'][:,:,:3], val_set['goal_position']], 
					 val_set['one_hot_goal'])

		self._reset()
		self.history = self.model.fit(
					[train_set['history_traj_data'][:,:,:3], train_set['goal_position']], 
					train_set['one_hot_goal'], 
					epochs=num_epochs, 
					validation_data=val_data,
					verbose=verbose)
		if verbose:
			self.plot_history()

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

	def predict(self, test_set):
		goal_pred = self.model.predict([test_set['history_traj_data'][:,:,:3], test_set['goal_position']])
		return goal_pred

class TrajLSTM(object):
	"""docstring for TrajLSTM"""
	def __init__(self, traj_input_shape, intent_input_shape, future_horizon, future_dim, hidden_dim=100):
		self.history    = None

		self.model = self._create_model(traj_input_shape, intent_input_shape, hidden_dim, future_horizon, future_dim)
		''' Debug '''
		# plot_model(self.model,to_file='traj_model.png')
		# print(self.model.summary())

	def _create_model(self, traj_input_shape, intent_input_shape, hidden_dim, future_horizon, future_dim):

		# Input to lstm
		lstm_input = Input(shape=(traj_input_shape),name="trajectory_input")

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
		model = Model([lstm_input,goals_input], decoder_fully_connected)

		# Compile model using loss
		model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

		self.init_weights = model.get_weights()

		return model

	def _reset(self): 
		self.model.set_weights(self.init_weights)

	def fit_model(self, train_set, val_set, num_epochs=100, verbose=0, use_goal_info=True):
		if use_goal_info:
			val_goal = val_set['one_hot_goal']
			train_goal = train_set['one_hot_goal']
		else:
			val_goal = np.zeros_like(val_set['one_hot_goal'])
			train_goal = np.zeros_like(train_set['one_hot_goal'])

		val_data   = ([val_set['history_traj_data'][:,:,:3], val_goal], 
					  val_set['future_traj_data'][:,:,:2])

		self._reset()
		self.history = self.model.fit(
					[train_set['history_traj_data'][:,:,:3], train_goal], 
					train_set['future_traj_data'][:,:,:2], 
					epochs=num_epochs, 
					validation_data=val_data, 
					verbose = verbose)
		if verbose:
			self.plot_history()

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

	def predict(self, test_set):
		# TODO: how to incorporate goal prediction
		traj_pred = self.model.predict([test_set['history_traj_data'][:,:,:3], test_set['one_hot_goal']])
		return traj_pred