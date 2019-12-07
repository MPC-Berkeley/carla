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

from datetime import datetime

class GoalLSTM(object):
	"""docstring for GoalLSTM"""
	def __init__(self, history_traj_data, goals_position, one_hot_goal, hidden_dim=100, beta=0.1):
		super(GoalLSTM, self).__init__()
		self.history_traj_data = history_traj_data
		self.goals_position    = goals_position
		self.one_hot_goal      = one_hot_goal

		self.hidden_dim = hidden_dim
		self.beta       = beta

		self.model      = None
		self.history    = None

		traj_input_shape =(self.history_traj_data.shape[1], self.history_traj_data.shape[2])
		goal_input_shape =(self.goals_position.shape[1],)
		n_outputs = self.one_hot_goal.shape[1]
		self.model = self.create_model(traj_input_shape, goal_input_shape, self.hidden_dim, n_outputs)
		plot_model(self.model, to_file='goal_model.png')
		print(self.model.summary())

	def max_ent_loss(self, y_true, y_pred):
		loss1 = K.categorical_crossentropy(y_true, y_pred)
		loss2 = K.categorical_crossentropy(y_pred, y_pred)
		loss  = loss1 + self.beta * loss2
		return loss

	def top_k_acc(self, y_true, y_pred, k=3):
		return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

	def create_model(self, traj_input_shape, goal_input_shape, hidden_dim, n_outputs):
		# Input to lstm
		lstm_input = Input(shape=(traj_input_shape),name="input_trajectory")

		# LSTM unit
		lstm = LSTM(hidden_dim,return_state=True,name="lstm_unit")

		# LSTM outputs
		lstm_outputs, state_h, state_c = lstm(lstm_input)

		# Input for goals
		goals_input = Input(shape=(goal_input_shape),name="goal_input")

		# Merge inputs with LSTM features
		concat_input = concatenate([goals_input,lstm_outputs],name="stacked_input")

		concat_output = Dense(100, activation="relu", name="concat_relu")(concat_input)

		# Final FC layer with a softmax activation
		goal_output = Dense(n_outputs,activation="softmax",name="goal_output")(concat_output)
		    
		# Create final model
		model = Model([lstm_input,goals_input], goal_output)

		# Compile model using loss
		#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.compile(loss=self.max_ent_loss, optimizer='adam', metrics=[self.top_k_acc])

		return model

	def fit_model(self, num_epochs=100):
		self.history = self.model.fit(\
					[self.history_traj_data, self.goals_position], \
					self.one_hot_goal, \
					epochs=num_epochs, validation_split=0.33)

	def plot_history(self):
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

	def save_model(self):
		now = datetime.now()
		dt_string = now.strftime('%m_%d_%H_%M')
		file_name = "goal_model_%.4f_%s.h5" % (self.goal_history.history['val_top_k_acc'][-1], dt_string)
		self.model.save(file_name)
		print("Saved goal model to disk")

	def predict(self, test_hist_traj, test_goals_position):
		goal_pred = self.model.predict([test_hist_traj, test_goals_position])
		return goal_pred

class TrajLSTM(object):
	"""docstring for TrajLSTM"""
	def __init__(self, history_traj_data, one_hot_goal, future_traj_data, hidden_dim=100):
		super(TrajLSTM, self).__init__()
		self.history_traj_data = history_traj_data
		self.one_hot_goal      = one_hot_goal
		self.future_traj_data  = future_traj_data

		self.hidden_dim = hidden_dim

		self.model      = None
		self.history    = None

		traj_input_shape=(history_traj_data.shape[1], history_traj_data.shape[2])
		intent_input_shape=(one_hot_goal.shape[1],)

		future_horizon = future_traj_data.shape[1]
		future_dim = future_traj_data.shape[2]
		self.model = self.create_model(traj_input_shape, intent_input_shape, self.hidden_dim, future_horizon, future_dim)
		plot_model(self.model,to_file='traj_model.png')
		print(self.model.summary())

	def create_model(self, traj_input_shape, intent_input_shape, hidden_dim, future_horizon, future_dim):

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

		return model

	def fit_model(self, num_epochs=100):
		self.history = self.model.fit(\
					[self.history_traj_data, self.one_hot_goal], \
					self.future_traj_data, \
					epochs=num_epochs, validation_split=0.33)

	def plot_history(self):
		plt.plot(self.history.history['acc'])
		plt.plot(self.history.history['val_acc'])
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

	def save_model(self):
		now = datetime.now()
		dt_string = now.strftime('%m_%d_%H_%M')
		file_name = "traj_model_%.4f_%s.h5" % (self.history.history['val_acc'][-1], dt_string)
		self.model.save(file_name)
		print("Saved traj model to disk")

	def predict(self, test_hist_traj, test_one_hot_goal):
		traj_pred = self.model.predict([test_hist_traj, test_one_hot_goal])
		return traj_pred