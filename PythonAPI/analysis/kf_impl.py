import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle, Ellipse
from utils import fix_angle
import pickle
import datetime
import scipy.spatial as ssp
import tensorflow as tf
from tfrecord_utils import _parse_function
from keras.utils import to_categorical

# This code provides the implementation of the EKF with constant velocity used for intent/goal and trajectory prediction.

class EKF_CV_MODEL(object):
	def __init__(self, 
				 x_init = np.zeros(5),                       # initial state guess
				 P_init = np.eye(5),                         # initial state covariance guess
				 Q = np.diag([0.1, 0.1, 0.01, 1., 0.1]),     # disturbance covariance
				 R = np.diag([1e-3]*3),               	     # measurement covariance
				 dt = 0.1                                    # model discretization time
				 ):

		self.nx = 5 # state: [x, y, psi, v, omega]
		self.nz = 3 # obs: [x ,y, psi]

		if Q.shape != (self.nx,self.nx):
			raise ValueError('Q should be a %d by %d matrix' % (self.nx, self.nx) )
		if R.shape != (self.nz, self.nz):
			raise ValueError('R should be a %d by %d matrix' % (self.nz, self.nz))
		if x_init.shape != (self.nx,):
			raise ValueError('x_init should be a %d vector' % (self.nx))
		if P_init.shape != (self.nx, self.nx):
			raise ValueError('P_init should be a %d by %d matrix' % (self.nx, self.nx))

		# initial state and covariance estimate
		self.x = x_init 
		self.P = P_init

		# process noise
		self.Q = Q

		# measurement noise
		self.R = R

		# measurement model (fixed)
		self.H = np.array([[1,0,0,0,0], \
						   [0,1,0,0,0], \
						   [0,0,1,0,0]])
		self.dt = dt

	def save(self, filename):
		kf_dict = {}
		kf_dict['Q'] =  self.Q
		kf_dict['R'] =  self.R
		kf_dict['dt'] = self.dt
		filename += '.pkl'
		pickle.dump(kf_dict, open(filename, 'wb'))

	def load(self, filename):
		kf_dict = pickle.load(open(filename, 'rb'))
		self.Q  = kf_dict['Q']
		self.R  = kf_dict['R']
		self.dt = kf_dict['dt']

	def fit(self, train_set):
		tset = EKF_CV_MODEL._extract_dict_from_tfrecords(train_set)
		self.Q = EKF_CV_MODEL._identify_Q_train(tset, self.dt)

		''' Debug '''
		#np.set_printoptions(precision=2)
		#print('Q identified as: ', self.Q)

	def predict(self, test_set):
		tset = EKF_CV_MODEL._extract_dict_from_tfrecords(test_set)

		# sort of inverted version compared to LSTM method
		# predict trajectory with the EKF then try to guess the goal.
		N_pred = tset['future_traj_data'][0].shape[0]
		traj_pred = self.traj_prediction(tset['history_traj_data'], N_pred)
		goal_pred = self.goal_prediction(traj_pred, tset['goal_position'])

		# unimodal trajectory prediction
		traj_pred_dict = {0: traj_pred}

		# return goal prediction, goal ground truth, 
		#        trajectory prediction, trajectory ground truth
		return goal_pred, tset['goal_ground_truth'], \
		       traj_pred_dict, tset['future_traj_data']

	def time_update(self):
		self.x = EKF_CV_MODEL._f_cv(self.x, self.dt)
		self.x[2] = fix_angle(self.x[2])
		A = EKF_CV_MODEL._f_cv_jacobian(self.x, self.dt)
		self.P =  A @ self.P @ A.T + self.Q

	def measurement_update(self, measurement): 
		inv_term = self.H @ self.P @ self.H.T + self.R
		K = self.P @ self.H.T @ np.linalg.pinv(inv_term)
		self.x = self.x + K @ (measurement - self.H @ self.x)
		self.x[2] = fix_angle(self.x[2])
		self.P = (np.eye(self.nx) - K @ self.H) @ self.P

	def get_x(self):
		return self.x.copy()

	def get_P(self):
		return self.P.copy()

	def traj_prediction(self, x_hists, N_pred, position_only=True):
		traj_pred = []
		for x_hist in x_hists:	
			# First init the KF with x_hist[0] for pose only.
			self.x = x_hist[0,:]
			self.x[3:] = 0. # velocity unknown, set to 0. WLOG
			self.P = np.eye(5) # may adjust this to be state dependent

			for i in range(1, x_hist.shape[0]):
				self.time_update()
				self.measurement_update(x_hist[i,:3]) # only pose
				
			# Then run predict for N_pred steps (extrapolation).
			x_pred = []; P_pred = []
			for i in range(N_pred):
				self.time_update()
				x_pred.append(self.get_x())
				P_pred.append(self.get_P())
			
			traj_pred.append(x_pred)
			# P_pred not used for now, could use for a pseudo-stochastic reachable set.

		traj_pred = np.array(traj_pred).reshape(x_hists.shape[0], N_pred, self.nx)

		if position_only:
			return traj_pred[:,:,:2] # just xy trajectory
		else:
			return traj_pred         # full state trajectory

	def goal_prediction(self, x_preds, goals, max_dist_thresh=20.):
		# Implements inverse Euclidean distance goal prediction.
		# max_dist_thresh used to throw out spots that are very far away
		# and put that into the "keep going" state.
		num_goals = int(goals.shape[1]) 
		goal_pred = np.zeros( (len(x_preds), num_goals + 1) )		

		for i, (x_pred, goal) in enumerate(zip(x_preds, goals)):
			inds_free = np.ravel( np.argwhere(goal[:,2] > 0.) ) # check occupancy to get free spots
			xy_pred_final = x_pred[-1,:2].reshape(1,2) # final position estimated in future trajectory
			xy_free = goal[inds_free,:2] # xy position of free spots identified
			dist = np.ravel( ssp.distance_matrix(xy_free, xy_pred_final) ) # Euclidean distance to free spots
			inv_dist = 1. / np.array([max(x, 0.01) for x in dist]) # inverse distance with thresh to avoid divide-by-0
			prob_intent = inv_dist / np.sum(inv_dist) # normalizing to get a prob. dist.
			prob_keep_going = 0 # prob of an undetermined/keep going state (used here for far away goals)

			for j in range(len(dist)):
				if dist[j] > max_dist_thresh:
					# if the goal is too far, move prob. mass to prob_keep_going
					prob_keep_going += prob_intent[j]
					prob_intent[j] = 0.

			# final predicted distribution passed to goal prediction result.
			goal_pred[i, inds_free] = prob_intent
			goal_pred[i, -1] = prob_keep_going

		return goal_pred

	@staticmethod
	def _extract_dict_from_tfrecords(files):
		# Given a set of tfrecord files, assembles
		# an aggregated dictionary for easier analysis.
		# Note: this isn't practical if the dataset is extremely large.
		tset = {}
		tset['history_traj_data'] = []
		tset['future_traj_data']  = []
		tset['goal_position']     = []
		tset['goal_ground_truth'] = []

		# Works with TF 2.0.  Else may need an iterator.
		dataset = tf.data.TFRecordDataset(files)
		dataset = dataset.map(_parse_function)
		
		for _, feature, label, goal in dataset:
			# Build up the data incrementally from tfrecords.
			# We don't need the image for the EKF.
			tset['history_traj_data'].append(feature.numpy())
			tset['future_traj_data'].append(label.numpy()[:,:5])
			tset['goal_position'].append(goal)
			
			goal_idx = label[0, -1]
			# Convert to one-hot and the last one is undecided (-1)
			tset['goal_ground_truth'].append(to_categorical(goal_idx, num_classes=33))

		tset['history_traj_data'] = np.array(tset['history_traj_data'])
		tset['future_traj_data']  = np.array(tset['future_traj_data'])
		tset['goal_position']     = np.array(tset['goal_position'])
		tset['goal_ground_truth'] = np.array(tset['goal_ground_truth']) 

		return tset

	@staticmethod
	def _f_cv(x, dt):
		# Motion model
		x_new = np.zeros(5)
		x_new[0] = x[0] + x[3] * np.cos(x[2]) * dt  #x
		x_new[1] = x[1] + x[3] * np.sin(x[2]) * dt  #y
		x_new[2] = x[2] + x[4] * dt                 #theta
		x_new[3] = x[3]                             #v
		x_new[4] = x[4]                             #omega
		return x_new
	
	@staticmethod
	def _f_cv_jacobian(x, dt):
		# Jacobian of motion model, computed by hand.
		v = x[3] 
		th = x[2]
		return np.array([[1, 0, -v*np.sin(th)*dt, np.cos(th)*dt,  0], \
						 [0, 1,  v*np.cos(th)*dt, np.sin(th)*dt,  0], \
						 [0, 0,                1,             0, dt], \
						 [0, 0,                0,             1,  0], \
						 [0, 0,                0,             0,  1]
						 ]).astype(np.float)
	@staticmethod
	def _f_cv_num_jacobian(x, dt, eps=1e-8):
		# Jacobian of motion model with finite differences (for gradient checking).
		nx = len(x)
		jac = np.zeros([nx,nx])
		for i in range(nx):
			xplus = x + eps * np.array([int(ind==i) for ind in range(nx)])
			xminus = x - eps * np.array([int(ind==i) for ind in range(nx)])

			f_plus  = EKF_CV_MODEL._f_cv(xplus, dt)
			f_minus = EKF_CV_MODEL._f_cv(xminus, dt)

			jac[:,i] = (f_plus - f_minus) / (2.*eps)
		return jac 

	@staticmethod
	def _identify_Q_sequence(x_sequence, dt=0.1, nx=5):
		# Given a sequence of states, estimate disturbance covariance Q.  Not really used.
		Q_est = np.zeros((nx,nx))
		N = len(x_sequence)
		for i in range(1, N):
			x_curr = x_sequence[i-1]
			x_next = x_sequence[i]
			x_next_model = EKF_CV_MODEL._f_cv(x_curr, dt)
			w = x_next - x_next_model
			Q_est += 1./N * np.outer(w,w)
		return Q_est

	@staticmethod
	def _identify_Q_train(train_set, dt=0.1):
		# "Model fit" for the EKF, estimate disturbance covariance from the train set.
		
		# number of training instances
		N_train = train_set['history_traj_data'].shape[0]

		# number of discrete timesteps in one training instance - 1 (i.e. how many "diffs")
		N_steps = train_set['history_traj_data'].shape[1] + train_set['future_traj_data'].shape[1] - 1
		
		# number of disturbance measurements from "diffs"
		N_Q_fit = N_train * N_steps

		# state dimension
		nx = train_set['history_traj_data'].shape[2]

		Q_est = np.zeros((nx, nx))

		for i in range(N_train):
			traj_full = np.concatenate((train_set['history_traj_data'][i], \
										train_set['future_traj_data'][i]), \
										axis = 0)
			for j in range(1, traj_full.shape[0] - 1):
				x_curr = traj_full[j-1]
				x_next = traj_full[j]
				x_next_model = EKF_CV_MODEL._f_cv(x_curr, dt)
				w = x_next - x_next_model
				w[2] = fix_angle(w[2])
				Q_est += 1./N_Q_fit * np.outer(w,w)

		return Q_est