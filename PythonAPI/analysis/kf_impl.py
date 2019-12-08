import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle, Ellipse
from utils import fix_angle
import pickle
import datetime
import pdb

'''
Improvements:
- variable input sizes (or better checking of arguments) for the filter
- base KF class implementation in case of multiple KF implementatations
'''

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

	def save(self):
		kf_dict = {}
		kf_dict['Q'] =  self.Q
		kf_dict['R'] =  self.R
		kf_dict['dt'] = self.dt

		now = datetime.now()
		dt_string = now.strftime('%m_%d_%H_%M')
		fname = "kf_%s.pkl" % dt_string
		pickle.dump(kf_dict, open(fname, 'wb'))

	def fit(self, train_set, val_set):
		self.Q = EKF_CV_MODEL._identify_Q_train(train_set, self.dt)

		''' Debug '''
		np.set_printoptions(precision=2)
		print('Q identified as: ', self.Q)

	def predict(self, test_set):
		# sort of inverted version compared to LSTM method
		N_pred = test_set['future_traj_data'][0].shape[0]
		traj_pred = self.traj_prediction(test_set['history_traj_data'], N_pred) # TODO
		goal_pred = self.goal_prediction(traj_pred, test_set['goal_position']) # TODO
		# TODO: clean up and fix analysis
		return goal_pred, traj_pred

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

	def traj_prediction(self, x_hists, N_pred):
		traj_pred = []
		for x_hist in x_hists:	
			# First init the KF with x_hist[0] for pose only.
			self.x = x_hist[0,:]
			self.x[3:] = 0. # velocity unknown, set to 0. WLOG
			self.P = np.eye(5) # may adjust this to be state dependent

			for i in range(1, x_hist.shape[0]):
				self.time_update()
				self.measurement_update(x_hist[i,:3]) # only pose
				
			# Then run predict for N_pred steps
			x_pred = []; P_pred = []
			for i in range(N_pred):
				self.time_update()
				x_pred.append(self.get_x())
				P_pred.append(self.get_P())
			
			traj_pred.append(np.array(x_pred))
			# P_pred not used for now.

		return traj_pred

	def goal_prediction(self, x_preds, goals):
		goal_pred = np.zeros( (len(x_preds), 33) )
		goal_pred[:,-1] = 1.
		'''
		for x_pred, goal_set in zip(x_preds, goals):
			pass
			# TODO
		'''
		return goal_pred

	@staticmethod
	def _f_cv(x, dt):
		x_new = np.zeros(5)
		x_new[0] = x[0] + x[3] * np.cos(x[2]) * dt  #x
		x_new[1] = x[1] + x[3] * np.sin(x[2]) * dt  #y
		x_new[2] = x[2] + x[4] * dt                 #theta
		x_new[3] = x[3]                             #v
		x_new[4] = x[4]                             #omega
		return x_new
	
	@staticmethod
	def _f_cv_jacobian(x, dt):
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



''' Code to be cleaned up.  WIP.
class EKF_CA_MODEL(object):
	def __init__(self, 
				 x_init = np.zeros(7),                               # initial state guess
				 P_init = np.eye(7),                                 # initial state covariance guess
				 Q = np.diag([0.1, 0.1, 0.01, 1., 0.1, 1., 0.1]),    # disturbance covariance
				 R = np.diag([1e-3]*3),               	             # measurement covariance
				 dt = 0.1                                            # model discretization time
				 ):

		self.nx = 7 # state: [x, y, psi, v, omega, v_dot, omega_dot]
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
		self.H = np.array([[1,0,0,0,0,0,0], \
						   [0,1,0,0,0,0,0], \
						   [0,0,1,0,0,0,0]])
		self.dt = dt

	def predict(self):
		self.x = EKF_CA_MODEL._f_ca(self.x, self.dt)
		A = EKF_CA_MODEL._f_ca_jacobian(self.x, self.dt)
		self.P =  A @ self.P @ A.T + self.Q

	def update(self, measurement): 
		inv_term = self.H @ self.P @ self.H.T + self.R
		K = self.P @ self.H.T @ np.linalg.pinv(inv_term)
		self.x = self.x + K @ (measurement - self.H @ self.x)
		self.P = (np.eye(self.nx) - K @ self.H) @ self.P

	def get_x(self):
		return self.x.copy()

	def get_P(self):
		return self.P.copy()

	@staticmethod
	def _f_ca(x, dt):
		x_new = np.zeros(7)
		x_new[0] = x[0] + x[3] * np.cos(x[2]) * dt  #x
		x_new[1] = x[1] + x[3] * np.sin(x[2]) * dt  #y
		x_new[2] = x[2] + x[4] * dt                 #theta
		x_new[3] = x[3] + x[5] * dt                 #v
		x_new[4] = x[4] + x[6] * dt                 #omega
		x_new[5] = x[5]                             # dv/dt
		x_new[6] = x[6]                             # domega/dt
		return x_new
	
	@staticmethod
	def _f_ca_jacobian(x, dt):
		v = x[3] 
		th = x[2]
		return np.array([[1, 0, -v*np.sin(th)*dt, np.cos(th)*dt,  0,  0,   0], \
						 [0, 1,  v*np.cos(th)*dt, np.sin(th)*dt,  0,  0,   0], \
						 [0, 0,                1,             0, dt,  0,   0], \
						 [0, 0,                0,             1,  0, dt,   0], \
						 [0, 0,                0,             0,  1,  0,  dt], \
						 [0, 0,                0,             0,  0,  1,   0], \
						 [0, 0,                0,             0,  0,  0,   1] \
						 ]).astype(np.float)

	@staticmethod
	def _f_ca_num_jacobian(x, dt, eps=1e-8):
		nx = len(x)
		jac = np.zeros([nx,nx])
		for i in range(nx):
			xplus = x + eps * np.array([int(ind==i) for ind in range(nx)])
			xminus = x - eps * np.array([int(ind==i) for ind in range(nx)])

			f_plus  = EKF_CA_MODEL._f_ca(xplus, dt)
			f_minus = EKF_CA_MODEL._f_ca(xminus, dt)

			jac[:,i] = (f_plus - f_minus) / (2.*eps)
		return jac

	@staticmethod
	def _identify_Q(x_sequence, dt=0.1, nx=7):
		Q_est = np.zeros((nx,nx))
		N = len(x_sequence)
		for i in range(1, N):
			x_curr = x_sequence[i-1]
			x_next = x_sequence[i]
			x_next_model = EKF_CA_MODEL._f_ca(x_curr, dt)
			w = x_next - x_next_model
			Q_est += 1./N * np.outer(w,w)
		return Q_est
'''

if __name__ == '__main__':
	filter_choice = 'cv' # 'cv', 'ca'

	# Dubin's car example
	dubins_traj = [[0., 0., 0., 10., 0.01]]
	dt = 0.1
	for i in range(50):
		dubins_traj.append( EKF_CV_MODEL._f_cv(dubins_traj[-1], dt) )
	for i in range(50):
		if i == 0:
			dubins_traj[-1][4] = -0.01
			dubins_traj[-1][3] = 9.0
		dubins_traj.append( EKF_CV_MODEL._f_cv(dubins_traj[-1], dt) )
	for i in range(50):
		if i == 0:
			dubins_traj[-1][4] = 0.
			dubins_traj[-1][3] = 5.0
		dubins_traj.append( EKF_CV_MODEL._f_cv(dubins_traj[-1], dt) )
		
	dubins_traj = np.array(dubins_traj)

	np.set_printoptions(precision=2)
	xfilt = []; xfiltstd = []
	
	if filter_choice == 'cv':
		kf = EKF_CV_MODEL(x_init = dubins_traj[0,:], dt=dt, P_init = np.eye(5) * 1.)
	elif filter_choice == 'ca':
		# init = dubins_traj[0,:].tolist() # hacky
		# init.extend([0,0])
		# kf = EKF_CA_MODEL(x_init = np.array(init), dt=dt, P_init = np.eye(7) * 1.)
		raise NotImplemented("WIP.")
	else:
		raise ValueError("Invalid filter choice: ", filter_choice)

	Q_fit = EKF_CV_MODEL._identify_Q_sequence(dubins_traj) 
	kf.Q = Q_fit
	print(kf.Q)

	for i in range(len(dubins_traj)):
		kf.predict()
		kf.update(dubins_traj[i,:3])
		x = kf.get_x()
		P = kf.get_P()
		xfilt.append(x)
		xfiltstd.append( np.sqrt(np.diag(P)) )

	xpred = []; xpredstd = []
	pred_steps = 50
	for i in range(pred_steps):
		kf.predict()
		x = kf.get_x()
		P = kf.get_P()
		xpred.append(x)
		xpredstd.append( np.sqrt(np.diag(P)) )
		
	xfilt = np.array(xfilt)
	xfiltstd = np.array(xfiltstd)
	xpred = np.array(xpred)
	xpredstd = np.array(xpredstd)

	inds_filt = np.arange(len(dubins_traj))
	inds_pred = np.arange(len(dubins_traj), len(dubins_traj) + pred_steps)

	plt.figure()
	plt.plot(xfilt[:,0], xfilt[:,1], 'r.')
	plt.plot(xpred[:,0], xpred[:,1], 'c')
	plt.plot(dubins_traj[:,0], dubins_traj[:,1], 'k')
	plt.title('Position: X vx Y')

	plt.figure()
	plt.plot(inds_filt, xfilt[:,2], 'r.')
	below = xfilt[:,2] - xfiltstd[:,2]
	above = xfilt[:,2] + xfiltstd[:,2]
	plt.fill_between(inds_filt, below, above, color='r', alpha=0.5)
	plt.plot(inds_pred, xpred[:,2], 'c.')
	below = xpred[:,2] - xpredstd[:,2]
	above = xpred[:,2] + xpredstd[:,2]
	plt.fill_between(inds_pred, below, above, color='c', alpha=0.5)
	plt.plot(inds_filt, dubins_traj[:,2], 'k')
	plt.title("Heading")

	plt.figure()
	plt.plot(inds_filt, xfilt[:,3], 'r.')
	below = xfilt[:,3] - xfiltstd[:,3]
	above = xfilt[:,3] + xfiltstd[:,3]
	plt.fill_between(inds_filt, below, above, color='r', alpha=0.5)
	plt.plot(inds_pred, xpred[:,3], 'c.')
	below = xpred[:,3] - xpredstd[:,3]
	above = xpred[:,3] + xpredstd[:,3]
	plt.fill_between(inds_pred, below, above, color='c', alpha=0.5)
	plt.plot(inds_filt, dubins_traj[:,3], 'k')
	plt.title("Velocity")

	plt.figure()
	plt.plot(inds_filt, xfilt[:,4], 'r.')
	below = xfilt[:,4] - xfiltstd[:,4]
	above = xfilt[:,4] + xfiltstd[:,4]
	plt.fill_between(inds_filt, below, above, color='r', alpha=0.5)
	plt.plot(inds_pred, xpred[:,4], 'c.')
	below = xpred[:,4] - xpredstd[:,4]
	above = xpred[:,4] + xpredstd[:,4]
	plt.fill_between(inds_pred, below, above, color='c', alpha=0.5)
	plt.plot(inds_filt, dubins_traj[:,4], 'k')
	plt.title("Angular Velocity")
	plt.show()
