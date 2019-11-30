import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle, Ellipse
#import pickle as pkl 
#import pdb

#from filterpy.kalman import MerweScaledSigmaPoints
#from filterpy.kalman import UnscentedKalmanFilter as UKF

class EKF_CV_MODEL(object):
	# Modified from https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py
	def __init__(self, 
		         x_init = np.zeros(5),                       # initial state guess
		         P_init = np.eye(5),                         # initial state covariance guess
		         Q = np.diag([0.1, 0.1, 0.01, 1., 0.1]),    # disturbance covariance
		         R = np.diag([1e-3]*3),               	 # measurement covariance
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

	def predict(self):
		self.x = EKF_CV_MODEL._f_cv(self.x, self.dt)

		A = EKF_CV_MODEL._f_cv_jacobian(self.x, self.dt)
		#A = EKF_CV_MODEL._f_cv_num_jacobian(self.x, self.dt)
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
	def _identify_Q(x_sequence, nx=5):
		Q_est = np.zeros((nx,nx))
		N = len(x_sequence)
		for i in range(1, N):
			w = x_sequence[i] - x_sequence[i-1]
			Q_est += 1./N * np.outer(w,w)
		return Q_est


class EKF_CA_MODEL(object):
	# Modified from https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py
	def __init__(self, 
		         x_init = np.zeros(7),                       # initial state guess
		         P_init = np.eye(7),                         # initial state covariance guess
		         Q = np.diag([0.1, 0.1, 0.01, 1., 0.1, 1., 0.1]),    # disturbance covariance
		         R = np.diag([1e-3]*3),               	 # measurement covariance
		         dt = 0.1                                    # model discretization time
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
		#Anum = EKF_CA_MODEL._f_ca_num_jacobian(self.x, self.dt)
		A = EKF_CA_MODEL._f_ca_jacobian(self.x, self.dt)
		#print('Norm: ', np.linalg.norm(Anum - A))
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

'''
class UKF_CV_MODEL(object):
	# Modified from https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/UKF.py
	# Having issues with positive definiteness of (lambda + n) * P in sigma points, need high alpha.
	# Also issue with residuals where sign of heading is flipping for really small variations in sin/cos sum.
	# For these reasons, dropping this filter for now.
	def __init__(self, 
		         x_init = np.zeros(5),                       # initial state guess
		         P_init = np.eye(5),                         # initial state covariance guess
		         Q = np.diag([0.1, 0.1, 0.01, 1., 0.1]),    # disturbance covariance
		         R = np.diag([1e-3]*3),               	 # measurement covariance
		         dt = 0.1                                    # model discretization time
		         ):

		nx = 5 # state: [x, y, psi, v, omega]
		nz = 3 # obs: [x ,y, psi]

		if Q.shape != (nx,nx):
			raise ValueError('Q should be a %d by %d matrix' % (nx,nx) )
		if R.shape != (nz, nz):
			raise ValueError('R should be a %d by %d matrix' % (nz, nz))
		if x_init.shape != (nx,):
			raise ValueError('x_init should be a %d vector' % (nz, nz))
		if P_init.shape != (nx, nx):
			raise ValueError('P_init should be a %d by %d matrix' % (nx, nx))

		# Could make these model params in constructor as well.
		points = MerweScaledSigmaPoints(n=nx, 
			                            alpha=0.75,
			                            beta=2., 
			                            kappa=-2.,
			                            subtract=UKF_CV_MODEL._residual_x_z)
		self.ukf = UKF(dim_x=nx,
		               dim_z=nz, 
		               fx=UKF_CV_MODEL._f_cv, 
		               hx=UKF_CV_MODEL._h_cv,
		               dt=dt, 
		               points=points, 
		               x_mean_fn=UKF_CV_MODEL._x_mean, 
		               z_mean_fn=UKF_CV_MODEL._z_mean, 
		               residual_x=UKF_CV_MODEL._residual_x_z, 
		               residual_z=UKF_CV_MODEL._residual_x_z)

		# initial state and covariance estimate
		self.ukf.x = x_init 
		self.ukf.P = P_init

		# process noise
		self.ukf.Q = Q

		# measurement noise
		self.ukf.R = R

	def predict(self):
		self.ukf.predict()
		self.ukf.sigmas_f = self.ukf.points_fn.sigma_points(self.ukf.x, self.ukf.P)


	def update(self, measurement):
		self.ukf.update(measurement)

	def get_x(self):
		return self.ukf.x.copy()

	def get_P(self):
		return self.ukf.P.copy()

	@staticmethod
	def _f_cv(z, dt):
		z_new = np.zeros(len(z))
		z_new[0] = z[0] + z[3] * np.cos(z[2]) * dt  #x
		z_new[1] = z[1] + z[3] * np.sin(z[2]) * dt  #y
		z_new[2] = z[2] + z[4] * dt                 #theta
		z_new[3] = z[3]                             #v
		z_new[4] = z[4]                             #omega
		return z_new
	
	@staticmethod        
	def _h_cv(z):
		return [z[0], z[1], UKF_CV_MODEL._normalize_angle(z[2])]  

	@staticmethod
	def _x_mean(sigmas, Wm):
		nx = sigmas.shape[1]
		x = np.zeros(nx)

		sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
		sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
		for i in range(nx):
		    if i is 2:
		        x[i] = np.arctan2(sum_sin, sum_cos)       # angle weighted avg
		    else:
		        x[i] = np.sum(np.dot(sigmas[:, i], Wm)) # normal weighted avg
		
		if np.abs( np.mean(sigmas, axis=0)[2] - x[2] ) > 0.1:
			pass
			#pdb.set_trace()
		return x

	@staticmethod
	def _z_mean(sigmas, Wm):
		nz = sigmas.shape[1]
		z = np.zeros(nz)

		sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
		sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
		for i in range(nz):
		    if i is 2:
		        z[i] = np.arctan2(sum_sin, sum_cos)
		    else:
		        z[i] = np.sum(np.dot(sigmas[:,i], Wm))
		return z

	@staticmethod
	def _normalize_angle(ang):
		while ang >= np.pi:
		    ang -= 2 * np.pi
		while ang <= -np.pi:
		    ang += 2 * np.pi
		return ang

	@staticmethod
	def _residual_x_z(a, b):
		y = a - b
		y[2] = UKF_CV_MODEL._normalize_angle(y[2])
		return y
'''

if __name__ == '__main__':
	filter_choice = 'ukf' # 'cv', 'ca', 'ukf'

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
	
	init = dubins_traj[0,:].tolist()
	init.extend([0,0])
	if filter_choice == 'cv':
		kf = EKF_CV_MODEL(x_init = dubins_traj[0,:], dt=dt, P_init = np.eye(5) * 1.)
	elif filter_choice == 'ca':
		kf = EKF_CA_MODEL(x_init = np.array(init), dt=dt, P_init = np.eye(7) * 1.)
	elif filter_choice == 'ukf':
		kf = UKF_CV_MODEL(x_init = dubins_traj[0,:], dt=dt, P_init = np.eye(5) * 1.)
	else:
		raise ValueError("Invalid filter choice: ", filter_choice)

	print(EKF_CV_MODEL._identify_Q(dubins_traj))

	for i in range(len(dubins_traj)):
		kf.predict()
		kf.update(dubins_traj[i,:3])
		x = kf.get_x()
		P = kf.get_P()
		#print(i,x,P,'\n')

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
		
		#print(i,x,P,'\n')

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
	#for i in range(xpred.shape[0]):
	#	rect = Ellipse((xpred[i,0] - xpredstd[i,0], xpred[i,1] - xpredstd[i,1] ), 2*xpredstd[i,0], 2*xpredstd[i,1], alpha=0.5, color='c')
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
