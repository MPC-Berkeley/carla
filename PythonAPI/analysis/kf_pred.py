import numpy as np
import matplotlib.pyplot as plt 
import pickle as pkl 
import pdb

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

class UKF_CV_MODEL(object):
	# Modified from https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/UKF.py
	def __init__(self, 
		         x_init = np.zeros(5),                       # initial state guess
		         P_init = np.eye(5),                         # initial state covariance guess
		         Q = np.diag([0.1, 0.1, 0.1, 1.0, 1.0]),  # disturbance covariance
		         R = np.diag([0.1, 0.1, 0.1]),            # measurement covariance
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
			                            alpha=0.000001, 
			                            beta=2., 
			                            kappa=-2,
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
		while ang >= 2 * np.pi:
		    ang -= 2 * np.pi
		while ang <= - 2 * np.pi:
		    ang += 2 * np.pi
		return ang

	@staticmethod
	def _residual_x_z(a, b):
		y = a - b
		y[2] = UKF_CV_MODEL._normalize_angle(y[2])
		return y

if __name__ == '__main__':
	# Dubin's car example
	dubins_traj = [[0., 0., 0., 10., 0.01]]
	dt = 0.1
	for i in range(50):
	    dubins_traj.append( UKF_CV_MODEL._f_cv(dubins_traj[-1], dt) )
	for i in range(50):
	    if i == 0:
	        dubins_traj[-1][4] = -0.01
	    dubins_traj.append( UKF_CV_MODEL._f_cv(dubins_traj[-1], dt) )
	for i in range(50):
	    if i == 0:
	        dubins_traj[-1][4] = 0.
	    dubins_traj.append( UKF_CV_MODEL._f_cv(dubins_traj[-1], dt) )
	    
	dubins_traj = np.array(dubins_traj)

	np.set_printoptions(precision=2)
	xfilt = []; xfiltstd = []
	ukf = UKF_CV_MODEL(x_init = dubins_traj[0,:], dt=dt).ukf

	for i in range(len(dubins_traj)):
		ukf.predict()
		#ukf.sigmas_f = ukf.points_fn.sigma_points(ukf.x, ukf.P) # see https://github.com/rlabbe/filterpy/issues/172
		ukf.update(dubins_traj[i,:3])
		xfilt.append(ukf.x.copy())
		xfiltstd.append(np.sqrt(np.diag(ukf.P)))

	xpred = []
	pred_steps = 50
	for i in range(pred_steps):
		ukf.predict()
		#ukf.sigmas_f = ukf.points_fn.sigma_points(ukf.x, ukf.P)
		xpred.append(ukf.x.copy())
		print(i, ukf.x)

	xfilt = np.array(xfilt)
	xfiltstd = np.array(xfiltstd)
	xpred = np.array(xpred)

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
	plt.plot(inds_filt, dubins_traj[:,2], 'k')
	plt.title("Heading")

	plt.figure()
	plt.plot(inds_filt, xfilt[:,3], 'r.')
	below = xfilt[:,3] - xfiltstd[:,3]
	above = xfilt[:,3] + xfiltstd[:,3]
	plt.fill_between(inds_filt, below, above, color='r', alpha=0.5)
	plt.plot(inds_pred, xpred[:,3], 'c.')
	plt.plot(inds_filt, dubins_traj[:,3], 'k')
	plt.title("Velocity")

	plt.figure()
	plt.plot(inds_filt, xfilt[:,4], 'r.')
	below = xfilt[:,4] - xfiltstd[:,4]
	above = xfilt[:,4] + xfiltstd[:,4]
	plt.fill_between(inds_filt, below, above, color='r', alpha=0.5)
	plt.plot(inds_pred, xpred[:,4], 'c.')
	plt.plot(inds_filt, dubins_traj[:,4], 'k')
	plt.title("Angular Velocity")

	plt.show()