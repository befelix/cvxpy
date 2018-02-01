"""
Copyright 2018 Anqi Fu

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy.linalg import qr_insert, qr_delete, solve_triangular

# Solve LS problem by QR decomposition
# https://en.wikipedia.org/wiki/QR_decomposition#Using_for_solution_to_linear_inverse_problems
def inner_prob(f, Q, R, Qt, Rt):
	m = Q.shape[0]
	n = R.shape[1]
	if m < n:
		Rt = Rt[0:m,:]
		Rif = solve_triangular(Rt.T, f, lower = True)
		Rif = np.insert(Rif, m, np.zeros((n-m,1)), axis = 0)
		return Qt.dot(Rif)
	else:
		Q = Q[:,0:n]
		R = R[0:n,:]
		Qf = Q.T.dot(f)
		return solve_triangular(R, Qf, lower = False)

# Trim left columns until condition number < threshold
def trim_cond(Q, R, Qt, Rt, rcond=np.inf):
	cond = np.linalg.cond(Q.dot(R))
	while (cond > rcond):
		Q, R = qr_delete(Q, R, 0, 1, which = 'col')
		Qt, Rt = qr_delete(Qt, Rt, 0, 1, which = 'row')
		cond = np.linalg.cond(Q.dot(R))
	return (Q, R, Qt, Rt)

def anderson_accel(g, x0, m, max_iters=10, rcond=np.inf):
	if(max_iters < 1):
		raise ValueError('max_iters must be >= 1')
	if(rcond < 1):
		raise ValueError('rcond must be >= 1')
	
	# Initialize
	x_o = x0
	x_n = g(x0)
	f_o = x_n-x0
	
	n = x0.shape[0]
	X_d = np.empty((n,0))
	Q = np.empty((n,n))   # F_d = Q.dot(R)
	R = np.empty((n,0))
	Qt = np.empty((0,n))  # F_d.T = Qt.dot(Rt)
	Rt = np.empty((n,n))
	
	for k in range(1, max_iters+1):
		# Compute data
		m_k = min(m,k)
		f_n = g(x_n) - x_n
		
		# Update difference matrices
		if k > m_k:
			Q, R = qr_delete(Q, R, 0, 1, which = 'col')
			Qt, Rt = qr_delete(Qt, Rt, 0, 1, which = 'row')
			X_d = np.delete(X_d, (0), axis = 1)
		if k == 1:
			Q, R = np.linalg.qr((f_n - f_o), mode = 'reduced')
			Qt, Rt = np.linalg.qr((f_n - f_o).T, mode = 'reduced')
		else:
			Q, R = qr_insert(Q, R, (f_n - f_o), R.shape[1], which = 'col')
			Qt, Rt = qr_insert(Qt, Rt, (f_n - f_o).T, Qt.shape[0], which = 'row')
		X_d = np.insert(X_d, X_d.shape[1], (x_n - x_o), axis = 1)
	
		# Trim columns if necessary
		Q, R, Qt, Rt = trim_cond(Q, R, Qt, Rt, rcond)
		m_k = R.shape[1]
		if X_d.shape[1] > m_k:
			X_d = np.delete(X_d, range(0, X_d.shape[1] - m_k), axis = 1)
	
		# Solve inner problem
		gam = inner_prob(f_n, Q, R, Qt, Rt)
		
		# Update solution
		x_o = x_n
		f_o = f_n
		x_n = x_n + f_n - (X_d + Q.dot(R)).dot(gam)
	return x_n

def basic_test():
	def g(x):
		x = x[:,0]
		y0 = (2*x[0] + x[0]**2 - x[1])/2.0
		y1 = (2*x[0] - x[0]**2 + 8)/9 + (4*x[1]-x[1]**2)/4.0
		return np.array([[y0, y1]]).T

	m = 5
	x0 = np.array([[-1.0, 1.0]]).T
	res = anderson_accel(g, x0, m, max_iters=10, rcond=2)
	print(res)   # [-1.17397479  1.37821681]

basic_test()
