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

from cvxpy.problems.problem import Problem, Minimize
from cvxpy.expressions.constants import Parameter
from cvxpy.atoms import sum_squares

import numpy as np
from time import time
from collections import defaultdict
from multiprocessing import Process, Pipe

# Flip sign of objective if maximization.
def flip_obj(prob):
	if isinstance(prob.objective, Minimize):
		return prob.objective
	else:
		return -prob.objective

# Spectral step size.
def step_ls(p, d):
	sd = np.sum(d**2)/np.sum(p*d)   # Steepest descent
	mg = np.sum(p*d)/np.sum(p**2)   # Minimum gradient
	
	if 2*mg > sd:
		return mg
	else:
		return (sd - mg)

# Step size correlation.
def step_cor(p, d):
	return np.sum(p*d)/np.sqrt(np.sum(p**2)*np.sum(d**2))

def step_safe(rho, a, b, a_cor, b_cor, eps = 0.2):
	"""Safeguarding rule for spectral step size update.
	
	Parameters
	----------
    rho : float
        The current step size.
    a : float
        Reciprocal of the curvature parameter alpha.
    b : float
        Reciprocal of the curvature parameter beta.
    a_cor : float
        Correlation of the curvature parameter alpha.
    b_cor : float
        Correlation of the curvature parameter beta.
    eps : float, optional
        The safeguarding threshold.
	"""
	if a_cor > eps and b_cor > eps:
		return np.sqrt(a*b)
	elif a_cor > eps and b_cor <= eps:
		return a
	elif a_cor <= eps and b_cor > eps:
		return b
	else:
		return rho

def step_spec(rho, k, dx, dxbar, du, duhat, eps = 0.2, C = 1e10):
	"""Calculates the generalized spectral step size with safeguarding.
	Xu, Taylor, et al. "Adaptive Consensus ADMM for Distributed Optimization."
	
	Parameters
    ----------
    rho : float
        The current step size.
    k : int
        The current iteration.
    dx : array
        Change in primal value from the last step size update.
    dxbar : array
        Change in average primal value from the last step size update.
    du : array
        Change in dual value from the last step size update.
    duhat : array
        Change in intermediate dual value from the last step size update.
    eps : float, optional
        The safeguarding threshold.
    C : float, optional
        The convergence constant.
    
    Returns
    ----------
    float
        The spectral step size for the next iteration.
	"""
	# Use old step size if unable to solve LS problem/correlations.
	if sum(dx**2) == 0 or sum(dxbar**2) == 0 or \
	   sum(du**2) == 0 or sum(duhat**2) == 0:
	   return rho
	
	# Compute spectral step size.
	a_hat = step_ls(dx, duhat)
	b_hat = step_ls(dxbar, du)
	
	# Estimate correlations.
	a_cor = step_cor(dx, duhat)
	b_cor = step_cor(dxbar, du)
	
	# Apply safeguarding rule.
	scale = 1 + C/(1.0*k**2)
	rho_hat = step_safe(rho, a_hat, b_hat, a_cor, b_cor, eps)
	return max(min(rho_hat, scale*rho), rho/scale)

def run_worker(pipe, p, rho_init, Tf, eps, C):
	f = flip_obj(p).args[0]
	cons = p.constraints
	
	# Add penalty for each variable.
	v = {}
	rho = Parameter(1, 1, value = rho_init, sign = "positive")
	for xvar in p.variables():
		xid = xvar.id
		size = xvar.size
		v[xid] = {"x": xvar, "xbar": Parameter(size[0], size[1], value = np.zeros(size)),
				  "u": Parameter(size[0], size[1], value = np.zeros(size))}
		f += (rho/2.0)*sum_squares(xvar - v[xid]["xbar"] - v[xid]["u"]/rho)
	prox = Problem(Minimize(f), cons)
	
	# Initiate step size variables.
	size_all = np.prod([np.prod(xvar.size) for xvar in p.variables()])
	v_old = {"x": np.zeros(size_all), "xbar": np.zeros(size_all),
			 "u": np.zeros(size_all), "uhat": np.zeros(size_all)}
	
	# ADMM loop.
	while True:
		prox.solve()
		xvals = {}
		for xvar in prox.variables():
			xvals[xvar.id] = xvar.value
		pipe.send(xvals)
		
		# Update u += x - x_bar.
		xbars, i = pipe.recv()
		v_flat = {"x": [], "xbar": [], "u": [], "uhat": []}
		for key in v.keys():
			xbar_old = v[key]["xbar"].value
			u_old = v[key]["u"].value
			
			v[key]["xbar"].value = xbars[key]
			v[key]["u"].value += (rho*(v[key]["x"] - v[key]["xbar"])).value
			
			# Intermediate variable for step size update.
			u_hat = u_old + rho*(xbar_old - v[key]["u"])
			v_flat["uhat"] += [np.asarray(u_hat.value).reshape(-1)]
		
		if i % Tf == 1:
			# Collect and flatten variables.
			for key in v.keys():
				v_flat["x"] += [np.asarray(v[key]["x"].value).reshape(-1)]
				v_flat["xbar"] += [np.asarray(v[key]["xbar"].value).reshape(-1)]
				v_flat["u"] += [np.asarray(v[key]["u"].value).reshape(-1)]
			
			for key in v_flat.keys():
				v_flat[key] = np.concatenate(v_flat[key])
			
			# Calculate change from old iterate.
			dx = v_flat["x"] - v_old["x"]
			dxbar = -v_flat["xbar"] + v_flat["xbar"]
			du = v_flat["u"] - v_old["u"]
			duhat = v_flat["uhat"] - v_old["uhat"]
			
			# Update step size.
			rho.value = step_spec(rho.value, i, dx, dxbar, du, duhat, eps, C)
			
			# Update step size variables.
			for key in v_flat.keys():
				v_old[key] = v_flat[key]

def consensus(p_list, max_iter = 100, rho_init = None, **kwargs):
	N = len(p_list)   # Number of problems.
	if rho_init is None:
		rho_init = N*[1.0]
	
	# Step size parameters.
	Tf = kwargs["Tf"] if "Tf" in kwargs else 2
	eps = kwargs["eps"] if "eps" in kwargs else 0.2
	C = kwargs["C"] if "C" in kwargs else 1e10
	
	# Set up the workers.
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target = run_worker, args = (remote, p_list[i], rho_init[i], Tf, eps, C))]
		procs[-1].start()

	# ADMM loop.
	start = time()
	for i in range(max_iter):
		# Gather and average x_i.
		xbars = defaultdict(float)
		xcnts = defaultdict(int)
		xvals = [pipe.recv() for pipe in pipes]
	
		for d in xvals:
			for key, value in d.items():
				xbars[key] += value
				++xcnts[key]
	
		for key in xbars.keys():
			if xcnts[key] != 0:
				xbars[key] /= xcnts[key]
	
		# Scatter x_bar.
		for pipe in pipes:
			pipe.send((xbars, i))
	end = time()

	[p.terminate() for p in procs]
	return {"xbars": xbars, "solve_time": (end - start)}
