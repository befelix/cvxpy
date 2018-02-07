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

# Safeguarding rule.
def step_safe(a, b, a_cor, b_cor, tau, eps = 0.2):
	if a_cor > eps and b_cor > eps:
		return np.sqrt(a*b)
	elif a_cor > eps and b_cor <= eps:
		return a
	elif a_cor <= eps and b_cor > eps:
		return b
	else:
		return tau

def step_spec(rho, dx, dxbar, du, duhat, k, eps = 0.2, C = 1e10):
	# Use old step size if unable to solve LS problem/correlations.
	if sum(dx**2) == 0 or sum(dxbar**2) == 0 or \
	   sum(du**2) == 0 or sum(duhat**2) == 0:
	   return rho
	
	# Compute spectral step sizes.
	a_hat = step_ls(dx, duhat)
	b_hat = step_ls(dxbar, du)
	
	# Estimate correlations.
	a_cor = step_cor(dx, duhat)
	b_cor = step_cor(dxbar, du)
	
	# Update step size.
	scale = 1 + C/(1.0*k**2)
	rho_hat = step_safe(a_hat, b_hat, a_cor, b_cor, rho, eps)
	return max(min(rho_hat, scale*rho), rho/scale)

def proc_results(p_list, xbars):
	# TODO: Handle statuses.
	
	# Save primal values.
	pvars = [p.variables() for p in p_list]
	pvars = list(set().union(*pvars))
	for x in pvars:
		x.save_value(xbars[x.id])
	
	# TODO: Save dual values (for constraints too?).
	
	# Compute full objective.
	val = sum([flip_obj(p).value for p in p_list])
	return val

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
			rho.value = step_spec(rho.value, dx, dxbar, du, duhat, i, eps, C)
			
			# Update step size variables.
			for key in v_flat.keys():
				v_old[key] = v_flat[key]

def consensus(p_list, max_iter = 100, rho_init = None, **kwargs):
	# Number of problems.
	N = len(p_list)
	if rho_init is None:
		rho_init = N*[1.0]
	
	# Step size parameters
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
	obj_val = proc_results(p_list, xbars)
	return obj_val, (end - start)

def solve_combined(p_list):	
	obj = sum([flip_obj(p).args[0] for p in p_list])
	cons = [c for c in p.constraints for p in p_list]
	prob = Problem(Minimize(obj), cons)
	return prob.solve()

def basic_test():
	np.random.seed(1)
	m = 100
	n = 10
	max_iter = 10
	x = Variable(n)
	y = Variable(n/2)

	# Problem data.
	alpha = 0.5
	A = np.random.randn(m*n).reshape(m,n)
	xtrue = np.random.randn(n)
	b = A.dot(xtrue) + np.random.randn(m)

	# List of all the problems with objective f_i.
	p_list = [Problem(Minimize(sum_squares(A*x-b)), [norm(x,2) <= 1]),
			  Problem(Minimize((1-alpha)*sum_squares(y)/2))
			 ]
	N = len(p_list)   # Number of problems.
	pvars = [p.variables() for p in p_list]
	pvars = list(set().union(*pvars))   # Variables of problems.
	
	# Solve with consensus ADMM.
	obj_admm, elapsed = consensus(p_list, rho_list = N*[1.0], max_iter = max_iter)
	x_admm = [x.value for x in pvars]

	# Solve combined problem.
	obj_comb = solve_combined(p_list)
	x_comb = [x.value for x in pvars]

	# Compare results.
	for i in range(N):
		print "ADMM Solution:\n", x_admm[i]
		print "Base Solution:\n", x_comb[i]
		print "MSE: ", np.mean(np.square(x_admm[i] - x_comb[i])), "\n"
	print "ADMM Objective: %f" % obj_admm
	print "Base Objective: %f" % obj_comb
	print "Elapsed Time: %f" % elapsed

from cvxpy import *
basic_test()

