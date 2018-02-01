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

def run_worker(p, rho, pipe):
	# Flip sign of objective if maximization.
	if isinstance(p.objective, Minimize):
		f = p.objective.args[0]
	else:
		f = -p.objective.args[0]
	cons = p.constraints
	
	# Add penalty for each variable.
	v = {}
	for xvar in p.variables():
		xid = xvar.id
		size = xvar.size
		v[xid] = {"x": xvar, "xbar": Parameter(size[0], size[1], value = np.zeros(size)),
				  "u": Parameter(size[0], size[1], value = np.zeros(size))}
		f += (rho/2.0)*sum_squares(xvar - v[xid]["xbar"] - v[xid]["u"])
	prox = Problem(Minimize(f), cons)
	
	# ADMM loop.
	while True:
		prox.solve()
		xvals = {}
		for xvar in prox.variables():
			xvals[xvar.id] = xvar.value
		pipe.send(xvals)
		
		# Update u += x - x_bar.
		xbars = pipe.recv()
		for key in v.keys():
			v[key]["xbar"].value = xbars[key]
			v[key]["u"].value += v[key]["x"].value - v[key]["xbar"].value

def consensus(p_list, rho_list = None, max_iter = 100):
	# Number of problems.
	N = len(p_list)
	if rho_list is None:
		rho_list = N*[1.0]
	
	# Set up the workers.
	pipes = []
	procs = []
	for i in range(N):
		local, remote = Pipe()
		pipes += [local]
		procs += [Process(target = run_worker, args = (p_list[i], rho_list[i], remote))]
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
			pipe.send(xbars)
	end = time()

	[p.terminate() for p in procs]
	return xbars, (end - start)

def get_combined_prob(p_list):
	obj = 0
	cons = []
	for p in p_list:
		if isinstance(p.objective, Minimize):
			f = p.objective.args[0]
		else:
			f = -p.objective.args[0]
		obj += f
		cons += p.constraints

	prob = Problem(Minimize(obj), cons)
	return prob

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
	
	# Solve with consensus ADMM.
	xbars, elapsed = consensus(p_list, rho_list = N*[1.0], max_iter = max_iter)

	# Solve combined problem.
	p_comb = get_combined_prob(p_list)
	p_comb.solve()

	# Compare results.
	pvars = [p.variables() for p in p_list]
	pvars = list(set().union(*pvars))
	for x in pvars:
		print x, "ADMM Solution:\n", xbars[x.id]
		print x, "Base Solution:\n", x.value
		print x, "MSE: ", np.mean(np.square(xbars[x.id] - x.value))
	print "Elapsed Time: %f" % elapsed

from cvxpy import *
basic_test()

