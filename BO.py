## Import basic python packages
import time
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from collections import OrderedDict
from matplotlib import cm
import scipy
import itertools

## Import GP python packages
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.optimization import GradientAcquisitionOptimizer

## Import defined functions
from utils_functions import *


def NonCausal_BO(num_trials, graph, dict_ranges, interventional_data_x, interventional_data_y, costs, 
			observational_samples, functions, min_intervention_value, min_y, intervention_variables, Causal_prior=False):

	## Get do function corresponding to the specified intervention_variables
	function_name = get_do_function_name(intervention_variables)
	do_function = graph.get_all_do()[function_name]

	## Compute input space dimension
	input_space = len(intervention_variables)

	## Initialise matrices for storing 
	current_best_x= np.zeros((num_trials + 1, input_space))
	current_best_y = np.zeros((num_trials + 1, 1))
	current_cost = np.zeros((num_trials + 1, 1))

	## Get functions for mean do and var do
	mean_function_do, var_function_do = mean_var_do_functions(do_function, observational_samples, functions)

	## Get interventional data
	data_x = interventional_data_x.copy()
	data_y = interventional_data_y.copy()

	
	## Assign the initial values 
	current_cost[0] = 0.
	current_best_y[0] = min_y
	current_best_x[0] = min_intervention_value
	cumulative_cost = 0.


	## Compute target function and space parameters
	target_function, space_parameters = Intervention_function(get_interventional_dict(intervention_variables),
																model = graph.define_SEM(), target_variable = 'Y', 
																min_intervention = list_interventional_ranges(graph.get_interventional_ranges(), intervention_variables)[0],
																max_intervention = list_interventional_ranges(graph.get_interventional_ranges(), intervention_variables)[1])


	if Causal_prior==False:
		#### Define the model without Causal prior
		gpy_model = GPy.models.GPRegression(data_x, data_y, GPy.kern.RBF(input_space, lengthscale=1., variance=1.), noise_var=1e-10)
		emukit_model= GPyModelWrapper(gpy_model)
	else:
		#### Define the model with Causal prior
		mf = GPy.core.Mapping(input_space, 1)
		mf.f = lambda x: mean_function_do(x)
		mf.update_gradients = lambda a, b: None
		kernel = CausalRBF(input_space, variance_adjustment=var_function_do, lengthscale=1., variance=1., rescale_variance = 1., ARD = False)
		gpy_model = GPy.models.GPRegression(data_x, data_y, kernel, noise_var=1e-10, mean_function=mf)
		emukit_model = GPyModelWrapper(gpy_model)


	## BO loop
	start_time = time.clock()
	for j in range(num_trials):
		print('Iteration', j)
		## Optimize model and get new evaluation point
		emukit_model.optimize()
		acquisition = ExpectedImprovement(emukit_model)
		optimizer = GradientAcquisitionOptimizer(space_parameters)
		x_new, _ = optimizer.optimize(acquisition)
		y_new = target_function(x_new)

		## Append the data
		data_x = np.append(data_x, x_new, axis=0)
		data_y = np.append(data_y, y_new, axis=0)
		emukit_model.set_data(data_x, data_y)

		## Compute cost
		x_new_dict = get_new_dict_x(x_new, intervention_variables)
		cumulative_cost += total_cost(intervention_variables, costs, x_new_dict)
		current_cost[j + 1] = cumulative_cost

		## Get current optimum
		results = np.concatenate((emukit_model.X, emukit_model.Y), axis =1)
		current_best_y[j + 1 ] = np.min(results[:,input_space])
		if results[results[:,input_space] == np.min(results[:,input_space]), :input_space].shape[0] > 1:
			best_x = results[results[:,input_space] == np.min(results[:,input_space]), :input_space][0]
		else:
			best_x = results[results[:,input_space] == np.min(results[:,input_space]), :input_space]
		print('Current best Y', np.min(results[:,input_space]))

	total_time = time.clock() - start_time

	return (current_cost, current_best_x, current_best_y, total_time)


