import time
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from collections import OrderedDict
from matplotlib import cm
import scipy
import itertools
import time 

from utils_functions import *



def CBO(num_trials, exploration_set, manipulative_variables, data_x_list, data_y_list,  best_intervention_value, opt_y, 
					best_variable, dict_ranges, functions, observational_samples, coverage_total, graph, 
					num_additional_observations, costs, full_observational_samples, task = 'min', max_N = 200, 
					initial_num_obs_samples =100, num_interventions=10, Causal_prior=False):
	
	## Initialise dicts to store values over trials and assign initial values
	current_cost = []
	global_opt = []
	current_best_x, current_best_y, x_dict_mean, x_dict_var, dict_interventions = initialise_dicts(exploration_set, task)
	current_best_y[best_variable].append(opt_y)
	current_best_x[best_variable].append(best_intervention_value)
	global_opt.append(opt_y)
	current_cost.append(0.)

	## Initialise variables
	observed = 0
	trial_intervened = 0.
	cumulative_cost = 0.
	cumulative_cost_mf = 0.
			
	## Define list to store info
	target_function_list = [None]*len(exploration_set)
	space_list = [None]*len(exploration_set)
	model_list = [None]*len(exploration_set)
	type_trial = []

	## Define intervention function
	for s in range(len(exploration_set)):
		target_function_list[s], space_list[s] = Intervention_function(get_interventional_dict(exploration_set[s]),
											model = graph.define_SEM(), target_variable = 'Y',
											min_intervention = list_interventional_ranges(graph.get_interventional_ranges(), exploration_set[s])[0],
											max_intervention = list_interventional_ranges(graph.get_interventional_ranges(), exploration_set[s])[1])

	
	############################# LOOP
	start_time = time.clock()
	for i in range(num_trials):
		print('Optimization step', i)
		## Decide to observe or intervene and then recompute the obs coverage
		coverage_obs = update_hull(observational_samples, manipulative_variables)
		rescale = observational_samples.shape[0]/max_N
		epsilon_coverage = (coverage_obs/coverage_total)/rescale

		uniform = np.random.uniform(0.,1.)

		## At least observe and interve once
		if i == 0:
			uniform = 0.
		if i == 1:
			uniform = 1.


		if uniform < epsilon_coverage:
			observed += 1
			type_trial.append(0)
			## Collect observations and append them to the current observational dataset
			new_observational_samples = observe(num_observation = num_additional_observations, 
												complete_dataset = full_observational_samples, 
												initial_num_obs_samples= initial_num_obs_samples)

			observational_samples = observational_samples.append(new_observational_samples)
			
			## Refit the models for the conditional distributions 
			functions = graph.refit_models(observational_samples)
			
			## Update the mean functions and var functions given the current set of observational data. This is updating the prior. 
			mean_functions_list, var_functions_list = update_all_do_functions(graph, exploration_set, functions, dict_interventions, 
														observational_samples, x_dict_mean, x_dict_var)
			
			## Update current optimal solution. If I observe the cost and the optimal y are the same of the previous trial
			global_opt.append(global_opt[i])
			current_cost.append(current_cost[i])



		else:
			type_trial.append(1)
			trial_intervened += 1
			## When we decid to interve we need to compute the acquisition functions based on the GP models and decide the variable/variables to intervene
			## together with their interventional data

			## Define list to store info
			y_acquisition_list = [None]*len(exploration_set)
			x_new_list = [None]*len(exploration_set)
			
			## This is the global opt from previous iteration
			current_global = find_current_global(current_best_y, dict_interventions, task)


			## If in the previous trial we have observed we want to update all the BO models as the mean functions and var functions computed 
			## via the DO calculus are changed 
			## If in the previous trial we have intervened we want to update only the BO model for the intervention for which we have collected additional data 
			if type_trial[i-1] == 0:
				for s in range(len(exploration_set)):
					#print('Updating model:', s)
					model_list[s] = update_BO_models(mean_functions_list[s], var_functions_list[s], data_x_list[s], data_y_list[s], Causal_prior)
			else:
				#print('Updating model:', index)
				model_to_update = index
				model_list[index] = update_BO_models(mean_functions_list[index], 
																var_functions_list[index], 
																data_x_list[index], data_y_list[index], Causal_prior)
				

			## Compute acquisition function given the updated BO models for the interventional data
			## Notice that we use current_global and the costs to compute the acquisition functions
			for s in range(len(exploration_set)):
				y_acquisition_list[s], x_new_list[s] = find_next_y_point(space_list[s], model_list[s], current_global, exploration_set[s], costs, task = task)
				

			## Selecting the variable to intervene based on the values of the acquisition functions
			var_to_intervene = exploration_set[np.where(y_acquisition_list == np.max(y_acquisition_list))[0][0]]
			index = np.where(y_acquisition_list == np.max(y_acquisition_list))[0][0]


			## Evaluate the target function at the new point
			y_new = target_function_list[index](x_new_list[index])

			print('Selected intervention: ', var_to_intervene)
			print('Selected point: ', x_new_list[index])
			print('Target function at selected point: ', y_new)


			## Append the new data and set the new dataset of the BO model
			data_x, data_y_x = add_data([data_x_list[index], data_y_list[index]], 
												  [x_new_list[index], y_new])
				
			data_x_list[index] = np.vstack((data_x_list[index], x_new_list[index])) 
			data_y_list[index] = np.vstack((data_y_list[index], y_new))
		
			model_list[index].set_data(data_x, data_y_x)
				

			## Compute cost
			x_new_dict = get_new_dict_x(x_new_list[index], dict_interventions[index])
			cumulative_cost += total_cost(var_to_intervene, costs, x_new_dict)
			var_to_intervene = dict_interventions[index]
			current_cost.append(cumulative_cost)


	 		## Update the dict storing the current optimal solution
			current_best_x[var_to_intervene].append(x_new_list[index][0][0])
			current_best_y[var_to_intervene].append(y_new[0][0])
				

			## Find the new current global optima
			current_global = find_current_global(current_best_y, dict_interventions, task)
			global_opt.append(current_global)
		
			print('####### Current_global #########', current_global)

			## Optimise BO model given the new data
			model_list[index].optimize()

	## Compute total time for the loop
	total_time = time.clock() - start_time

	return (current_cost, current_best_x, current_best_y, global_opt, observed, total_time)

