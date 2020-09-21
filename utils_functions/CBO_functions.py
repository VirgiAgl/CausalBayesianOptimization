
## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns



def update_hull(observational_samples, manipulative_variables):
    ## This function computes the coverage of the observations 
    list_variables = []

    for i in range(len(manipulative_variables)):
      list_variables.append(observational_samples[manipulative_variables[i]])

    stack_variables = np.transpose(np.vstack((list_variables)))
    coverage_obs = scipy.spatial.ConvexHull(stack_variables).volume

    return coverage_obs


def observe(num_observation, complete_dataset = None, initial_num_obs_samples = None):
    observational_samples = complete_dataset[initial_num_obs_samples:(initial_num_obs_samples+num_observation)]
    return observational_samples
    

def compute_coverage(observational_samples, manipulative_variables, dict_ranges):
    list_variables = []
    list_ranges = []

    for i in range(len(manipulative_variables)):
      list_variables.append(observational_samples[manipulative_variables[i]])
      list_ranges.append(dict_ranges[manipulative_variables[i]])

    vertices = list(itertools.product(*[list_ranges[i] for i in range(len(manipulative_variables))]))
    coverage_total = scipy.spatial.ConvexHull(vertices).volume

    stack_variables = np.transpose(np.vstack((list_variables)))
    coverage_obs = scipy.spatial.ConvexHull(stack_variables).volume
    hull_obs = scipy.spatial.ConvexHull(stack_variables)

    alpha_coverage = coverage_obs/coverage_total
    return alpha_coverage, hull_obs, coverage_total


def define_initial_data_CBO(interventional_data, num_interventions, exploration_set, name_index, task):

    data_list = []
    data_x_list = []
    data_y_list = []
    opt_list = []


    for j in range(len(exploration_set)):
      data = interventional_data[j].copy()
      num_variables = data[0]
      if num_variables == 1:
        data_x = np.asarray(data[(num_variables+1)])
        data_y = np.asarray(data[-1])
      else:
        data_x = np.asarray(data[(num_variables+1):(num_variables*2)][0])
        data_y = np.asarray(data[-1])


      if len(data_y.shape) == 1:
          data_y = data_y[:,np.newaxis]

      if len(data_x.shape) == 1:
          data_x = data_x[:,np.newaxis]
      


      all_data = np.concatenate((data_x, data_y), axis =1)

      ## Need to reset the global seed 
      state = np.random.get_state()

      np.random.seed(name_index)
      np.random.shuffle(all_data)


      np.random.set_state(state)

      subset_all_data = all_data[:num_interventions]

      data_list.append(subset_all_data)
      data_x_list.append(data_list[j][:,:-1])
      data_y_list.append(data_list[j][:,-1][:,np.newaxis])


      if task == 'min':
        opt_list.append(np.min(subset_all_data[:,-1])) 
        var_min = exploration_set[np.where(opt_list == np.min(opt_list))[0][0]]
        opt_y = np.min(opt_list)
        opt_intervention_array = data_list[np.where(opt_list == np.min(opt_list))[0][0]]
      else:
        opt_list.append(np.max(subset_all_data[:,-1])) 
        var_min = exploration_set[np.where(opt_list == np.max(opt_list))[0][0]]
        opt_y = np.max(opt_list)
        opt_intervention_array = data_list[np.where(opt_list == np.max(opt_list))[0][0]]


    if len(var_min) ==  3:
      best_variable1 = var_min[0]
      best_variable2 = var_min[1]
      best_variable3 = var_min[2]
      best_variable = best_variable1 + best_variable2 + best_variable3
    
    if len(var_min) ==  2:
      best_variable1 = var_min[0]
      best_variable2 = var_min[1]
      best_variable = best_variable1 + best_variable2
    
    if len(var_min) ==  1:
      best_variable = var_min[0]


    shape_opt = opt_intervention_array.shape[1] - 1
    if task == 'min':
      best_intervention_value = opt_intervention_array[opt_intervention_array[:,shape_opt] == np.min(opt_intervention_array[:,shape_opt]), :shape_opt][0]
    else:
      best_intervention_value = opt_intervention_array[opt_intervention_array[:,shape_opt] == np.max(opt_intervention_array[:,shape_opt]), :shape_opt][0]
   

    return data_x_list, data_y_list, best_intervention_value, opt_y, best_variable

