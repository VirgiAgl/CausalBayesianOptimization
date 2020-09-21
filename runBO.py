## Import python packages
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from matplotlib import cm
from collections import OrderedDict
import scipy
import itertools
import time
from multiprocessing import Pool
import argparse 
import pathlib
import os
import sys


## Import my functions
from utils_functions import *
from BO import * 
from graphs import * 

## Args to be passed to the script
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--initial_num_obs_samples', default = 100, type = int, help = 'num initial observations')
parser.add_argument('--num_interventions', default = 10, type = int, help = 'num initial BO points')
parser.add_argument('--type_cost', default = 1, type = int, help = 'cost structure')
parser.add_argument('--num_additional_observations', default = 20, type = int, help = 'num_additional_observations')
parser.add_argument('--num_trials', default = 40, type = int, help = 'BO trials')
parser.add_argument('--name_index', default = 0, type = int, help = 'name_index')
parser.add_argument('--experiment', default = 'CompleteGraph', type = str, help = 'experiment')
parser.add_argument('--exploration_set', default = 'BO', type = str, help = 'exploration set')
args = parser.parse_args()

"""
Parameters
----------
initial_num_obs_samples : int
    Initial number of observational samples
type_cost : int 
    Type of cost per node. Fix_equal = 1, Fix_different = 2, Fix_different_variable = 3, Fix_equal_variable = 4
num_interventions : int   
    Size of the initial interventional dataset. Can be <=20
num_additional_observations: int
    Number of additional observations collected for every decision to observe
num_trials: int
    Number of BO trials
name_index : int
   	Index of interventional dataset used. 
"""

## Set the seed
seed = 9
np.random.seed(seed=int(seed))


## Get parameters passed to the script
initial_num_obs_samples = args.initial_num_obs_samples
type_cost = args.type_cost
num_interventions = args.num_interventions
num_additional_observations = args.num_additional_observations
num_trials = args.num_trials
name_index = args.name_index
experiment = args.experiment


## Set folder where to save objects
folder = set_saving_folder(args)
pathlib.Path("./Data/" + folder).mkdir(parents=True, exist_ok=True)


## Import the data
observational_samples = pd.read_pickle('./Data/' + str(args.experiment) + '/' + 'observations.pkl')[:initial_num_obs_samples]

if experiment == 'CompleteGraph':
  graph = CompleteGraph(observational_samples)

if experiment == 'ToyGraph':
    graph = ToyGraph(observational_samples)


## Fitting all the models required to compute do-effects
functions = graph.fit_all_models()


## Define the set of manipulative variables
manipulative_variables = graph.get_set_BO()


## Define interventional ranges for all interventional variables and create a dict to store them
dict_ranges =graph.get_interventional_ranges()


## Get true interventional data
interventional_data_x = np.load('./Data/' + str(args.experiment) + '/' + 'interventional_data_x_BO.npy', allow_pickle=True)
interventional_data_y = np.load('./Data/' + str(args.experiment) + '/' + 'interventional_data_y_BO.npy', allow_pickle=True)


interventional_data = manipulative_variables.copy()
interventional_data.append(np.asarray(interventional_data_x.copy()))
interventional_data.append(np.asarray(interventional_data_y.copy()))

data_x, data_y, min_intervention_value, min_y = define_initial_data_BO([interventional_data], num_interventions, manipulative_variables, name_index)


## Define cost structure
costs = graph.get_cost_structure(type_cost = type_cost)


## BO NO prior
print('Doing BO without Causal prior')
(current_cost_BO, current_best_BO, 
current_best_y_BO, total_time_BO) = NonCausal_BO(num_trials, graph, dict_ranges, data_x, 
												data_y, costs, observational_samples, functions, min_intervention_value, min_y, manipulative_variables)

## Save results
save_results_BO(folder,  args, current_cost_BO, current_best_BO, current_best_y_BO, total_time_BO, Causal_prior=False)


## BO prior
print('Doing BO with Causal prior')
(current_cost_BO_mf, current_best_BO_mf, 
current_best_y_BO_mf, total_time_BO_mf) = NonCausal_BO(num_trials, graph, dict_ranges, data_x, 
												data_y, costs, observational_samples, functions, min_intervention_value, min_y, manipulative_variables, Causal_prior=True)

## Save results
save_results_BO(folder,  args, current_cost_BO_mf, current_best_BO_mf, current_best_y_BO_mf, total_time_BO_mf, Causal_prior=True)

print('Saved BO results')
print('type_cost', args.type_cost)
print('total_time_BO', total_time_BO_mf)
print('folder', folder)



