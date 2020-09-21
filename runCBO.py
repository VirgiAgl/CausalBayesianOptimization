## Import basic packages
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from collections import OrderedDict
from matplotlib import cm
import scipy
import itertools
import time
from multiprocessing import Pool
import argparse 
import pathlib

## My functions
from utils_functions import *
from CBO import * 
from graphs import * 

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


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--initial_num_obs_samples', default = 100, type = int, help = 'num initial observations')
parser.add_argument('--num_interventions', default = 10, type = int, help = 'num initial BO points')
parser.add_argument('--type_cost', default = 1, type = int, help = 'cost structure')
parser.add_argument('--num_additional_observations', default = 20, type = int, help = 'num_additional_observations')
parser.add_argument('--num_trials', default = 40, type = int, help = 'BO trials')
parser.add_argument('--name_index', default = 0, type = int, help = 'name_index')
parser.add_argument('--exploration_set', default = 'MIS', type = str, help = 'exploration set')
parser.add_argument('--causal_prior', default = False,  type = bool, help = 'Do not specify when want to set to False')
parser.add_argument('--experiment', default = 'CompleteGraph', type = str, help = 'experiment')
parser.add_argument('--task', default = 'min', type = str, help = 'experiment')
args = parser.parse_args()


## Set the seed
seed = 9
np.random.seed(seed=int(seed))

## Set the parameters
initial_num_obs_samples = args.initial_num_obs_samples
type_cost = args.type_cost
num_interventions = args.num_interventions
num_additional_observations = args.num_additional_observations
num_trials = args.num_trials
name_index = args.name_index
exploration_set = args.exploration_set
causal_prior = args.causal_prior
experiment = args.experiment
task = args.task

print('exploration_set', exploration_set)
print('initial_num_obs_samples', initial_num_obs_samples)
print('num_interventions', num_interventions)
print('type_cost', type_cost)
print('num_trials', num_trials)
print('causal_prior', causal_prior)
print('experiment', experiment)
print('task', task)


## Import the data
observational_samples = pd.read_pickle('./Data/' + str(args.experiment) + '/' + 'observations.pkl')[:initial_num_obs_samples]
full_observational_samples = pd.read_pickle('./Data/' + str(args.experiment) + '/' + 'observations.pkl')

if experiment == 'ToyGraph':
    graph = ToyGraph(observational_samples)

if experiment == 'CompleteGraph':
    graph = CompleteGraph(observational_samples)

if experiment == 'CoralGraph':
    true_observational_samples = pd.read_pickle('./Data/' + str(args.experiment) + '/' + 'true_observations.pkl')
    graph = CoralGraph(observational_samples, true_observational_samples)

if experiment == 'SimplifiedCoralGraph':
    true_observational_samples = pd.read_pickle('./Data/' + str(args.experiment) + '/' + 'true_observations.pkl')
    graph = SimplifiedCoralGraph(observational_samples, true_observational_samples)



## Set folder where to save objects
folder = set_saving_folder(args)
pathlib.Path("./Data/" + folder).mkdir(parents=True, exist_ok=True)


## Givent the data fit all models used for do calculus
functions = graph.fit_all_models()


## Define optimisation sets and the set of manipulative variables
max_N = initial_num_obs_samples + 50
MIS, POMIS, manipulative_variables = graph.get_sets()

## Define interventional ranges for all interventional variables and create a dict to store them
dict_ranges = graph.get_interventional_ranges()

## Compute observation coverage
alpha_coverage, hull_obs, coverage_total = compute_coverage(observational_samples, manipulative_variables, dict_ranges)

## Get true interventional data
interventional_data = np.load('./Data/' + str(args.experiment) + '/' + 'interventional_data.npy', allow_pickle=True)


## Get the initial optimal solution and the interventional data corresponding to a random permutation of the intervential data with seed given by name_index
data_x_list, data_y_list, best_intervention_value, opt_y, best_variable = define_initial_data_CBO(interventional_data, num_interventions, eval(exploration_set), name_index, task)


## Define cost structure
costs = graph.get_cost_structure(type_cost = type_cost)


print('Exploring ' + str(exploration_set) + ' with CEO and Causal prior = ' + str(causal_prior))

(current_cost, current_best_x, current_best_y, 
global_opt, observed, total_time) = CBO(num_trials, eval(exploration_set), manipulative_variables, data_x_list, data_y_list,  
													best_intervention_value, opt_y, 
													best_variable, dict_ranges,functions, observational_samples, coverage_total, graph, 
													num_additional_observations, costs, full_observational_samples, task, max_N, initial_num_obs_samples, 
													num_interventions, Causal_prior = causal_prior)


save_results(folder,  args, current_cost, current_best_x, current_best_y, global_opt, observed, total_time)


print('Saved results')
print('exploration_set', args.exploration_set)
print('causal_prior', args.causal_prior)
print('type_cost', args.type_cost)
print('total_time', total_time)
print('folder', folder)



