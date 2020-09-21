## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns


def set_saving_folder(args):
  if args.type_cost == 1:
    folder = str(args.experiment) + '/Fix_equal/' + str(args.initial_num_obs_samples) + '/' + str(args.num_interventions) + '/' 

  if args.type_cost == 2:
    folder = str(args.experiment) + '/Fix_different/' + str(args.initial_num_obs_samples) + '/' + str(args.num_interventions) + '/' 

  if args.type_cost == 3:
    folder = str(args.experiment) + '/Fix_different_variable/' + str(args.initial_num_obs_samples) + '/' + str(args.num_interventions) + '/' 

  if args.type_cost == 4:
    folder = str(args.experiment) + '/Fix_equal_variable/' + str(args.initial_num_obs_samples) + '/' + str(args.num_interventions) + '/' 

  return folder


def save_results(folder,  args, current_cost, current_best_x, current_best_y, global_opt, observed, total_time):
    np.save("./Data/" + folder + "cost_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(args.name_index) + ".npy", current_cost)
    np.save("./Data/" + folder + "best_x_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(args.name_index) + ".npy",current_best_x)
    np.save("./Data/" + folder + "best_y_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(args.name_index) + ".npy", current_best_y)
    np.save("./Data/" + folder + "total_time_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(args.name_index) + ".npy",total_time)
    np.save("./Data/" + folder + "observed_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(args.name_index) + ".npy", observed)
    np.save("./Data/" + folder + "global_opt_" + str(args.exploration_set) + '_' + str(args.causal_prior) + str(args.name_index) + ".npy",global_opt)


def save_results_BO(folder,  args, current_cost, current_best_x, current_best_y, total_time, Causal_prior):
    np.save("./Data/" + folder + "cost_" + str(args.exploration_set) + '_' + str(Causal_prior) + str(args.name_index) + ".npy", current_cost)
    np.save("./Data/" + folder + "best_x_" + str(args.exploration_set) + '_' + str(Causal_prior) + str(args.name_index) + ".npy",current_best_x)
    np.save("./Data/" + folder + "best_y_" + str(args.exploration_set) + '_' + str(Causal_prior) + str(args.name_index) + ".npy", current_best_y)
    np.save("./Data/" + folder + "total_time_" + str(args.exploration_set) + '_' + str(Causal_prior) + str(args.name_index) + ".npy",total_time)
