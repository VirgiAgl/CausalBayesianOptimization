import sys
sys.path.append("..") 

## Import basic packages
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from collections import OrderedDict
from matplotlib import cm
import scipy
import itertools
import sys
from numpy.random import randn
import copy
import seaborn as sns

from . import graph
from utils_functions import fit_single_GP_model


from emukit.core.acquisition import Acquisition

## Import GP python packages
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from .CompleteGraph_DoFunctions import *
from .CompleteGraph_CostFunctions import define_costs

class CompleteGraph(graph.GraphStructure):
    """
    An instance of the class graph giving the graph structure in the synthetic example 
    
    Parameters
    ----------
    """

    def __init__(self, observational_samples):

        self.A = np.asarray(observational_samples['A'])[:,np.newaxis]
        self.B = np.asarray(observational_samples['B'])[:,np.newaxis]
        self.C = np.asarray(observational_samples['C'])[:,np.newaxis]
        self.D = np.asarray(observational_samples['D'])[:,np.newaxis]
        self.E = np.asarray(observational_samples['E'])[:,np.newaxis]
        self.F = np.asarray(observational_samples['F'])[:,np.newaxis]
        self.Y = np.asarray(observational_samples['Y'])[:,np.newaxis]

    def define_SEM(self):

        def fU1(epsilon, **kwargs):
          return epsilon[0]

        def fU2(epsilon, **kwargs):
          return epsilon[1]

        def fF(epsilon, **kwargs):
          return epsilon[8]

        def fA(epsilon, U1, F, **kwargs):
          return F**2 + U1 + epsilon[2]

        def fB(epsilon, U2, **kwargs):
          return U2 + epsilon[3]

        def fC(epsilon, B, **kwargs):
          return np.exp(-B) + epsilon[4]

        def fD(epsilon, C, **kwargs):
          return np.exp(-C)/10. + epsilon[5]

        def fE(epsilon, A, C, **kwargs):
          return np.cos(A) + C/10. + epsilon[6]

        def fY(epsilon, D, E, U1, U2, **kwargs):
          return np.cos(D) - D/5. + np.sin(E) - E/4. + U1 + np.exp(-U2) + epsilon[7]

        graph = OrderedDict ([
              ('U1', fU1),
              ('U2', fU2),
              ('F', fF),
              ('A', fA),
              ('B', fB),
              ('C', fC),
              ('D', fD),
              ('E', fE),
              ('Y', fY),
            ])
        return graph


    def get_sets(self):
        MIS = [['B'], ['D'], ['E'], ['B', 'D'], ['B', 'E'], ['D', 'E']]
        POMIS = [['B'], ['D'], ['E'], ['B', 'D'], ['D', 'E']]
        manipulative_variables = ['B', 'D', 'E']
        return MIS, POMIS, manipulative_variables


    def get_set_BO(self):
        manipulative_variables = ['B', 'D', 'E']
        return manipulative_variables


    def get_interventional_ranges(self):
        min_intervention_e = -6
        max_intervention_e = 3

        min_intervention_b = -5
        max_intervention_b = 4

        min_intervention_d = -5
        max_intervention_d = 5

        min_intervention_f = -4
        max_intervention_f = 4

        dict_ranges = OrderedDict ([
          ('E', [min_intervention_e, max_intervention_e]),
          ('B', [min_intervention_b, max_intervention_b]),
          ('D', [min_intervention_d, max_intervention_d]),
          ('F', [min_intervention_f, max_intervention_f])
        ])
        return dict_ranges


    def fit_all_models(self):
        functions = {}
        inputs_list = [self.B, self.F, np.hstack((self.D,self.C)), np.hstack((self.B,self.C)), np.hstack((self.A,self.C,self.E)), np.hstack((self.B,self.C,self.D)), 
                    np.hstack((self.D,self.E,self.C,self.A)),np.hstack((self.B,self.E,self.C,self.A)), np.hstack((self.A,self.B,self.C,self.D,self.E)), 
                    np.hstack((self.A,self.B,self.C,self.D,self.E, self.F))]
        output_list = [self.C, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y]
        name_list = ['gp_C', 'gp_A', 'gp_D_C', 'gp_B_C', 'gp_A_C_E', 'gp_B_C_D', 'gp_D_E_C_A', 'gp_B_E_C_A', 'gp_A_B_C_D_E', 'gp_A_B_C_D_E_F']
        parameter_list = [[1.,1.,0.0001, False], [1.,1.,10., False], [1.,1.,1., False], [1.,1.,1., False], [1.,1.,10., False], 
                            [1.,1.,1., False], [1.,1.,10., False], [1.,1.,10., False], [1.,1.,10., False],[1.,1.,10., False]]


        ## Fit all conditional models
        for i in range(len(inputs_list)):
            X = inputs_list[i]
            Y = output_list[i]
            functions[name_list[i]] = fit_single_GP_model(X, Y, parameter_list[i])

        return functions


    def refit_models(self, observational_samples):
        A = np.asarray(observational_samples['A'])[:,np.newaxis]
        B = np.asarray(observational_samples['B'])[:,np.newaxis]
        C = np.asarray(observational_samples['C'])[:,np.newaxis]
        D = np.asarray(observational_samples['D'])[:,np.newaxis]
        E = np.asarray(observational_samples['E'])[:,np.newaxis]
        F = np.asarray(observational_samples['F'])[:,np.newaxis]
        Y = np.asarray(observational_samples['Y'])[:,np.newaxis]


        functions = {}
        inputs_list = [B, np.hstack((A,C,E)), np.hstack((D,C)), np.hstack((B,C)), np.hstack((B,C,D)), 
                    np.hstack((D,E,C,A)),np.hstack((B,E,C,A))]
        output_list = [C, Y, Y, Y, Y, Y, Y, Y]
        name_list = ['gp_C', 'gp_A_C_E', 'gp_D_C', 'gp_B_C', 'gp_B_C_D', 'gp_D_E_C_A', 'gp_B_E_C_A']
        parameter_list = [[1.,1.,10., False], [1.,1.,10., False], [1.,1.,1., False], [1.,1.,10., False], [1.,1.,10., False], [1.,1.,10., False], [1.,1.,10., False]]


        ## Fit all conditional models
        for i in range(len(inputs_list)):
            X = inputs_list[i]
            Y = output_list[i]
            functions[name_list[i]] = fit_single_GP_model(X, Y, parameter_list[i])
  
        return functions

    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs


    def get_all_do(self):
        do_dict = {}
        do_dict['compute_do_BDEF'] = compute_do_BDEF
        do_dict['compute_do_BDE'] = compute_do_BDE
        do_dict['compute_do_BD'] = compute_do_BD
        do_dict['compute_do_BE'] = compute_do_BE
        do_dict['compute_do_DE'] = compute_do_DE
        do_dict['compute_do_B'] = compute_do_B
        do_dict['compute_do_D'] = compute_do_D
        do_dict['compute_do_E'] = compute_do_E
        return do_dict



