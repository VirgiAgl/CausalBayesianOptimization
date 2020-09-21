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
from sklearn.linear_model import LinearRegression
import sklearn.mixture

from . import graph
from utils_functions import fit_single_GP_model


from emukit.core.acquisition import Acquisition

## Import GP python packages
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from .CoralGraph_DoFunctions import *
from .CoralGraph_CostFunctions import define_costs

class CoralGraph(graph.GraphStructure):
    """
    An instance of the class graph giving the graph structure in the Coral reef example 
    
    Parameters
    ----------
    """

    def __init__(self, observational_samples, true_observational_samples):

        self.Y = np.asarray(observational_samples['Y'])[:,np.newaxis]
        self.N = np.asarray(observational_samples['N'])[:,np.newaxis]
        self.CO = np.asarray(observational_samples['CO'])[:,np.newaxis]
        self.T = np.asarray(observational_samples['T'])[:,np.newaxis]
        self.D = np.asarray(observational_samples['D'])[:,np.newaxis]
        self.P = np.asarray(observational_samples['P'])[:,np.newaxis]
        self.O = np.asarray(observational_samples['O'])[:,np.newaxis]
        self.S = np.asarray(observational_samples['S'])[:,np.newaxis]
        self.L = np.asarray(observational_samples['L'])[:,np.newaxis]
        self.TE = np.asarray(observational_samples['TE'])[:,np.newaxis]
        self.C = np.asarray(observational_samples['C'])[:,np.newaxis]

        true_Y = np.asarray(true_observational_samples['Y'])[:,np.newaxis]
        true_N = np.asarray(true_observational_samples['N'])[:,np.newaxis]
        true_CO = np.asarray(true_observational_samples['CO'])[:,np.newaxis]
        true_T = np.asarray(true_observational_samples['T'])[:,np.newaxis]
        true_D = np.asarray(true_observational_samples['D'])[:,np.newaxis]
        true_P = np.asarray(true_observational_samples['P'])[:,np.newaxis]
        true_O = np.asarray(true_observational_samples['O'])[:,np.newaxis]
        true_S = np.asarray(true_observational_samples['S'])[:,np.newaxis]
        true_L = np.asarray(true_observational_samples['L'])[:,np.newaxis]
        true_TE = np.asarray(true_observational_samples['TE'])[:,np.newaxis]
        true_C = np.asarray(true_observational_samples['C'])[:,np.newaxis]


        self.reg_Y = LinearRegression().fit(np.hstack((true_L, true_N, true_P, true_O, true_C, true_CO, true_TE)), true_Y)
        self.reg_P = LinearRegression().fit(np.hstack((true_S,true_T, true_D, true_TE)), true_P)
        self.reg_O = LinearRegression().fit(np.hstack((true_S, true_T, true_D, true_TE)), true_O)
        self.reg_CO = LinearRegression().fit(np.hstack((true_S, true_T, true_D, true_TE)), true_CO)
        self.reg_T = LinearRegression().fit(true_S, true_T)
        self.reg_D = LinearRegression().fit(true_S, true_D)
        self.reg_C = LinearRegression().fit(np.hstack((true_N, true_L, true_TE)), true_C)
        self.reg_S = LinearRegression().fit(true_TE, true_S)
        self.reg_TE = LinearRegression().fit(true_L, true_TE)


        ## Define distributions for the exogenous variables
        params_list = scipy.stats.gamma.fit(true_L)
        self.dist_Light = scipy.stats.gamma(a = params_list[0], loc = params_list[1], scale = params_list[2])

        mixture = sklearn.mixture.GaussianMixture(n_components=3)
        mixture.fit(true_N)
        self.dist_Nutrients_PC1 = mixture


    def define_SEM(self):

        def fN(epsilon, **kwargs):
            return self.dist_Nutrients_PC1.sample(1)[0][0][0]

        def fL(epsilon, **kwargs):
            return self.dist_Light.rvs(1)[0]

        def fTE(epsilon, L, **kwargs):
            X = np.ones((1,1))*L
            return np.float64(self.reg_TE.predict(X))


        def fC(epsilon, N, L, TE, **kwargs):
            X = np.ones((1,1))*np.hstack((N, L, TE))
            return np.float64(self.reg_C.predict(X))
            #return value

        def fS(epsilon, TE, **kwargs):
            X = np.ones((1,1))*TE
            return np.float64(self.reg_S.predict(X))
            #return value

        def fT(epsilon, S, **kwargs):
            X = np.ones((1,1))*S
            return np.float64(self.reg_T.predict(X))
            #return value

        def fD(epsilon, S, **kwargs):
            X = np.ones((1,1))*S
            return np.float64(self.reg_D.predict(X))
            #return value

        def fP(epsilon, S, T, D, TE, **kwargs):
            X = np.ones((1,1))*np.hstack((S,T, D, TE))
            return np.float64(self.reg_P.predict(X))
            #return value

        def fO(epsilon, S, T, D, TE, **kwargs):
            X = np.ones((1,1))*np.hstack((S,T, D, TE))
            return np.float64(self.reg_O.predict(X))
            #return value

        def fCO(epsilon, S, T, D, TE, **kwargs):
            X = np.ones((1,1))*np.hstack((S, T, D, TE))
            return np.float64(self.reg_CO.predict(X))
            #return value

        def fY(epsilon, L, N, P, O, C, CO, TE, **kwargs):
            X = np.ones((1,1))*np.hstack((L, N, P, O, C, CO, TE))
            return np.float64(self.reg_Y.predict(X)) 
            #return value

        graph = OrderedDict ([
          ('N', fN),
          ('L', fL),
          ('TE', fTE),
          ('C', fC),
          ('S', fS),
          ('T', fT),
          ('D', fD),
          ('P', fP),
          ('O', fO),
          ('CO', fCO),
          ('Y', fY)
        ])

        return graph


    def get_sets(self):
        MIS_1 = [['N'], ['O'], ['C'], ['T'], ['D']]
        MIS_2 = [['N', 'O'], ['N', 'C'], ['N', 'T'], ['N', 'D'], ['O', 'C'], ['O', 'T'], ['O', 'D'], ['T', 'C'], ['T', 'D'], ['C', 'D']]
        MIS_3 = [['N', 'O', 'C'], ['N', 'O', 'T'], ['N', 'O', 'D'], ['N', 'C', 'T'], ['N', 'C', 'D'], ['N','T', 'D'], 
         ['O', 'C', 'T'], ['O', 'C', 'D'], ['C', 'T', 'D'], ['O','T', 'D']]
        MIS_4 = [['N','O','C','T'], ['N','O','C','D'], ['N','O','T','D'], ['N','T','D','C'], ['T','D','C','O']]
        MIS_5 = [['N','O','C','T','D']]

        MIS = MIS_1 + MIS_2 + MIS_3

        ## To change
        POMIS = MIS

        manipulative_variables = ['N', 'O', 'C', 'T', 'D']
        return MIS, POMIS, manipulative_variables


    def get_set_BO(self):
        manipulative_variables = ['N', 'O', 'C', 'T', 'D']
        return manipulative_variables


    def get_interventional_ranges(self):
        min_intervention_N = -2 
        max_intervention_N = 5

        min_intervention_O  = 2
        max_intervention_O = 4

        min_intervention_C = 0
        max_intervention_C = 1

        min_intervention_T = 2450
        max_intervention_T = 2500

        min_intervention_D = 1950
        max_intervention_D = 1965


        # min_intervention_N = -2 
        # max_intervention_N = 5

        # min_intervention_O  = 2
        # max_intervention_O = 4

        # min_intervention_C = 0
        # max_intervention_C = 1

        # min_intervention_T = 2400
        # max_intervention_T = 2500

        # min_intervention_D = 1950
        # max_intervention_D = 2100

        dict_ranges = OrderedDict ([
          ('N', [min_intervention_N, max_intervention_N]),
          ('O', [min_intervention_O, max_intervention_O]),
          ('C', [min_intervention_C, max_intervention_C]),
          ('T', [min_intervention_T, max_intervention_T]),
          ('D', [min_intervention_D, max_intervention_D])
        ])
        return dict_ranges


    def fit_all_models(self):
        functions = {}
        inputs_list = [self.N, np.hstack((self.O,self.S, self.T,self.D,self.TE)), np.hstack((self.C,self.N, self.L,self.TE)), np.hstack((self.T,self.S)),
                        np.hstack((self.D,self.S)), np.hstack((self.N,self.O,self.S, self.T,self.D,self.TE)), np.hstack((self.N,self.T,self.S)),
                        np.hstack((self.N,self.D,self.S)), np.hstack((self.O,self.C,self.N, self.L, self.TE, self.S, self.T, self.D)),
                        np.hstack((self.T,self.C,self.S,self.TE,self.L,self.N)), np.hstack((self.T,self.D,self.S)), 
                        np.hstack((self.C,self.D,self.S, self.TE, self.L, self.N)), np.hstack((self.N,self.C,self.T, self.S, self.N, self.L, self.TE)),
                        np.hstack((self.N,self.T,self.D, self.S)), np.hstack((self.C,self.T,self.D, self.S, self.N, self.L, self.TE))]

        output_list = [self.Y, self.Y ,  self.Y , self.Y, self.Y ,self.Y , self.Y,  self.Y , self.Y,self.Y, self.Y,  self.Y, self.Y, self.Y, self.Y]

        name_list = ['gp_N', 'gp_O_S_T_D_TE', 'gp_C_N_L_TE', 'gp_T_S', 'gp_D_S', 'gp_N_O_S_T_D_TE', 'gp_N_T_S', 'gp_N_D_S', 'gp_O_C_N_L_TE_S_T_D',
                    'gp_T_C_S_TE_L_N', 'gp_T_D_S', 'gp_C_D_S_TE_L_N', 'gp_N_C_T_S_N_L_TE', 'gp_N_T_D_S', 'gp_C_T_D_S_N_L_TE']

        parameter_list = [[1.,1.,10., False], [1.,1.,1., True], [1.,1.,1., True],[1.,1.,1., True],[1.,1.,10., True], 
                        [1.,1.,1., False], [1.,1.,1., False],[1.,1.,1., False], [1.,1.,1., False],[1.,1.,1., False], [1.,1.,1., False], [1.,1.,1., False],
                        [1.,1.,1., False],[1.,1.,1., False], [1.,1.,1., False]]


        ## Fit all conditional models
        for i in range(len(inputs_list)):
            X = inputs_list[i]
            Y = output_list[i]
            functions[name_list[i]] = fit_single_GP_model(X, Y, parameter_list[i])

        return functions


    def refit_models(self, observational_samples):
        Y = np.asarray(observational_samples['Y'])[:,np.newaxis]
        N = np.asarray(observational_samples['N'])[:,np.newaxis]
        CO = np.asarray(observational_samples['CO'])[:,np.newaxis]
        T = np.asarray(observational_samples['T'])[:,np.newaxis]
        D = np.asarray(observational_samples['D'])[:,np.newaxis]
        P = np.asarray(observational_samples['P'])[:,np.newaxis]
        O = np.asarray(observational_samples['O'])[:,np.newaxis]
        S = np.asarray(observational_samples['S'])[:,np.newaxis]
        L = np.asarray(observational_samples['L'])[:,np.newaxis]
        TE = np.asarray(observational_samples['TE'])[:,np.newaxis]
        C = np.asarray(observational_samples['C'])[:,np.newaxis]


        functions = {}
        inputs_list = [N, np.hstack((O,S, T,D,TE)), np.hstack((C,N, L,TE)), np.hstack((T,S)),
                        np.hstack((D,S)), np.hstack((N,O,S, T,D,TE)), np.hstack((N,T,S)),
                        np.hstack((N,D,S)), np.hstack((O,C,N, L, TE, S, T, D)),
                        np.hstack((T,C,S,TE,L,N)), np.hstack((T,D,S)), 
                        np.hstack((C,D,S, TE, L, N)), np.hstack((N,C,T, S, N, L, TE)),
                        np.hstack((N,T,D, S)), np.hstack((C,T,D, S, N, L, TE))]

        output_list = [Y, Y ,  Y , Y, Y ,Y , Y,  Y , Y,Y, Y,  Y, Y, Y, Y]

        name_list = ['gp_N', 'gp_O_S_T_D_TE', 'gp_C_N_L_TE', 'gp_T_S', 'gp_D_S', 'gp_N_O_S_T_D_TE', 'gp_N_T_S', 'gp_N_D_S', 'gp_O_C_N_L_TE_S_T_D',
                    'gp_T_C_S_TE_L_N', 'gp_T_D_S', 'gp_C_D_S_TE_L_N', 'gp_N_C_T_S_N_L_TE', 'gp_N_T_D_S', 'gp_C_T_D_S_N_L_TE']
        
        parameter_list = [[1.,1.,10., False], [1.,1.,1., True], [1.,1.,1., True],[1.,1.,1., True],[1.,1.,10., True], 
                        [1.,1.,1., False], [1.,1.,1., False],[1.,1.,1., False], [1.,1.,1., False],[1.,1.,1., False], [1.,1.,1., False], [1.,1.,1., False],
                        [1.,1.,1., False],[1.,1.,1., False], [1.,1.,1., False]]

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
        do_dict['compute_do_N'] = compute_do_N
        do_dict['compute_do_O'] = compute_do_O
        do_dict['compute_do_C'] = compute_do_C
        do_dict['compute_do_T'] = compute_do_T
        do_dict['compute_do_D'] = compute_do_D

        do_dict['compute_do_NO'] = compute_do_NO
        do_dict['compute_do_NC'] = compute_do_NC
        do_dict['compute_do_NT'] = compute_do_NT
        do_dict['compute_do_ND'] = compute_do_ND
        do_dict['compute_do_OC'] = compute_do_OC
        do_dict['compute_do_OT'] = compute_do_OT
        do_dict['compute_do_OD'] = compute_do_OD
        do_dict['compute_do_TC'] = compute_do_TC
        do_dict['compute_do_TD'] = compute_do_TD
        do_dict['compute_do_CD'] = compute_do_CD

        do_dict['compute_do_NOC'] = compute_do_NOC
        do_dict['compute_do_NOT'] = compute_do_NOT
        do_dict['compute_do_NOD'] = compute_do_NOD
        do_dict['compute_do_NCT'] = compute_do_NCT
        do_dict['compute_do_NCD'] = compute_do_NCD
        do_dict['compute_do_NTD'] = compute_do_NTD
        do_dict['compute_do_OCT'] = compute_do_OCT
        do_dict['compute_do_OCD'] = compute_do_OCD
        do_dict['compute_do_CTD'] = compute_do_CTD
        do_dict['compute_do_OTD'] = compute_do_OTD


        return do_dict



