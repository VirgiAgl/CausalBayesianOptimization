import sys
sys.path.append("..") 

## Import basic packages
import numpy as np
from collections import OrderedDict

## Define a cost variable for each intervention
def cost_N_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_O_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_C_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_T_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_D_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost


## Define a cost variable for each intervention
def cost_N_fix_different(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_O_fix_different(intervention_value, **kwargs):
    fix_cost = 10.
    return fix_cost

def cost_C_fix_different(intervention_value, **kwargs):
    fix_cost = 2.
    return fix_cost

def cost_T_fix_different(intervention_value, **kwargs):
    fix_cost = 5.
    return fix_cost

def cost_D_fix_different(intervention_value, **kwargs):
    fix_cost = 20.
    return fix_cost




## Define a cost variable for each intervention
def cost_N_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_O_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 10.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_C_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 2.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_T_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 5.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_D_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 20.
    return np.sum(np.abs(intervention_value)) + fix_cost



## Define a cost variable for each intervention
def cost_N_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_O_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_C_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_T_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_D_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost



def define_costs(type_cost):
    if type_cost == 1:
        costs = OrderedDict ([
        ('N', cost_N_fix_equal),
        ('O', cost_O_fix_equal),
        ('C', cost_C_fix_equal),
        ('T', cost_T_fix_equal),
        ('D', cost_D_fix_equal),
            ])
    if type_cost == 2:
        costs = OrderedDict ([
        ('N', cost_N_fix_different),
        ('O', cost_O_fix_different),
        ('C', cost_C_fix_different),
        ('T', cost_T_fix_different),
        ('D', cost_D_fix_different),
            ])

    if type_cost == 3:
        costs = OrderedDict ([
        ('N', cost_N_fix_different_variable),
        ('O', cost_O_fix_different_variable),
        ('C', cost_C_fix_different_variable),
        ('T', cost_T_fix_different_variable),
        ('D', cost_D_fix_different_variable),
            ])

    if type_cost == 4:
        costs = OrderedDict ([
        ('N', cost_N_fix_equal_variable),
        ('O', cost_O_fix_equal_variable),
        ('C', cost_C_fix_equal_variable),
        ('T', cost_T_fix_equal_variable),
        ('D', cost_D_fix_equal_variable),
            ])

    return costs

