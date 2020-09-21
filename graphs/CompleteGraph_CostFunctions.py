import sys
sys.path.append("..") 

## Import basic packages
import numpy as np
from collections import OrderedDict

## Define a cost variable for each intervention
def cost_A_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_B_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_C_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_D_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_E_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_F_fix_equal(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost


## Define a cost variable for each intervention
def cost_A_fix_different(intervention_value, **kwargs):
    fix_cost = 1.
    return fix_cost

def cost_B_fix_different(intervention_value, **kwargs):
    fix_cost = 10.
    return fix_cost

def cost_C_fix_different(intervention_value, **kwargs):
    fix_cost = 2.
    return fix_cost

def cost_D_fix_different(intervention_value, **kwargs):
    fix_cost = 5.
    return fix_cost

def cost_E_fix_different(intervention_value, **kwargs):
    fix_cost = 20.
    return fix_cost

def cost_F_fix_different(intervention_value, **kwargs):
    fix_cost = 3.
    return fix_cost



## Define a cost variable for each intervention
def cost_A_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_B_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 10.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_C_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 2.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_D_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 5.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_E_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 20.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_F_fix_different_variable(intervention_value, **kwargs):
    fix_cost = 3.
    return np.sum(np.abs(intervention_value)) + fix_cost



## Define a cost variable for each intervention
def cost_A_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_B_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_C_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_D_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_E_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost

def cost_F_fix_equal_variable(intervention_value, **kwargs):
    fix_cost = 1.
    return np.sum(np.abs(intervention_value)) + fix_cost



def define_costs(type_cost):
    if type_cost == 1:
        costs = OrderedDict ([
        ('A', cost_A_fix_equal),
        ('B', cost_B_fix_equal),
        ('C', cost_C_fix_equal),
        ('D', cost_D_fix_equal),
        ('E', cost_E_fix_equal),
        ('F', cost_F_fix_equal),
            ])
    if type_cost == 2:
        costs = OrderedDict ([
        ('A', cost_A_fix_different),
        ('B', cost_B_fix_different),
        ('C', cost_C_fix_different),
        ('D', cost_D_fix_different),
        ('E', cost_E_fix_different),
        ('F', cost_F_fix_different),
            ])

    if type_cost == 3:
        costs = OrderedDict ([
        ('A', cost_A_fix_different_variable),
        ('B', cost_B_fix_different_variable),
        ('C', cost_C_fix_different_variable),
        ('D', cost_D_fix_different_variable),
        ('E', cost_E_fix_different_variable),
        ('F', cost_F_fix_different_variable),
            ])

    if type_cost == 4:
        costs = OrderedDict ([
        ('A', cost_A_fix_equal_variable),
        ('B', cost_B_fix_equal_variable),
        ('C', cost_C_fix_equal_variable),
        ('D', cost_D_fix_equal_variable),
        ('E', cost_E_fix_equal_variable),
        ('F', cost_F_fix_equal_variable),
            ])

    return costs

