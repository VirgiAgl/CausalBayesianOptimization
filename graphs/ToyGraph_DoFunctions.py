import sys
sys.path.append("..") 

## Import basic packages
import numpy as np
from collections import OrderedDict

def Causal_effect_DO(*interventions, functions, 
                     parents_Y, 
                     children, 
                     parents, 
                     independent_nodes):
    ## This function can be used to compute the CE of variables that are confounded via Back door adjustment
    ## so that no adjustment is needed

    final_variables = OrderedDict()
    
    num_models = len(functions)
    num_interventions = len(children)
    num_observations = list(parents_Y.items())[0][1].shape[0]

    ## We should aggregate the tuple here
    for variable, value in interventions[0].items():
    
        num_children = len(children[variable])
        num_parents = len(parents[variable])
        num_independent_nodes = len(independent_nodes[variable])
        
        subset_children = children[variable]
        subset_parents = parents[variable]
        subset_independent_nodes = independent_nodes[variable]
        
        
        ## This is changing the intervention variable
        final_variables[variable] = value*np.ones((num_observations,1))
        
        ## This is changing the children
        if num_children != 0:
            for i in range(num_children):
                ## This should update the values of the children - eg for X this is modifying Z
                functions_to_get = list(subset_children.items())[0][0][-1]
                ## If this function gets other variables that are not children of the intervention we need to add them here
                ## This is changing the X_2 - taking the mean value of GP2
                children_value = functions[functions_to_get].predict(value*np.ones((num_observations,1)))[0]
                final_variables[list(subset_children.keys())[0]] = children_value

            
        ## The independent nodes stay the same - If dont exist dont need to provide
        if num_independent_nodes != 0:
            for j in range(num_independent_nodes):
                final_variables[list(subset_independent_nodes.keys())[j]] = list(subset_independent_nodes.items())[j][1]
            
        ## The parents nodes stay the same - If dont exist dont need to provide
        if num_parents != 0:
            for j in range(num_parents):
                final_variables[list(subset_parents.keys())[j]] = list(subset_parents.items())[j][1]
    
    
    ## after having changed all the variables, we predict the Y
    num_parents_Y = len(parents_Y.keys())
    Inputs_Y = np.zeros((num_observations, num_parents_Y))
    
    ## Getting the parents of Y to compute the CE on Y 
    for i in range(len(parents_Y.keys())):
        var = list(parents_Y.keys())[i]
        Inputs_Y[:,i] = final_variables[var][:,0]

    gp_Y = functions['Y']
    
    #samples = (gp_Y.posterior_samples_f(Inputs_Y, size=1000))**2
    causal_effect_mean = np.mean(gp_Y.predict(Inputs_Y)[0])
    causal_effect_var = np.mean(gp_Y.predict(Inputs_Y)[1])

    return causal_effect_mean, causal_effect_var


def compute_do_X(observational_samples, functions, value):
    # gp_X = functions['gp_X']
    
    # mean_do = np.mean(gp_X.predict(np.ones((1,1))*value)[0])
    
    # var_do = np.mean(gp_X.predict(np.ones((1,1))*value)[1])

    # Compute Do effects as in the notebook
    Z = observational_samples['Z']

    parents_Y = OrderedDict ([('Z', Z)])

    functions = OrderedDict ([
      ('Y', functions['Y']),
      ('Z', functions['Z']),
      ('X', [])
        ])


    children = OrderedDict([('X', OrderedDict ([('Z', Z)]) )])
    independent_nodes = OrderedDict([('X', OrderedDict ([]) )])
    parents_nodes = OrderedDict([('X', OrderedDict ([]) )])

    num_interventions = value.shape[0]
    mean_do = np.zeros((num_interventions, 1))
    var_do = np.zeros((num_interventions, 1))
    
    for i in range(num_interventions):
        
        mean_do[i], var_do[i] = Causal_effect_DO({'X':value[i]}, functions = functions, parents_Y = parents_Y, 
                                               children = children, parents=parents_nodes, 
                                               independent_nodes = independent_nodes)
    return mean_do, var_do


def compute_do_Z(observational_samples, functions, value):

    # gp_X_Z = functions['gp_X_Z']
    
    # X = np.asarray(observational_samples['X'])[:,np.newaxis]
    
    # intervened_inputs = np.hstack((X,np.repeat(value, X.shape[0])[:,np.newaxis]))
    # mean_do = np.mean(gp_X_Z.predict(intervened_inputs)[0])
    
    # var_do = np.mean(gp_X_Z.predict(intervened_inputs)[1])

    # Compute Do effects as in the notebook
    Z = observational_samples['Z']

    parents_Y = OrderedDict ([('Z', Z)])

    functions = OrderedDict ([
      ('Y', functions['Y']),
      ('Z', functions['Z']),
      ('X', [])
        ])

    children = OrderedDict([('X', OrderedDict ([('Z', Z)]) ), ('Z', OrderedDict ([]) )])
    independent_nodes = OrderedDict([('X', OrderedDict ([]) ), ('Z', OrderedDict ([]) )])
    parents_nodes = OrderedDict([('X', OrderedDict ([]) ), ('Z', OrderedDict ([('X', Z)]))])

    num_interventions = value.shape[0]
    mean_do = np.zeros((num_interventions, 1))
    var_do = np.zeros((num_interventions, 1))
    
    for i in range(num_interventions):
        mean_do[i], var_do[i] = Causal_effect_DO({'Z':value[i]}, functions = functions, parents_Y = parents_Y, 
                                               children = children, parents=parents_nodes, 
                                               independent_nodes = independent_nodes)
    

    return mean_do, var_do
   

def compute_do_XZ(observational_samples, functions, value):
    
    # gp_X_Z = functions['gp_X_Z']
    
    # X = np.asarray(observational_samples['X'])[:,np.newaxis]
    
    # intervened_inputs = np.hstack((np.repeat(value[0], X.shape[0])[:,np.newaxis], np.repeat(value[1], X.shape[0])[:,np.newaxis]))
    # mean_do = np.mean(gp_X_Z.predict(intervened_inputs)[0])
    
    # var_do = np.mean(gp_X_Z.predict(intervened_inputs)[1])

    # Compute Do effects as in the notebook
    Z = observational_samples['Z']

    parents_Y = OrderedDict ([('Z', Z)])

    functions = OrderedDict ([
      ('Y', functions['Y']),
      ('Z', functions['Z']),
      ('X', [])
        ])

    children = OrderedDict([('X', OrderedDict ([('Z', Z)]) ), ('Z', OrderedDict ([]) )])
    independent_nodes = OrderedDict([('X', OrderedDict ([]) ), ('Z', OrderedDict ([]) )])
    parents_nodes = OrderedDict([('X', OrderedDict ([]) ), ('Z', OrderedDict ([('X', Z)]))])

    mean_do, var_do = Causal_effect_DO({'X':value[0], 'Z':value[1]}, functions = functions, parents_Y = parents_Y, 
                                               children = children, parents=parents_nodes, 
                                               independent_nodes = independent_nodes)

    return mean_do, var_do

