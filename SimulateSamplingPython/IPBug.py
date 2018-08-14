import math
import numpy as np
import pulp
from scipy.linalg import solve
import sys

def solve_ilp(objective, constraints):
    """Solve the integer programming problem.
    
    Args:
        objective: the function you are going to minimize
        constraints: the possible constraints you are going to follow
            while optimizing the objective

    Returns:
        None: if there is no possible solution for the optimization problem
        otherwise, returns the variables suitable for the optimization problem
   
    """
    print("objective", objective)
    print("constraaints", constraints)
    prob = pulp.LpProblem('LP1', pulp.LpMinimize)
    prob += objective
    for cons in constraints:
        prob += cons
    print("prob", prob)    
    # The MIP solver will terminate (with an optimal result) 
    # when the gap between the lower and upper objective bound 
    # is less than MIPGap times the absolute value of the upper bound.
    status = prob.solve(pulp.GLPK(options=['--mipgap', '0', '--cgr']))
    # status = prob.solve(pulp.COIN_CMD(fracGap = 0));
    # status = prob.solve(pulp.PULP_CBC_CMD()); # default
    if status != 1:
        # print('status', status)
        # LpStatus = {-3: 'Undefined', -2: 'Unbounded', -1: 'Infeasible', 0:
        # 'No...
        return None
    else:
        return [v.varValue.real for v in prob.variables()]
        
MAX = (1<<31)-1
MIN = -(1<<31)
variablesNum = 3;
V_NUM = variablesNum + 1
variables = []

# Variables
pulp_variables = []
for i in range(1, variablesNum + 1):                                   
    pulp_variables.append(pulp.LpVariable('X%d' % i, cat = pulp.LpInteger, 
                                            lowBound = MIN, upBound = MAX))

pulp_variables.append(pulp.LpVariable('t', cat = pulp.LpInteger))
print("pulp_variables", pulp_variables)

# Objective function
F = [0] * variablesNum
F.append(1)
objective = sum([F[i] * pulp_variables[i] for i in range(0, V_NUM)]) 
print(objective)

# Build Constraints
constraints = []
constraints.append(sum([F[i] * pulp_variables[i] for i in range(0, V_NUM)]) <= 0)
for k in range(0, 1):
    temp_a = [0, 1, 1, 4]
    temp_b = temp_a[variablesNum]
    # add fi - t <= 0
    temp_a[variablesNum] = -1
    constraints.append(sum([temp_a[i] * pulp_variables[i] for i in range(0, V_NUM)]) <= -temp_b)               
    # add fi + t >= 0 [for the equation constraints
    temp_a[variablesNum] = 1
    constraints.append(sum([temp_a[i] * pulp_variables[i] for i in range(0, V_NUM)]) >= -temp_b)  
        
temp_v = [0] * variablesNum
for k in range(0, variablesNum):
    temp_v[k] = 1
    constraints.append(sum([temp_v[i] * pulp_variables[i] for i in range(0, variablesNum)]) >= MIN)
    constraints.append(sum([temp_v[i] * pulp_variables[i] for i in range(0, variablesNum)]) <= MAX)
    temp_v[k] = 0

variables = solve_ilp(objective, constraints)
print("variables", variables)

print("Program ends");