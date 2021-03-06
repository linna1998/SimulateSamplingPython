import math
import numpy as np
import pulp
from scipy.linalg import solve
import sys
import matlab.engine
import cvxpy

class Value:
    """The possible returning type of a constraint

    Attributes:
        intValue: An integer value of the statement.
        isValid: A boolean indicating if the value meets the determine
            statement or not.
    """
    intValue = 0
    isValid = False

def getValue(constraintsId, variables):
    """Simulate a statement in the real program
    
    Args:
        constraintsId: the # of 'if' statement going to simulate
        variables: the chosen variables value

    Returns:
        returnValue: is in Value type
            it contains a value of the statement, 
            and the judgement that the determine statement is True / False
    """
    returnValue = Value()
    for i in range(0, variablesNum):
        returnValue.intValue += parameter_a[constraintsId][i] * variables[i]
    returnValue.intValue += parameter_b[constraintsId]
    if ((not parameter_isEqual[constraintsId]) and returnValue.intValue <= 0):
        returnValue.isValid = True
    elif (parameter_isEqual[constraintsId] and returnValue.intValue == 0):
        returnValue.isValid = True
    else:
        returnValue.isValid = False
    return returnValue

def getValueNonlinear(constraintsId, variables):
    # constraints is written in the program
    returnValue = Value()
    if (constraintsId == 0):
        returnValue.intValue = variables[1] + variables[2] + 4;
    elif (constraintsId == 1):        
        returnValue.intValue = variables[0] - variables[1] + 2;
    elif (constraintsId == 2):        
        returnValue.intValue = variables[0] * variables[1] + variables[2];
    if ((not parameter_isEqual[constraintsId]) and returnValue.intValue <= 0):
        returnValue.isValid = True
    elif (parameter_isEqual[constraintsId] and returnValue.intValue == 0):
        returnValue.isValid = True
    else:
        returnValue.isValid = False
    return returnValue

def simulateFunction(variables):
    """Simulate the whole nested statements in the real program.
    
    Args:
        variables: the chosen variables value

    Returns:
        returnValueList: a list of the value of the statements 
            execution stopped when the result of final 'if' statement is False
    """
    returnValueList = []
    for i in range(0, constraintsNum):
        returnValue = getValue(i, variables)
        # returnValue = getValueNonlinear(i, variables);
        returnValueList.append(returnValue)
        if (returnValue.isValid == False):
           break
    return returnValueList

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
    # status = prob.solve(pulp.GLPK(options=['--mipgap', '0', '--cgr']))
    status = prob.solve(pulp.GLPK(options=['--exact']))
    # status = prob.solve(pulp.COIN_CMD(fracGap = 0));
    # status = prob.solve(pulp.PULP_CBC_CMD()); # default
    if status != 1:
        # print('status', status)
        # LpStatus = {-3: 'Undefined', -2: 'Unbounded', -1: 'Infeasible', 0:
        # 'No...
        return None
    else:
        return [v.varValue.real for v in prob.variables()]

# return a variable which suits
# beginning 0 ~ constraintsId - 1 constraints
def getVariables(constraintsId, parameter_isEqual, 
                 variablesNum, variables_lower, variables_upper, parameter_result,
                 block, position_list):    

    variables = []
    if constraintsId == 0:
         for j in range(0, variablesNum):
            variables.append(np.random.randint(variables_lower[j], high = variables_upper[j]))
         return variables
    else:
        # Randomly get a variable
        # which suits for 0 ~ constraintsId - 1 constraints
        V_NUM = variablesNum + 1
        
        # Variables
        pulp_variables = []
        for i in range(1, variablesNum + 1):                                   
            # using block & position_list to divide the space into several
            # parts
            partialLength = (variables_upper[i - 1] - variables_lower[i - 1]) / block
            temp_lowBound = max(round(variables_lower[i - 1] + partialLength * position_list[i - 1]), variables_lower[i - 1]);
            temp_upBound = min(round(temp_lowBound + partialLength), variables_upper[i - 1]);            

            pulp_variables.append(pulp.LpVariable('X%d' % i, cat = pulp.LpInteger, 
                                                  lowBound = temp_lowBound, upBound = temp_upBound))

        pulp_variables.append(pulp.LpVariable('t', cat = pulp.LpInteger))
        # print("pulp_variables", pulp_variables)

        # Objective function
        F = [0] * variablesNum
        F.append(1)
        objective = sum([F[i] * pulp_variables[i] for i in range(0, V_NUM)]) 
        # objective += sum([F[i] * pulp_variables[i] for i in range(0, V_NUM)]);
        # print(objective)

        # Build Constraints
        constraints = []
        constraints.append(sum([F[i] * pulp_variables[i] for i in range(0, V_NUM)]) <= 0)
        for k in range(0, constraintsId):
            temp_a = list(parameter_result[k])
            temp_b = temp_a[variablesNum]
            # add fi - t <= 0
            temp_a[variablesNum] = -1
            constraints.append(sum([temp_a[i] * pulp_variables[i] for i in range(0, V_NUM)]) <= -temp_b)            
            if (parameter_isEqual[k]):   
                # add fi + t >= 0 [for the equation constraints
                temp_a[variablesNum] = 1
                constraints.append(sum([temp_a[i] * pulp_variables[i] for i in range(0, V_NUM)]) >= -temp_b)  
        temp_v = [0] * variablesNum
        for k in range(0, variablesNum):
            temp_v[k] = 1
            constraints.append(sum([temp_v[i] * pulp_variables[i] for i in range(0, variablesNum)]) >= variables_lower[k])
            constraints.append(sum([temp_v[i] * pulp_variables[i] for i in range(0, variablesNum)]) <= variables_upper[k])
            temp_v[k] = 0
        # print("constraints", constraints)

        variables = solve_ilp(objective, constraints)
        # print("variables", variables)
        
        print("variables", variables)
        if (variables == None):
            print("No possible solution! QAQ")
            return variables
        # delete the last "t" variable
        variables.pop()
        return variables         

def matlabSolver(A, B):
    eng = matlab.engine.start_matlab()
    B = list(map(list, zip(*[B])))  # transpose B to size (n, 1)
    C = eng.getLinear(matlab.double(A),matlab.double(B))
    # no possible solution for the linear equation A x = B
    if (C == matlab.double([])):
        print("No possible solution for the linear equation! QAQ");
        return None;
    C = list(map(list, zip(*C)))[0]
    print("C = ", C)
    for i in range(0, len(C)):
        C[i] = round(C[i])
    print("Round C = ", C)
    return C

# Random to get the variables
def solveConstraint(constraintsId, parameter_isEqual, constraintsNum, 
                 variablesNum, variables_lower, variables_upper, parameter_result, 
                 sampling_count):    
    # in linear equations
    A = []
    B = []
    equalNum = 0
    for i in range(0, constraintsId):
        if (parameter_isEqual[i]):
            equalNum += 1;
    
    block = 1
    position_list = [0] * variablesNum
    isLinear = 0;

    while (len(A) < variablesNum + 1):   
        # select a possible variables
        variables = getVariables(constraintsId, parameter_isEqual, 
                variablesNum, variables_lower, variables_upper, parameter_result,
                block, position_list);
        sampling_count += 1;
        
        # decide whether it is non-linear
        # when position_list = [0, 0, ..., 0]
        if (position_list == [0] * variablesNum and block > 1):
            if (isLinear == 0):
                print("Non-linear constraints! No possible solutions! QAQ")
                return parameter_result, sampling_count, None
            isLinear = 0;
        
        # calculate the carry, update position_list and block
        position_list[variablesNum - 1] += 1;
        final_input = 0;
        for i in range(variablesNum - 1, -1, -1):
            if (position_list[i] >= block):
                position_list[i] -= block
                if (i > 0):
                    position_list[i - 1] += 1
                else:
                    final_input = 1;
        if (final_input == 1):            
            position_list = [0] * variablesNum
            block = block * 2;
                 
        if variables == None:
            print("No possible solution in these constraints QAQ")
            continue        
                
        returnValueList = simulateFunction(variables)

        if (len(returnValueList) == constraintsId and constraintsId == constraintsNum and returnValueList[constraintsId - 1].isValid):
        # give the final input for the original constraints
            print("Final input for the fuzzer:", variables)
            return parameter_result, sampling_count, variables

        if (len(returnValueList) <= constraintsId):
            print("The solution variable is not suitable for the problem. QAQ")
            print(variables)
            continue
        
        variables.append(1)
        A.append(variables)
        # guarantee the linear independence
        rank = np.linalg.matrix_rank(A)
        if (rank < len(A) and rank < variablesNum + 1 - equalNum):
            A.pop()
            continue
        
        # add a line into matrix A
        isLinear = 1;
        B.append(returnValueList[constraintsId].intValue)

    print("A = ", A)
    print("B = ", B)
    C = matlabSolver(A, B)     
    if (C == None):
        print("No possible solution for the linear equations! TAT");
        print("Cannot solve constarint id", constraintsId);
        return parameter_result, sampling_count, None;

    if constraintsId == 0:
        parameter_result.append(C)        
    else:
        parameter_result = np.vstack((parameter_result, C))
    return parameter_result, sampling_count, variables

# set parameters

MAX = 1000
MIN = -1000

MAX = (1<<31)-1
MIN = -(1<<31)
print(MAX, MIN)

parameter_a = [[0, 1, 1], [1, 0, 0], [1, -1, 0]]
parameter_b = [4, 3, 2]
parameter_isEqual = [True, False, True]
constraintsNum = 3

variablesNum = 3
variables_lower = [MIN, MIN, MIN]
variables_upper = [MAX, MAX, MAX]

total_count = 0
exeTimes = 1
for T in range(0, exeTimes):
    parameter_result = []  # the result of simulation
    sampling_count = 0 # the number of sampling
    for i in range(0, constraintsNum + 1):
        parameter_result, sampling_count, variables = solveConstraint(i, parameter_isEqual, 
                                                                      constraintsNum, variablesNum, variables_lower, 
                                                                      variables_upper, parameter_result, sampling_count)
        if (variables == None):
            print("Non-linear constraints! TAT");
            break;
    print("sampling counts: ", sampling_count)
    total_count = total_count + sampling_count

print("parameter_result", parameter_result)
print(exeTimes, " total counts: ", total_count);
