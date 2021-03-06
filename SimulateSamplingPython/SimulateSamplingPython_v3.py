import math
import numpy as np
import pulp
from scipy.linalg import solve
import sys
import matlab.engine

parameter_a = [[]]
parameter_b = []

class Value:
    intValue = 0
    isValid = False

def getValue(constraintsId, variables):
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

def simulateFunction(variables):
    returnValueList = []
    for i in range(0, constraintsNum):
        returnValue = getValue(i, variables);
        returnValueList.append(returnValue);
        if (returnValue.isValid == False):
           break
    return returnValueList

def solve_ilp(objective, constraints):
    print("objective", objective)
    print("constraaints", constraints)
    prob = pulp.LpProblem('LP1', pulp.LpMinimize)
    prob += objective
    for cons in constraints:
        prob += cons
    print("prob", prob);    
    status = prob.solve(pulp.GLPK(options=['--mipgap', '0.000001', '--cgr']));
    # status = prob.solve(pulp.COIN_CMD(fracGap = 0));
    # status = prob.solve(pulp.PULP_CBC_CMD());  # default
    if status != 1:
        # print('status', status)
        # LpStatus = {-3: 'Undefined', -2: 'Unbounded', -1: 'Infeasible', 0: 'No...
        return None
    else:
        return [v.varValue.real for v in prob.variables()]

# return a variable which suits
# beginning 0 ~ constraintsId - 1 constraints
def getVariables(constraintsId, parameter_isEqual, 
                 variablesNum, variables_lower, variables_upper, parameter_result):    
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
        pulp_variables = [];
        for i in range(1, variablesNum + 1):
            #temp_lowBound = np.random.randint(variables_lower[i - 1], high = variables_upper[i - 1]);
            #temp_upBound = np.random.randint(variables_lower[i - 1], high = variables_upper[i - 1]);
            temp_lowBound = round(variables_lower[i-1] * np.random.rand());
            temp_upBound = round(variables_upper[i-1] * np.random.rand());
            if (temp_lowBound >= temp_upBound):
                temp = temp_lowBound;
                temp_lowBound = temp_upBound;
                temp_upBound = temp;                
            pulp_variables.append(pulp.LpVariable('X%d' % i, cat = pulp.LpInteger, 
                                                  lowBound = temp_lowBound, upBound = temp_upBound));
        
        pulp_variables.append(pulp.LpVariable('t', cat = pulp.LpInteger))
        # print("pulp_variables", pulp_variables)

        # Objective function
        F = [0] * variablesNum
        F.append(1)
        objective = sum([F[i] * pulp_variables[i] for i in range(0, V_NUM)]) 
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
            # add fi <= 0
            temp_a[variablesNum] = -1
            constraints.append(sum([temp_a[i] * pulp_variables[i] for i in range(0, V_NUM)]) <= -temp_b)
            if (parameter_isEqual[k]):   
                # add fi + t >= 0 [for the equation constraints
                temp_a[variablesNum] = 1
                constraints.append(sum([temp_a[i] * pulp_variables[i] for i in range(0, V_NUM)]) >= -temp_b) 
                ##  add fi == 0
                temp_a[variablesNum] = 0;
                constraints.append(sum([temp_a[i] * pulp_variables[i] for i in range(0, V_NUM)]) == -temp_b) 
        temp_v = [0] * variablesNum
        for k in range(0, variablesNum):
            temp_v[k] = 1
            constraints.append(sum([temp_v[i] * pulp_variables[i] for i in range(0, variablesNum)]) >= variables_lower[k])
            constraints.append(sum([temp_v[i] * pulp_variables[i] for i in range(0, variablesNum)]) <= variables_upper[k])
            temp_v[k] = 0
        # print("constraints", constraints)

        variables = solve_ilp(objective, constraints)
        # print("variables", variables)

        ## TODO 怎么设置停机时间   
        #while (True):     
        #    if (variables == None):
        #        # no possible solution for the ilp problem.
        #        print("No possible solution! Test another random constraints QAQ")
        #        variables = getVariables(constraintsId, parameter_isEqual, 
        #             variablesNum, variables_lower, variables_upper, parameter_result);
        #        return variables
        #    # delete the last "t" variable
        #    variables.pop()
        #    # ganrantee the variables fit the constraints
        #    if (len(returnValueList) == constraintsId and constraintsId == constraintsNum and returnValueList[constraintsId - 1].isValid):
        #       return variables;

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
    equalNum = 0;
    for i in range(0, constraintsId):
        if (parameter_isEqual[i]):
            equalNum += 1;

    while (len(A) < variablesNum + 1):        
        variables = getVariables(constraintsId, parameter_isEqual, 
                variablesNum, variables_lower, variables_upper, parameter_result);   
        sampling_count += 1;

        returnValueList = simulateFunction(variables);

        if (len(returnValueList) == constraintsId and constraintsId == constraintsNum and returnValueList[constraintsId - 1].isValid):
        # give the final input for the original constraints
            print("Final input for the fuzzer:", variables);
            return parameter_result, sampling_count, variables

        if (len(returnValueList) <= constraintsId):
            print("The solution variable is not suitable for the problem. QAQ")
            print(variables);
            continue
        
        variables.append(1);
        A.append(variables);
        # guarantee the linear independence
        rank = np.linalg.matrix_rank(A);
        if (rank < len(A) and rank < variablesNum + 1 - equalNum):
            A.pop();
            continue;

        B.append(returnValueList[constraintsId].intValue)

    print("A = ", A)
    print("B = ", B)
    C = matlabSolver(A, B) 

    if constraintsId == 0:
        parameter_result.append(C)        
    else:
        parameter_result = np.vstack((parameter_result, C))
    return parameter_result, sampling_count, variables

# set parameters
MAX = 2147483647
MIN = -2147483648
MAX = 1000
MIN = -1000

# g1(x1; x2) := x1 - x2 == 0
# if (x1 - x2 == 0) {
#parameter_a = [[1, -1]]
#parameter_b = [0]
#parameter_isEqual = [True]
#constraintsNum = 1

#variablesNum = 2
#variables_lower = [MIN, MIN]
#variables_upper = [MAX, MAX]

##
## f1(x1; x2) := x1 + 1 <= 0
## g1(x1; x2) := x1 - x2 == 0
## if (x1 - x2 == 0) {
##      if (x1 + 1 <= 0) {
#parameter_a = [[1, -1], [1, 0]]
#parameter_b = [0, 1]
#parameter_isEqual = [True, False]
#constraintsNum = 2

#variablesNum = 2
#variables_lower = [MIN, MIN]
#variables_upper = [MAX, MAX]

#parameter_a = [[1, -1, 0], [1, 0, 0]]
#parameter_b = [2, 3]
#parameter_isEqual = [True, False]
#constraintsNum = 2
##

parameter_a = [[1, -1, 0], [1, 0, 0], [0, 1, 1]]
parameter_b = [2, 3, 4]
parameter_isEqual = [True, False, True]
constraintsNum = 3

variablesNum = 3
variables_lower = [MIN, MIN, MIN]
variables_upper = [MAX, MAX, MAX]

total_count = 0;
exeTimes = 1;
for T in range(0, exeTimes):
    parameter_result = []  # the result of simulation
    sampling_count = 0 # the number of sampling
    for i in range(0, constraintsNum + 1):
        parameter_result, sampling_count, variables = solveConstraint(i, parameter_isEqual, 
                                                                      constraintsNum, variablesNum, variables_lower, 
                                                                      variables_upper, parameter_result, sampling_count);
    print("sampling counts: ", sampling_count);
    total_count = total_count + sampling_count;

print("parameter_result", parameter_result)
print(exeTimes, " total counts: ", total_count);
