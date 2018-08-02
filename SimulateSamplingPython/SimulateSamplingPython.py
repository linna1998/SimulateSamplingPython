import numpy as np
from scipy.linalg import solve
import sys
parameter_a = [[]];
parameter_b = [];
parameter_isEqual = [];
constraintsNum = 0;
variables = [];
variablesNum = 0;
variables_lower = [];  # the bounds of the certain variable
variables_upper = [];

parameter_result = [];  # the result of simulation
sampling_count = 0; # the number of sampling

class Value:
    intValue = 0;
    isValid = False;

def getValue(constraintsId):
    returnValue = Value();
    for i in range(0, variablesNum):
        returnValue.intValue += parameter_a[constraintsId][i] * variables[i];
    returnValue.intValue += parameter_b[constraintsId];
    if ((not parameter_isEqual[constraintsId]) and returnValue.intValue <= 0):
        returnValue.isValid = True;
    elif (parameter_isEqual[constraintsId] and returnValue.intValue == 0):
        returnValue.isValid = True;
    else:
        returnValue.isValid = False;
    return returnValue;

def simulateFunction():
    returnValueList = [];
    for i in range(0, constraintsNum):
        returnValue = getValue(i);
        returnValueList.append(returnValue);
        if (returnValue.isValid == False):
           break;
    return returnValueList;

# f1(x1; x2) := x1 + 1 <= 0
# g1(x1; x2) := x1 - x2 == 0
# if (x1 - x2 == 0) {
#      if (x1 + 1 <= 0) {

# set parameters
MAX = (1<<31) - 1;
MIN = -(1<<31);
# change the order of two 'if' statements
parameter_a = [[1, 0], [1, -1]];
parameter_b = [1, 0];
parameter_isEqual = [False, True];
#parameter_a = [[1, -1], [1, 0]];
#parameter_b = [0, 1];
#parameter_isEqual = [True, False];

constraintsNum = 2;
variablesNum = 2;
variables_lower = [MIN, MIN];
variables_upper = [MAX, MAX];

# Random to get the variables
def solveConstraint(constraintsId):
    global variables, parameter_result, sampling_count
    # in linear equations
    A = [];
    B = [];
    while (len(A) < variablesNum + 1) :
        sampling_count += 1;
        if (sampling_count % 1000 == 0):
            print("temp sampling counts: ", sampling_count);
        variables = [];
        for j in range(0, variablesNum):
            variables.append(np.random.randint(variables_lower[j], high = variables_upper[j]));
        returnValueList = simulateFunction();
        if (len(returnValueList) <= constraintsId):
            continue;
        variables.append(1);
        A.append(variables);    
        B.append(returnValueList[constraintsId].intValue);

    C = solve(A, B);
    for i in range(0, len(C)):
        C[i] = round(C[i]);
    if constraintsId == 0:
        parameter_result.append(C);
    else:
        parameter_result = np.vstack((parameter_result, C));

for i in range(0, constraintsNum):
    solveConstraint(i);
    for j in range(0, variablesNum):
        if (parameter_result[i][j] != parameter_a[i][j]):
            print("QAQ");
            break;
print(parameter_result);
print("total sampling counts: ", sampling_count);
    
        
