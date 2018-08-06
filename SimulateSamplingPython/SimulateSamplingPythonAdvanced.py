import cvxpy
import math
import numpy as np
import pulp
from scipy.linalg import solve
import sys
import matlab.engine

##LIMIT = 0  # the abs limit for all random variables, in this version is useless
parameter_a = [[]]
parameter_b = []
#parameter_isEqual = []
#constraintsNum = 0
#variables = []
#variablesNum = 0
#variables_lower = []  # the bounds of the certain variable
#variables_upper = []

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
    print(objective)
    print(constraints)
    prob = pulp.LpProblem('LP1', pulp.LpMinimize)
    prob += objective
    for cons in constraints:
        prob += cons
    print(prob)
    status = prob.solve()
    if status != 1:
        # print('status', status)
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
            temp_lowBound = np.random.randint(variables_lower[i - 1], high = variables_upper[i - 1]);
            temp_upBound = np.random.randint(variables_lower[i - 1], high = variables_upper[i - 1]);
            if (temp_lowBound >= temp_upBound):
                temp = temp_lowBound;
                temp_lowBound = temp_upBound;
                temp_upBound = temp;                
            pulp_variables.append(pulp.LpVariable('X%d' % i, cat = pulp.LpInteger, 
                                                  lowBound = temp_lowBound, upBound = temp_upBound));

        #pulp_variables = [pulp.LpVariable('X%d' % i, cat = pulp.LpInteger, 
        #                                  lowBound = np.random.randint(variables_lower[i - 1], high = variables_upper[i - 1]), 
        #                                  upBound = np.random.randint(randon_lower, high = variables_upper[i - 1])) 
        #                                  # lowBound = max(variables_lower[i - 1], -LIMIT) * np.random.random(), 
        #                                  # upBound = min(variables_upper[i - 1], LIMIT) * np.random.random()) 
        #                  for i in range(1, variablesNum + 1)]
        
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

        # TODO 怎么设置停机时间
        while (variables == None):
            print("No possible solution! QAQ")
            variables = getVariables(constraintsId, parameter_isEqual, 
                 variablesNum, variables_lower, variables_upper, parameter_result);
            return variables
        # delete the last "t" variable
        variables.pop()
        return variables                
    
# return a variable which suits
# beginning 0 ~ constraintsId - 1 constraints
def getVariables2(constraintsId, parameter_isEqual, 
                 variablesNum, variables_lower, variables_upper, parameter_result):    
    
    variables = []
    if constraintsId == 0:
         for j in range(0, variablesNum):
            variables.append(np.random.randint(variables_lower[j], high = variables_upper[j]))
         return variables
    else:
        # Randomly get a variable
        # which suits for 0 ~ constraintsId - 1 constraints
        prob = pulp.LpProblem("getVariables2", pulp.LpMinimize);

        V_NUM = variablesNum + 1;
        
        # Variables
        pulp_variables = [pulp.LpVariable('X%d' % i, cat = pulp.LpInteger, 
                                          lowBound = variables_lower[i - 1] * np.random.random(), 
                                          upBound = variables_upper[i - 1] * np.random.random()) 
                                          # lowBound = max(variables_lower[i - 1], -LIMIT) * np.random.random(), 
                                          # upBound = min(variables_upper[i - 1], LIMIT) * np.random.random()) 
                          for i in range(1, variablesNum + 1)]
        pulp_variables.append(pulp.LpVariable('t', cat = pulp.LpInteger))
        # print("pulp_variables", pulp_variables)

        # Objective function
        F = [0] * variablesNum
        F.append(1)
        objective = sum([F[i] * pulp_variables[i] for i in range(0, V_NUM)]) 
        prob += objective;
        # print(objective)

        # Build Constraints
        constraints = []
        temp_constraint = sum([F[i] * pulp_variables[i] for i in range(0, V_NUM)]) <= 0;
        constraints.append(temp_constraint);
        prob += temp_constraint;
        for k in range(0, constraintsId):
            temp_a = list(parameter_result[k])
            temp_b = temp_a[variablesNum]
            # add fi - t <= 0
            temp_a[variablesNum] = -1;
            temp_constraint = sum([temp_a[i] * pulp_variables[i] for i in range(0, V_NUM)]) <= -temp_b;
            constraints.append(temp_constraint);
            prob += temp_constraint;
            if (parameter_isEqual[k]):   
                # add fi + t >= 0 [for the equation constraints
                temp_a[variablesNum] = 1;
                temp_constraint = sum([temp_a[i] * pulp_variables[i] for i in range(0, V_NUM)]) >= -temp_b;
                constraints.append(temp_constraint); 
                prob += temp_constraint;
        temp_v = [0] * variablesNum
        for k in range(0, variablesNum):
            temp_v[k] = 1;
            temp_constraint = sum([temp_v[i] * pulp_variables[i] for i in range(0, variablesNum)]) >= variables_lower[k];
            constraints.append(temp_constraint); 
            prob += temp_constraint;
            temp_constraint = sum([temp_v[i] * pulp_variables[i] for i in range(0, variablesNum)]) <= variables_upper[k];
            constraints.append(temp_constraint); 
            prob += temp_constraint;
            temp_v[k] = 0
        # print("constraints", constraints)        

        
        # pulp.COINMP_DLL().solve(prob);
        # pulp.CPLEX().solve(prob);
        # pulp.GLPK().solve(prob);
        # pulp.LpSolver().solve(prob);
        prob.solve();        
        # prob.solve(pulp.solvers.PULP_CBC_CMD(fracGap=0.00000001));
        print("before prob.solve")

        prob.solve(pulp.solvers.PULP_CBC_CMD(fracGap = 0, options = ['integerTolerance', '1e-8']));
        print(prob);
        variables = prob.variables();        
        for i in range(0, V_NUM):
            variables[i] = variables[i].value();

        # variables = solve_ilp(objective, constraints)

        print("variables", variables)
        if (variables == None):
            print("No possible solution! QAQ")
            return variables
        # delete the last "t" variable
        variables.pop()
        return variables                

def matlabGetVariables(constraintsId, parameter_isEqual, 
                 variablesNum, variables_lower, variables_upper, parameter_result):
    eng = matlab.engine.start_matlab();
    parameter_isEqual = matlab.logical(parameter_isEqual);
    variables_lower = matlab.double(variables_lower);
    variables_upper = matlab.double(variables_upper);
    parameter_result = matlab.double(parameter_result);
    variables = eng.getVariables(constraintsId, parameter_isEqual, 
                 variablesNum, variables_lower, variables_upper, parameter_result)
    variables = list(map(list, zip(*variables)))[0]
    print(variables);
    return variables


def cvxpySolver(A, B):
    # declare the integer-valued optimization variable
    x = cvxpy.Int(variablesNum + 1)

    # set up the L2-norm minimization problem
    obj = cvxpy.Minimize(cvxpy.norm(list(map(list, zip(*A))) * x - B, 1))
    #obj = cvxpy.Minimize(cvxpy.norm(A * x - B, 2))
    prob = cvxpy.Problem(obj)   

    # solve the problem using an appropriate solver
    sol = prob.solve(solver = 'ECOS_BB')    

    # the optimal value of x is
    C = x.value
    C = np.matrix.tolist(C)
    C = list(map(list, zip(*C)))[0]
    print("Int: C = ", C)
    for i in range(0, len(C)):
        C[i] = round(C[i])
    print("Round Int: C = ", C)   
    return C

def lstsqSolver(A, B):
    C = np.linalg.lstsq(A, B, rcond = -1)[0] 
    # C = solve(A, B);
    print("C = ", C)
    # Remark: 不能直接弄round到整数，
    # 因为在lstsq的情况下，可能本来是(1, 0, 1) / (0, 1, 1)，
    # 结果被解出了(0.5, 0.5, 1) 然后如果一下近似就凉掉了

    # 2018.7.24.  16:20
    # 但是好像不round也会出问题，
    # 所以只能去想有没有一个尽量都解出整数的解法
    for i in range(0, len(C)):
        C[i] = round(C[i])
    print("Round C = ", C)
    return C

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
    while (len(A) < variablesNum + 1):
        variables = getVariables(constraintsId, parameter_isEqual, 
                variablesNum, variables_lower, variables_upper, parameter_result);   
        sampling_count += 1;
        
        # variables = [];
        # for j in range(0, variablesNum):
        #     variables.append(np.random.randint(variables_lower[j], high =
        #     variables_upper[j]));

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
        # B.append(returnValueList[constraintsId - 1].intValue);
        B.append(returnValueList[constraintsId].intValue)

    print("A = ", A)
    print("B = ", B)

    # C = cvxpySolver(A, B)
    # C = lstsqSolver(A, B)
    C = matlabSolver(A, B)

    #for i in range(0, len(A)):
    #    sum = 0
    #    for j in range(0, len(C)):
    #        sum = sum + A[i][j] * C[j]
    #    print("rest ", i, " ", B[i] - sum)

    ## change the LIMIT for variables
    #LIMIT = max(LIMIT, max(C) * 100)

    if constraintsId == 0:
        parameter_result.append(C)        
    else:
        parameter_result = np.vstack((parameter_result, C))
    return parameter_result, sampling_count, variables

# set parameters
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
for T in range(0, 1):
    parameter_result = []  # the result of simulation
    sampling_count = 0 # the number of sampling
    for i in range(0, constraintsNum + 1):
        parameter_result, sampling_count, variables = solveConstraint(i, parameter_isEqual, 
                                                                      constraintsNum, variablesNum, variables_lower, 
                                                                      variables_upper, parameter_result, sampling_count)   
    total_count = total_count + sampling_count;

print("parameter_result", parameter_result)
# print("sampling counts: ", sampling_count);
print("prob.solve(pulp.solvers.PULP_CBC_CMD(fracGap = 0, options = ['IntegerTolerance', '1e-8']));");
print("100 total counts: ", total_count);
