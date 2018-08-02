# Ref: https://stackoverflow.com/questions/43163682/integer-linear-least-squares
import numpy as np
import cvxpy

#np.random.seed(123) # for reproducability

## generate A and y
#m, n = 10, 10
#A = np.random.randn(m,n)
#B = np.random.randn(m)

## declare the integer-valued optimization variable
#x = cvxpy.Int(n)

A = [[614071408, 1569877410, -1844044328, 1], [1631203151, 2039156791, -1073442812, 1], [-1736958565, 2037741599, 635288874, 1], [-210093103, 354655146, -718127209, 1]] 
B = [-955806000, -407953638, -3774700162, -564748247]
#x = cvxpy.Int(4)
x = cvxpy.Int(4)
x2 = cvxpy.Variable(4)

# set up the L2-norm minimization problem
obj = cvxpy.Minimize(cvxpy.norm(A * x - B, 2))
prob = cvxpy.Problem(obj)
obj2 = cvxpy.Minimize(cvxpy.norm(A * x2 - B, 2))
prob2 = cvxpy.Problem(obj2)

print("prob is DCP:", prob.is_dcp())
print("prob2 is DCP:", prob2.is_dcp())

# solve the problem using an appropriate solver
sol = prob.solve(solver = 'ECOS_BB')
sol2 = prob2.solve(solver = 'ECOS_BB')

# sol = prob.solve(solver = 'SCS')
# sol2 = prob2.solve(solver = 'SCS')

print("A = ", A)
print("B = ", B)
# the optimal value of x is 
print("x = ", x.value)
print("rest = ", A * x - B)

# the optimal value of x is     
C = x.value
print("Int: C = ", C)
C = x2.value
print("Variable: C = ", C)

for i in range(0, 4):
    sum = 0;
    for j in range(0, 4):
        sum = sum + A[i][j] * x.value[j];
    print("rest ", i, " ", B[i] - sum);

for i in range(0, 4):
    sum = 0;
    for j in range(0, 4):
        sum = sum + A[i][j] * x2.value[j];
    print("rest ", i, " ", B[i] - sum);