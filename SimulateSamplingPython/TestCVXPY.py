# Ref: https://stackoverflow.com/questions/43163682/integer-linear-least-squares
import numpy as np
import cvxpy

np.random.seed(123) # for reproducability

# generate A and B
m, n = 4, 4
A = np.random.randn(m,n)
B = np.random.randn(m)

A = [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
B = [9, 6, 3, 11]

A = [[3.0, 1.0, 5.0, 1.0], [2.0, 4.0, 6.0, 1.0], [1.0, 9.0, 9.0, 1.0], [7.0, 6.0, 8.0, 1.0]]
B = [9, 6, 3, 11]
x0 = [1, -1, 1, 2]

A = [[614071408, 1569877410, -1844044328, 1], [1631203151, 2039156791, -1073442812, 1], [-1736958565, 2037741599, 635288874, 1], [-210093103, 354655146, -718127209, 1]] 
B = [-955806000, -407953638, -3774700162, -564748247]


A = [[-68308474.0, -68308472.0, -341366010.0, 1], [-47581579.0, -47581577.0, -1504785200.0, 1], [-18930757.0, -18930755.0, -2143493400.0, 1], [-19290484.0, -19290482.0, -2057089700.0, 1]]
B =  [-68308471.0, -47581576.0, -18930754.0, -19290481.0]
print("A = ", A)
print("rank A = ", np.linalg.matrix_rank(A))
print("B = ", B)

# declare the integer-valued optimization variable
x = cvxpy.Int(n)

# set up the L2-norm minimization problem
obj = cvxpy.Minimize(cvxpy.norm(list(map(list, zip(*A))) * x - B, 1))
prob = cvxpy.Problem(obj)

# solve the problem using an appropriate solver
# sol = prob.solve(solver = 'ECOS_BB')
sol = prob.solve()

# the optimal value of x is 


C = x.value
C = np.matrix.tolist(C)
C = list(map(list, zip(*C)))[0]
print("Int: C = ", C)
for i in range(0, len(C)):
    C[i] = round(C[i])
print("Round Int: C = ", C)   

for i in range(0, 4):
    sum = 0;
    for j in range(0, 4):
        sum = sum + A[i][j] * C[j];
    print("rest ", i, " ", B[i] - sum);
