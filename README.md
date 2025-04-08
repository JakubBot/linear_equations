## Project
The aim of this study was to implement and analyze selected numerical methods for solving systems of linear equations. 
Specifically, two iterative methods (Jacobi and Gauss-Seidel) and one direct method (LU factorization) were implemented 
and tested. The test systems are based on large banded matrices, which are common in real-world engineering and 
physical problems such as structural analysis, fluid dynamics, wave propagation, and thermal simulations. Each method 
was evaluated in terms of convergence, accuracy, and computational performance. The results were visualized with plots 
and discussed comparatively. 

**The report** can be found here - [Linear Equations Documentation](https://github.com/JakubBot/linear_equations/blob/master/docs/Linear%20Equations.pdf)

Index: 197839 

### Task A

Create a system of equations for a1=5+e, where e is the fourth digit of your index,
a2=a3=-1.The size of the matrix N is defined in section 2 of this manual. b is a vector
of length N, whose first element is the value of sin(n-(f+1)), where f is the third digit of your index.

### Task B

Implement iterative methods for solving systems of linear equations:
Jacobi and Gauss-Seidel. Describe how many iterations each method needs to determine the
solution of the system of equations from task A, assuming that the condition for completing the calculation
is to achieve a residuum norm of less than 10^-9.For both methods, show on the graph
how the norm of the residuum changes in successive iterations performed to determine the
solution(y-axis on a logarithmic scale).Compare the running times of both algorithms.

### Task C

Define the system of equations for a1=3, a_2 = a_3 = -1 and let  N be the vector that corresponds to the specification in the task for matrix A. Do the iterative methods converge for such values of the matrix elements A? For both methods, the plot shows how the residual norm changes in successive iterations (the Y-axis is in logarithmic scale).

### Task D

Implement a direct method for solving systems of linear equations: the LU decomposition method. Use this implementation to find the solution to the equation from Task C. What is the value of the residual norm in this case?

### Task E

Create a plot showing the dependency of time taken to determine the solution for three given methods as a function of the number of unknowns 
N={100,500,1000,2000,3000} for the matrix described in task A. The plots should illustrate identical data, with the first plot having a linear scale on the Y-axis, and the second one having a logarithmic scale on the Y-axis