import numpy as np
import time
from typing import List
from helpers import l_u_decomposition,l_u_decomposition_optimized
import math

class OutputSolution:
  def __init__(self, x: np.ndarray, iterations: int, errors: List[float], does_converge: bool, time: float):
      self.x = x
      self.iterations = iterations
      self.errors = errors
      self.does_converge = does_converge
      self.time = time
      
  
def solve_jacobi_optimized(
    A: np.ndarray,
    b: np.ndarray,
    precision: float = 1e-9,
    max_error: float = 1e9,
) -> OutputSolution:
   # based on https://www.quantstart.com/articles/Jacobi-Method-in-Python-and-NumPy/
    """
    x_{k+1} = D^{-1} (b - R x_k),
    where D = diag(A), R = A - D.
    """
    start_time = time.time()

    D = np.diag(A) # Getting the diagonal of A
    if np.any(D == 0):
        raise ZeroDivisionError("Matrix A has 0 on diagonal!")
      
    D_inv = 1.0 / D                            # Create D^{-1}

    # R is the rest of the matrix, without the diagonal 
    R = A - np.diagflat(D)                     # R = A - D

    x = np.zeros_like(b, dtype=float).ravel()              # shape (n,)
    errors: list[float] = [float('Inf')]       # list of errors
    iterations = 0
    does_converge = True

    while errors[-1] > precision:
        x_new = (b.ravel() - R @ x) * D_inv

        err = np.linalg.norm(A @ x_new - b.ravel())
        errors.append(err)

        if err > max_error:
            does_converge = False
            x = x_new
            break

        x = x_new
        iterations += 1

    elapsed_time = time.time() - start_time
    
    # errors[1:] to remove the first error which is always Inf
    return OutputSolution(x=x.reshape(-1,1),
                          iterations=iterations,
                          errors=errors[1:],
                          does_converge=does_converge,
                          time=elapsed_time)

def solve_jacobi(A: np.ndarray, b: np.ndarray, precision: float = 1e-9, max_error: float = 1e9) -> OutputSolution:
  # algorithm source https://en.wikipedia.org/wiki/Jacobi_method
  start_time = time.time()
  
  n = len(A)
  
  errors = [float('Inf')]
  iterations = 0
  does_converge = True
  
  x = np.zeros((n, 1), dtype=float)
  
  while errors[-1] > precision:
    new_x = np.copy(x)
    
    # A = L + U + D, L is lower triangular, U is upper triangular, D is diagonal
    for i in range(n):
      sum_L_U = sum(A[i][j] * x[j][0] for j in range(n) if i != j) # K = (L + U) * x
      new_x[i][0] = (b[i][0] - sum_L_U) / A[i][i] # (b - K) / D
      
    
    current_error = np.linalg.norm(A @ new_x - b)
    errors.append(current_error)
    
    if current_error > max_error:
       does_converge = False
       break
     
    x = new_x
    iterations += 1
    
  elapsed_time = time.time() - start_time
  return OutputSolution(x=x,
                        iterations=iterations,
                        errors=errors[1:],
                        does_converge=does_converge,
                        time=elapsed_time)

def solve_gauss_seidel_optimized(A: np.ndarray, b: np.ndarray, precision: float = 1e-9, max_error: float = 1e9) -> OutputSolution:
  # based on https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
  #
  #  x_i^(k+1) = (b_i - sum_{j<i} a_{ij} x_j^(k+1) - sum_{j>i} a_{ij} x_j^(k))
  #    ---------------------------------------------------------------
  #                             a_{ii}
  #
  start_time = time.time()
  
  D = np.diag(A)                        
  if np.any(D == 0):
     raise ZeroDivisionError("Matrix A has 0 on diagonal!")

  errors = [float('Inf')]
  iterations = 0
  does_converge = True
  
  n = A.shape[0]
  
  A_lower = np.tril(A, k=-1) # lower triangular part of A     
  A_upper = np.triu(A, k=+1) # upper triangular part of A  
  
  b_vec = b.ravel() # shape (n,)
  x = np.zeros_like(b, dtype=float).ravel() # shape (n,)

  D_inv = 1.0 / D
  
  while errors[-1] > precision:
    x_new = np.zeros_like(x, dtype=float)
    for i in range(n):
        sum1 = A_lower[i, :i] @ (x_new[:i])
        sum2 = A_upper[i, i+1:] @ (x[i+1:])
        x_new[i] = (b_vec[i] - sum1 - sum2) * D_inv[i]
    
    x = x_new

    current_error = np.linalg.norm((A @ x) - b_vec)
    errors.append(current_error)
    iterations += 1
    
    if current_error > max_error:
       does_converge = False
       break
     
    
  elapsed_time = time.time() - start_time
    
  # errors[1:] to remove the first error which is always Inf
  return OutputSolution(x=x.reshape(-1,1),
                          iterations=iterations,
                          errors=errors[1:],
                          does_converge=does_converge,
                          time=elapsed_time)

def solve_gauss_seidel(A: np.ndarray, b: np.ndarray, precision: float = 1e-9, max_error: float = 1e9) -> OutputSolution:
  # based on https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
  start_time = time.time()
  
  n = len(A)
  
  errors = [float('Inf')]
  iterations = 0
  does_converge = True
  
  x = np.zeros((n, 1), dtype=float)
  
  while errors[-1] > precision:
    
    x_new = np.copy(x)
    for i in range(n):
      
      sum1 = sum(A[i][j] * x[j][0] for j in range(i+1, n)) 
      sum2 = sum(A[i][j] * x_new[j][0] for j in range(i))
      
      x_new[i][0] = (b[i][0] - sum1 - sum2) / A[i][i]
      
    
    current_error = np.linalg.norm(A @ x_new - b)
    errors.append(current_error)
    
    x = x_new
    iterations += 1
    
    if current_error > max_error:
       does_converge = False
       break
     

    
  elapsed_time = time.time() - start_time
    
  # errors[1:] to remove the first error which is always Inf
  return OutputSolution(x=x,
                        iterations=iterations,
                        errors=errors[1:],
                        does_converge=does_converge,
                        time=elapsed_time)

def solve_l_u_optimized(A: np.ndarray, b: np.ndarray, precision: float = 1e-9) -> OutputSolution:
   # idea of an algorithm https://www.cl.cam.ac.uk/teaching/1314/NumMethods/supporting/mcmaster-kiruba-ludecomp.pdf
   start_time = time.time()
   
   L,U = l_u_decomposition_optimized(A)
   
   n = A.shape[0]
   b_vec = b.ravel() # shape (n,)
   x = np.zeros_like(b_vec, dtype=float) # shape (n,)
   d = np.zeros_like(b_vec, dtype=float) # shape (n,)

   for i in range(n):
    d[i] = b_vec[i] - L[i, :i] @ d[:i] 
     
   for i in range(n-1, -1, -1):
      x[i] = (d[i] - (U[i, i+1:] @ x[i+1:])) / U[i][i]
   
   norm = np.linalg.norm((A @ x) - b_vec)
   
   elapsed_time = time.time() - start_time
   
   does_converge = norm < precision
   
   return OutputSolution(x=x.reshape(-1, 1),  # dopasowujemy do poprzedniego interfejsu
                          iterations=0,
                          errors=[norm],
                          does_converge=does_converge,
                          time=elapsed_time)
 
def solve_l_u(A: np.ndarray, b: np.ndarray, precision: float = 1e-9) -> OutputSolution:
   # based on https://www.cl.cam.ac.uk/teaching/1314/NumMethods/supporting/mcmaster-kiruba-ludecomp.pdf
   start_time = time.time()
   
   l,u = l_u_decomposition(A)
   
   n = len(A)
   d = np.zeros((n, 1), dtype=float)
   x = np.zeros((n, 1), dtype=float)
   
   for i in range(n):
     d[i][0] = b[i][0] - sum(l[i][j] * d[j][0] for j in range(i))
     
   for i in range(n-1, -1, -1):
      x[i][0] = (d[i][0] - sum(u[i][j] * x[j][0] for j in range(i+1, n))) / u[i][i]
   
   norm = np.linalg.norm(A @ x - b)
   
   elapsed_time = time.time() - start_time
   
   does_converge = norm < precision
   
   return OutputSolution(x=x,
                          iterations=0,
                          errors=[norm],
                          does_converge=does_converge,
                          time=elapsed_time)