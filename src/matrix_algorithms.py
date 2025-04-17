import numpy as np
import time
from typing import List
from helpers import l_u_decomposition
import math

class OutputSolution:
  def __init__(self, x: np.ndarray, iterations: int, errors: List[float], does_converge: bool, time: float):
      self.x = x
      self.iterations = iterations
      self.errors = errors
      self.does_converge = does_converge
      self.time = time
      
  
def solve_jacobi(
    A: np.ndarray,
    b: np.ndarray,
    precision: float = 1e-9,
    max_iter: int = 10000,
    max_error: float = 1e9,
) -> OutputSolution:
   # algorithm source https://www.quantstart.com/articles/Jacobi-Method-in-Python-and-NumPy/
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

    x = np.zeros_like(b).ravel()               # shape (n,)
    errors: list[float] = []
    does_converge = True

    for k in range(1, max_iter + 1):
        x_new = D_inv * (b.ravel() - R @ x)

        resid = A @ x_new - b.ravel()
        err = np.linalg.norm(resid)
        errors.append(err)

        if err < precision:
            x = x_new
            break
        if err > max_error:
            does_converge = False
            x = x_new
            break

        x = x_new

    elapsed_time = time.time() - start_time
    return OutputSolution(x=x.reshape(-1,1),
                          iterations=k,
                          errors=errors,
                          does_converge=does_converge,
                          time=elapsed_time)

def solve_gauss_seidel(A: np.ndarray, b: np.ndarray, precision: float = 1e-9, max_error: float = 1e9) -> OutputSolution:
  # algorithm source https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
  start_time = time.time()
  
  errors = [float('Inf')]
  iterations = 0
  does_converge = True
  
  x = np.ones((len(A), 1), dtype=float)
  
  while errors[-1] > precision:
    
    x_new = np.copy(x)
    for i in range(len(A)):
      
      sum1 = sum(A[i][j] * x[j][0] for j in range(i+1, len(A))) 
      sum2 = sum(A[i][j] * x_new[j][0] for j in range(i))
      
      x_new[i][0] = (b[i][0] - sum1 - sum2) / A[i][i]
      
    
    current_error = np.linalg.norm(np.dot(A, x_new) - b)
    errors.append(current_error)
    
    x = x_new
    iterations += 1
    
    if current_error > max_error:
       does_converge = False
       break
     

    
  elapsed_time = time.time() - start_time
    
  # errors[1:] to remove the first error which is always Inf
  return OutputSolution(x, iterations,errors[1:], does_converge,elapsed_time)

def solve_l_u(A: np.ndarray, b: np.ndarray, precision: float = 1e-9) -> OutputSolution:
   # idea of an algorithm https://www.cl.cam.ac.uk/teaching/1314/NumMethods/supporting/mcmaster-kiruba-ludecomp.pdf
   start_time = time.time()
   
   l,u = l_u_decomposition(A)
   
   d = np.zeros((len(A), 1), dtype=float)
   x = np.zeros((len(A), 1), dtype=float)
   
   for i in range(len(A)):
     d[i][0] = b[i][0] - sum(l[i][j] * d[j][0] for j in range(i))
     
   for i in range(len(A)-1, -1, -1):
      x[i][0] = (d[i][0] - sum(u[i][j] * x[j][0] for j in range(i+1, len(A)))) / u[i][i]
   
   norm = np.linalg.norm(np.dot(A, x) - b)
   
   elapsed_time = time.time() - start_time
   
   does_converge = norm < precision
   
   return OutputSolution(x, 0,[norm],does_converge, elapsed_time)
 
 