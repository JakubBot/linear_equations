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
      
  

def solve_jacobi(A: np.ndarray, b: np.ndarray, precision: float = 1e-9, max_error: float = 1e9) -> OutputSolution:
  # algorithm source https://en.wikipedia.org/wiki/Jacobi_method
  start_time = time.time()
  
  errors = [float('Inf')]
  iterations = 0
  does_converge = True
  
  x = np.ones((len(A), 1), dtype=float)
  
  while errors[-1] > precision:
    new_x = np.copy(x)
    
    # A = L + U + D, L is lower triangular, U is upper triangular, D is diagonal
    for i in range(len(A)):
      sum_L_U = sum(A[i][j] * x[j][0] for j in range(len(A)) if i != j) # K = (L + U) * x
      new_x[i][0] = (b[i][0] - sum_L_U) / A[i][i] # (b - K) / D
      
    
    current_error = np.linalg.norm(np.dot(A, new_x) - b)
    errors.append(current_error)
    
    if current_error > max_error:
       does_converge = False
       break
     
    x = new_x
    iterations += 1
    
  elapsed_time = time.time() - start_time
  return OutputSolution(x, iterations,errors, does_converge,elapsed_time)

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
 
 