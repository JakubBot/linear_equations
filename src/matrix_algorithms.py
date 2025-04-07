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
 
 
def solve_jacobi_sparse_no_lib(A, b, precision=1e-9, max_error=1e9):
    """
    Rozwiązanie układu równań metodą Jacobiego, w którym obliczenia wykonywane są 
    wyłącznie na niezerowych elementach macierzy A, bez użycia bibliotek do macierzy rzadkich.
    
    Parametry:
      A         - macierz współczynników (lista list, gdzie A[i][j] to współczynnik)
      b         - wektor wyrazów wolnych (lista)
      precision - zadana precyzja rozwiązania
      max_error - maksymalny dopuszczalny błąd, po którego przekroczeniu uznajemy, 
                  że metoda nie zbiega

    Zwraca:
      Obiekt OutputSolution zawierający rozwiązanie, liczbę iteracji, historię błędów,
      informację o zbieżności oraz czas wykonania.
    """
    start_time = time.time()
    n = len(A)
    # Inicjalizacja rozwiązania
    x = [1.0 for _ in range(n)]
    
    # Preprocesing macierzy:
    # D - elementy przekątne,
    # off_diag - dla każdego wiersza lista par (indeks, wartość) dla niezerowych elementów poza przekątną
    D = [A[i][i] for i in range(n)]
    off_diag = []
    for i in range(n):
        row_nonzero = []
        for j in range(n):
            if i != j and A[i][j] != 0:
                row_nonzero.append((j, A[i][j]))
        off_diag.append(row_nonzero)
    
    errors = [float('inf')]
    iterations = 0
    does_converge = True
    
    # Iteracyjna metoda Jacobiego
    while errors[-1] > precision:
        new_x = x.copy()
        # Obliczenie nowego przybliżenia dla każdego elementu x[i]
        for i in range(n):
            s = 0.0
            for j, aij in off_diag[i]:
                s += aij * x[j]
            new_x[i] = (b[i] - s) / D[i]
        
        # Obliczenie normy różnicy: ||Ax - b||
        Ax_minus_b = [0.0 for _ in range(n)]
        for i in range(n):
            s = A[i][i] * new_x[i]
            for j, aij in off_diag[i]:
                s += aij * new_x[j]
            Ax_minus_b[i] = s - b[i]
        current_error = math.sqrt(sum(val * val for val in Ax_minus_b))
        errors.append(current_error)
        
        if current_error > max_error:
            does_converge = False
            break
        
        x = new_x
        iterations += 1
    
    elapsed_time = time.time() - start_time
    return OutputSolution(x, iterations, errors, does_converge, elapsed_time)