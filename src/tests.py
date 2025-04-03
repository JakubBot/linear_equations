from matrix_algorithms import solve_jacobi
import numpy as np


def test_jacobi():
   # Macierz A (układ równań)
  A = np.array([[4, -1, 0, 0],
                [-1, 4, -1, 0],
                [0, -1, 4, -1],
                [0, 0, -1, 3]], dtype=float)
  
  # Wektor b
  b = np.array([15, 10, 10, 10], dtype=float)
  
  # Wywołanie funkcji solve_jacobi
  x, iterations, errors, does_converge = solve_jacobi(A, b)
  
  # Wyniki
  print("Solution:", x)
  print("Iterations:", iterations)
  print("Errors:", errors)
  print("Does converge:", does_converge)
  
def test_jacobi2():
  A = np.array([[10, -1, 2],
                [-1, 10, -1],
                [2, -1, 10]], dtype=float)

  # Wektor b
  b = np.array([6, 5, -2], dtype=float)

  # Wywołanie funkcji solve_jacobi
  x, iterations, errors, does_converge = solve_jacobi(A, b)

  # Wyniki
  print("Solution:", x)
  print("Iterations:", iterations)
  print("Errors:", errors)
  print("Does converge:", does_converge)  

  
test_jacobi()
test_jacobi2()