import numpy as np


# This function creates a band matrix with the specified diagonal, upper, and lower values.
# Returns a 2D numpy array representing the band matrix.
def create_band_matrix(diagonal, second, third, N):
  band_matrix = np.zeros((N, N), dtype=int)

  for i in range(0, N):
    for j in range(0, N):
      if (i == j):
        band_matrix[i][j] = diagonal
      elif abs(i-j) == 1:
        band_matrix[i][j] = second
      elif abs(i-j) ==2:
        band_matrix[i][j] = third
      
  return band_matrix


# This function creates a forcing matrix with the specified value, value equals sin(n-th element * (third_index_num  + 1)).
# Returns a 2D numpy array representing the forcing matrix.      
def create_forcing_matrix(N, value):
  forcing_matrix = np.zeros((N,1), dtype=float)
  
  for i in range(0, N):
    value = np.sin((i + 1) * value)
    forcing_matrix[i][0] = value
    
  return forcing_matrix 