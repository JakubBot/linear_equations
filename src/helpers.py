import numpy as np
import matplotlib.pyplot as plt
from app_types import Config


# This function creates a band matrix with the specified diagonal, upper, and lower values.
# Returns a 2D numpy array representing the band matrix.
def create_band_matrix(diagonal: int, second: int, third: int, N: int) -> np.ndarray:
    band_matrix = np.zeros((N, N), dtype=int)

    for i in range(0, N):
        for j in range(0, N):
            if i == j:
                band_matrix[i][j] = diagonal
            elif abs(i - j) == 1:
                band_matrix[i][j] = second
            elif abs(i - j) == 2:
                band_matrix[i][j] = third

    return band_matrix


# This function creates a forcing matrix with the specified value, value equals sin(n-th element * (third_index_num  + 1)).
# Returns a 2D numpy array representing the forcing matrix.
def create_forcing_matrix(N: int, value: int) -> np.ndarray:
    forcing_matrix = np.zeros((N, 1), dtype=float)

    for i in range(0, N):
        value = np.sin((i + 1) * value)
        forcing_matrix[i][0] = value

    return forcing_matrix


def create_graph(config: Config):
    x_label = config.get("x_label", None)
    y_label = config.get("y_label", None)
    title = config.get("title", None)
    savedImageName = config.get(
        "path", "plot.png"
    )  # Domyślny plik, jeśli brak w config
    log_y_axis = config.get("log_y_axis", True)  # Domyślnie False
    has_x_axis = config.get("has_x_axis", False)  # Domyślnie False
    plot = config.get("plot", [])
    axhline = config.get("axhline", [])
    fig_size = config.get("figsize", None)

    if fig_size:
        plt.figure(figsize=fig_size)

    if has_x_axis:
        for i in range(len(plot)):
            plt.plot(plot[i][0], plot[i][1], label=plot[i][2])
            plt.xticks(plot[i][0])
    else:
        for i in range(len(plot)):
            x = np.arange(1, len(plot[i][0]) + 1)
            plt.plot(x, plot[i][0], label=plot[i][1])

        max_y = max([len(plot[i][0]) for i in range(len(plot))])
        step = max_y // 6 # calculating xticks for x axis (should be 6 + 1(first) ticks)
        plt.xticks(np.arange(1, max_y + 1, step=step))

    if log_y_axis:
        plt.yscale("log")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if axhline:
        for y, label in axhline:
            plt.axhline(y=y, linestyle="--", color="red", label=label)

    plt.legend()
    plt.grid()
    plt.savefig(savedImageName)
    plt.show()


def l_u_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # algorithm https://www.geeksforgeeks.org/doolittle-algorithm-lu-decomposition/
    n = len(A)

    U = np.zeros((n, n), dtype=float)
    L = np.zeros((n, n), dtype=float)

    for i in range(n):

        # creating upper triangular
        for k in range(i, n):
            U[i][k] = A[i][k] - sum(L[i][j] * U[j][k] for j in range(i))

        # creating lower triangular
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                L[k][i] = (A[k][i] - sum(L[k][j] * U[j][i] for j in range(i))) / U[i][i]

    return L, U


def l_u_decomposition_optimized(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # algorithm https://www.geeksforgeeks.org/doolittle-algorithm-lu-decomposition/
    n = len(A)

    U = np.zeros((n, n), dtype=float)
    L = np.eye(n, dtype=float)  # insert ones on diagonal by default

    for i in range(n):
        # creating upper triangular
        U[i, i:] = A[i, i:] - L[i, :i].dot(U[:i, i:])

        # creating lower triangular
        if i + 1 < n:
            L[i + 1 :, i] = (A[i + 1 :, i] - L[i + 1 :, :i] @ U[:i, i]) / U[i, i]

    return L, U
