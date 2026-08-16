import numpy as np


def calculate_surprise(actual_input: np.ndarray,
                       predicted_input: np.ndarray):

    error = actual_input - predicted_input

    surprise = np.linalg.norm(error)

    return error, surprise