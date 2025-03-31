# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 20:08:20 2024

@author: hendro
"""

import numpy as np

def generate_neighbor(solution, lb, ub):
    """
    Generates a neighboring solution by making small perturbations to the current solution.
    Ensures that the new solution stays within the bounds.
    """
    perturbation = np.random.uniform(-0.1, 0.1, len(solution))  # Adjust perturbation range if needed
    neighbor = np.clip(solution + perturbation, lb, ub)
    return neighbor
