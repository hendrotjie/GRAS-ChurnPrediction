import numpy as np
import time
from GSA_implementation import GSA  # Ensure GSA is defined in GSA_implementation.py
from generate_neighbor import generate_neighbor  # Import neighboring solution generator
from utils import fitness_function  # Import the custom fitness function for evaluating solutions

def gsa_sa_iterative(X_train, y_train, max_iters=100, gsa_iters=10, sa_iters=50, alpha=0.93, convergence_threshold=0.01, clf_name="KNN"):
    """
    Adaptive GSA-SA function: runs GSA at each SA iteration to find a starting solution, 
    then refines it using Simulated Annealing.
    """
    lb = [0] * X_train.shape[1]
    ub = [1] * X_train.shape[1]
    dim = len(lb)
    PopSize = 20  # Adjust based on computational power

    # Initialize best solution tracking variables
    best_solution = None
    best_fitness = float('inf')  # Assume we are minimizing fitness (e.g., error rate)
    results = []  # List to store each iteration's results

    # Run SA with GSA initialization at each SA iteration
    for sa_iter in range(sa_iters):
        print(f"Starting SA Iteration {sa_iter + 1}")

        # Run GSA to get an initial solution for this SA iteration
        gsa_solution = GSA(
            objf=lambda solution, _: fitness_function(solution, X_train, y_train, clf=clf_name),  # Pass X_train and y_train using lambda  # Add _ as a placeholder for df
            lb=lb, 
            ub=ub, 
            dim=dim, 
            PopSize=PopSize, 
            iters=gsa_iters, 
            df=X_train
        )
        current_solution = gsa_solution.gBest
        current_fitness = gsa_solution.best_fitness

        # Initialize SA variables
        temperature = 2 * len(current_solution)  # Initial temperature based on feature count
        cooling_schedule = alpha

        # SA refinement loop
        for sa_inner_iter in range(max_iters):
            start_time = time.time()

            # Generate a neighboring solution
            neighbor_solution = generate_neighbor(current_solution, lb, ub)
            neighbor_fitness = fitness_function(neighbor_solution, X_train, y_train, clf = clf_name)

            # Acceptance criteria
            if neighbor_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - neighbor_fitness) / temperature):
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                print(f"Accepted new solution with fitness {current_fitness}")

            # Update best solution if improved
            if current_fitness < best_fitness:
                best_solution = current_solution
                best_fitness = current_fitness

            # Select features based on current solution
            selected_features = [i for i, value in enumerate(current_solution) if value > 0.5]
            feature_count = len(selected_features)

            # Cool down
            temperature *= cooling_schedule
            end_time = time.time()
            iteration_time = end_time - start_time

            # Save current iteration results
            results.append({
                'SA Iteration': sa_iter + 1,
                'SA Inner Iteration': sa_inner_iter + 1,
                'Fitness': current_fitness,
                'Best Fitness': best_fitness,
                'Run Time (s)': iteration_time,
                'Feature Set': selected_features,  # Subset fitur
                'Feature Count': feature_count
            })

            # Convergence check for early stopping
            # if abs(best_fitness - current_fitness) < convergence_threshold:
            #     print(f"No significant improvement. Stopping SA Inner Iteration {sa_inner_iter + 1}.")
            #     break

    print(f"Best solution found: {best_solution} with fitness: {best_fitness}")
    return best_solution, results
