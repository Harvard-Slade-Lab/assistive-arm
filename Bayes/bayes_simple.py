import numpy as np
from bayes_opt import BayesianOptimization
import bayes_opt
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
import matplotlib.pyplot as plt

# Track scores for visualization
score_history = []

# Simplified objective function
def objective(force1_end_time, force1_peak_force, force2_start_time, force2_peak_time):
    if force2_peak_time <= force2_start_time:
        return -1

    # Smooth function with a clear peak
    target = (
        -((force1_end_time - 0.5) ** 2)
        - ((force1_peak_force - 0.8) ** 2)
        - ((force2_start_time - 0.7) ** 2)
        - ((force2_peak_time - 0.9) ** 2)
    )
    score = target  # Simulated scoring metric
    score_history.append(score)

    return score

# Parameter bounds
pbounds = {
    "force1_end_time": (0.0, 1.0),
    "force1_peak_force": (0.0, 1.0),
    "force2_start_time": (0.0, 1.0),
    "force2_peak_time": (0.0, 1.0),
}

def load_optimizer(kappa):
    acquisition = bayes_opt.acquisition.UpperConfidenceBound(kappa=kappa)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,                   # Parameter bounds
        acquisition_function=acquisition,  # Acquisition function: Upper Confidence Bound
        random_state=0,                    # Random seed accepts integer value and is used for reproducibility.
        verbose=2                          # Verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    return optimizer

# Plot score history
def plot_scores():
    plt.figure(figsize=(10, 6))
    plt.plot(score_history, label="Score History")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Optimization Score Trend")
    plt.legend()
    plt.show()

# Main function
def main():
    optimizer = load_optimizer(kappa=2.5)

    optimizer.maximize(init_points=10, n_iter=100)  # 5 random points, 25 optimization steps

    # Output the best parameters and score
    print("Best parameters found:", optimizer.max["params"])
    print("Best score:", optimizer.max["target"])

    # Plot score trend
    plot_scores()

if __name__ == "__main__":
    main()
