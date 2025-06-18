# import os
# import pickle
# import neat
# import numpy as np
# from Simulation.projections_util import Objective, create_neat_proj, create_paper_objectives
# from Data.data_load_util import make_dataset
# from tqdm import tqdm

#testing
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# Minimize:    c^T x = x0 + x1
# Subject to:  2*x0 + x1 >= 10
#              x0 + 3*x1 >= 15
#              x0, x1 >= 0 and integer

# Objective coefficients
c = np.array([1, 1])

# Variable bounds (non-negative)
bounds = Bounds([0, 0], [np.inf, np.inf])

# Constraints matrix and bounds
A = np.array([
    [2, 1],
    [1, 3]
])
lb = np.array([10, 15])
ub = np.array([np.inf, np.inf])
constraints = LinearConstraint(A, lb, ub)

# Integer constraint
integrality = np.array([1, 1])  # 1 = integer, 0 = continuous

# Solve MILP
res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

# Output result
print("Status:", res.message)
print("Objective value:", res.fun)
print("x =", res.x)
