There are two files in this directory:

1. `gauss_seidel_solver.py`: This file contains the implementation of the differentiable Gauss-Seidel solver, which is capable of handling various types of constraints such as distance-based, dihedral-based, and center of mass (COM) constraints.

2. `configs_model_type.py`: This file includes configuration settings for the evaluation of Protenix model.

--- 
# Differentiable Gauss-Seidel Solver

This repository provides a differentiable implementation of the Gauss-Seidel solver designed for handling various types of constraints, including:
- **Distance-based constraints** (e.g., VDW overlap, covalent bond lengths, posebusters)
- **Dihedral-based constraints** (e.g., chiral atoms, stereo bonds, planar bonds)
- **Center of Mass (COM) constraints** for chain alignment.

The solver uses the Gauss-Seidel method with a differentiable framework to solve these constraints, making it suitable for optimization tasks where gradients are required.

# Usage

## Function: diff_gauss_seidel_solve_all

This function solves the constraint system by updating the atomic coordinates and minimizing energy associated with constraints.

## Input:

- **coords**: Initial atomic coordinates with shape (N_atoms, 3).
- **dist_index**: A tensor with shape (2, N_constraints) representing the atom pairs involved in distance constraints.
- **dist_k_vals**: Stiffness values for the distance constraints.
- **dist_lower_bounds**: Lower bounds for the distance constraints.
- **dist_upper_bounds**: Upper bounds for the distance constraints.
- **dihed_index**: A tensor with shape (4, N_constraints) representing the four atoms involved in dihedral constraints.
- **dihed_k_vals**: Stiffness values for dihedral constraints.
- **dihed_lower_bounds**: Lower bounds for the dihedral constraints.
- **dihed_upper_bounds**: Upper bounds for the dihedral constraints.
- **dihed_use_abs**: Whether to use absolute values for dihedral constraints.
- **com_chain_index**: A tensor representing the chain indices for COM constraints.
- **com_args**: A tuple containing additional arguments related to the COM constraints, specifically
- **com_k_vals**: Stiffness values for COM constraints.
- **com_lower_bounds**: Lower bounds for the COM constraints.
- **com_upper_bounds**: Upper bounds for the COM constraints.
- **guidance_weight**: Weight for guidance in the constraint system.
- **parameters**: Parameters for additional coefficients or parameters relevant to the constraint system.
- **args**: Additional arguments for customization or specific optimization purposes.
- **alpha**: Regularization factor for stability.
- **n_iterations**: Number of iterations to run for the Gauss-Seidel solver.
- **debug**: Whether to enable debugging output.

## Output:

- **updated_coords**: Updated atomic coordinates after solving the constraints.
- **dist_lagrangian**: Lagrange multipliers for distance constraints.
- **dihed_lagrangian**: Lagrange multipliers for dihedral constraints.
- **com_lagrangian**: Lagrange multipliers for COM constraints.
