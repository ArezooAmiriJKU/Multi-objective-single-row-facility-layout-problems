# Multi-Objective Single Row Facility Layout Problem (SRFLP)

This repository contains the data, code, and output tables related to the multi-objective single row facility layout problem (SRFLP) explored in the corresponding thesis.
The project focuses on finding all non-dominated solutions and lower bounds for the problem using **exact approaches**.

## Overview

Facility layout problems are critical in production and service systems to improve efficiency and reduce costs. The single row configuration is an important NP-hard variant,
where departments are arranged in a linear fashion. Multi-objective optimization is essential, as practical problems often involve conflicting objectives such as flow cost,
rearrangement cost, and closeness ratings.  

In this project, two exact approaches are implemented:

1. **Weighted-Sum Approach** – Combines multiple objectives into a single objective using weighting factors to generate supported Pareto-optimal points.  
2. **Epsilon-Constraint Approach** – Optimizes one objective while converting others into constraints with varying thresholds to generate the full Pareto frontier,
    including non-supported non-dominated points.

Both approaches are implemented in Python and solved using the Gurobi solver.

## Repository Contents

- `AmaralWeightedSum.py` – Python code implementing the weighted-sum approach.  
- `AmaralEpsilonConstraint.py` – Python code implementing the epsilon-constraint approach.  
- `data/` – Input instance files for benchmart instances and random generated instances with different numbers of departments, densities, flow ranges, and lengths.
- Files for second objectives are named by adding `_1` to the end of fisrt objective files.
- Includes the two sets used for the sensitivity analysis. 
- `output_tables/` – LaTeX tables generated from the computations. These include detailed results for all instances; due to the large number of tables (108 in total),
   they are provided here for readers interested in the full dataset.  

## Sensitivity Analysis

The repository also contains the input data  for the **sensitivity analysis**, including 12-department instances with different densities, flow ranges, and layout lengths.
This allows for a detailed study of how problem parameters affect computation time and solution distribution for the epsilon-constraint approach.

## Usage

1. Ensure Gurobi is installed and licensed on your system.  
2. Run either `AmaralWeightedSum.py` or `AmaralEpsilonConstraint.py` with the appropriate input data file.
3. You need `output`, `outputE`, and `results` directory to store the output of weightes-sum (latex,PNG,CSV), epsilon-costraint and overall table of results
  respectively.
5. You need `Instance` directory to store appropriate input.

## Notes

- The repository complements the thesis, where only representative tables and figures are included.  
- Full computational results and instance data are provided here for transparency, reproducibility, and further research.  

## Keywords

Single Row Facility Layout, Multi-Objective Optimization, Exact Methods
