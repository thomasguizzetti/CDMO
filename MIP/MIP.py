import sys

if len(sys.argv[1]) != 2:
    print("Usage: python script_name.py <2-digit_number>")
    raise

arg = sys.argv[1]

# Check if the argument is a 2-digit number between 01 and 21
if len(arg) != 2 or not arg.isdigit() or not (1 <= int(arg) <= 21):
    print("Error: Please provide a 2-digit number between 01 and 21.")
    raise

# If the argument is valid, you can access it as a string
print("You provided the number:", arg)

InstanceNumber = arg






import gurobipy as gp  # import the installed package


# import shutil

# # Move the license file to the current working directory
# shutil.move('/content/gurobi.lic', './gurobi.lic')


# import os


# # Set the Gurobi license file path
# os.environ['GRB_LICENSE_FILE'] = './gurobi.lic'

def import_variables(name):
  with open(name, 'r') as f:
    # Read the first line and convert it to an integer
    first_line = int(f.readline().strip())

    # Read the second line and convert it to an integer
    second_line = int(f.readline().strip())

    # Read the next two lines and convert them to arrays
    array1 = list(map(int, f.readline().strip().split()))
    array2 = list(map(int, f.readline().strip().split()))

    # Read the rest of the lines and convert them to a two-dimensional array
    two_d_array = []
    for line in f:
        row = list(map(int, line.strip().split()))
        two_d_array.append(row)

  # number of couriers
  m = first_line

  # number of items
  n = second_line

  # maximum load size of each courier
  l = array1

  # each item's size
  s = array2

  # Distance between distribution point i
  # and distribution point j (each items destination)
  D = two_d_array

  return m, n, l, s, D


# m is the number of couriers
# n is the number of items
# l is the capacity of couriers
# s is the sizeof each item
# d is the distances between destinations
m, n, l, s, d = import_variables(f"../Instances/inst{InstanceNumber}.dat")


def get_element_info(numbers):
    element_info = {}

    for index, num in enumerate(numbers):
        if num in element_info:
            element_info[num]['indexes'].append(index)
            element_info[num]['count'] += 1
        else:
            element_info[num] = {'indexes': [index], 'count': 1}

    for i in element_info.values():
      shifted_lst = [None] + i['indexes'][:-1]
      i['before_indexes'] = dict(zip(i['indexes'], shifted_lst))

    return element_info

l_reap = get_element_info(l)

from pyomo.environ import *
import numpy as np
import pyomo
import pyomo.opt
import logging
import time

from timeit import default_timer as timer

start_time = timer()

# Create a ConcreteModel
model = ConcreteModel()

# Define the decision variable matrix
# i is the courier and j is the item
model.x = Var(range(m), range(n), within=Boolean)
model.roots = Var(range(m), range(n+1), range(n+1), within=Boolean)

# Define the MTZ variables
model.u = Var(range(m), range(n+1), within=NonNegativeIntegers)

# Define the objective function to minimize the total distance traveled by the couriers
model.max_distance = Var(within=NonNegativeIntegers)

# total distance
model.distance = Var(within=NonNegativeIntegers)

# Define the solver
solver = SolverFactory('gurobi', tee=True)
#solver = SolverFactory('glpk', tee=True)

# Define the constraints
model.constraints = ConstraintList()

#####################################
model.constraints.add(model.distance == sum(model.roots[i, j, k] * d[j][k] for i in range(m) for j in range(n+1) for k in range(n+1)))
#####################################

# 1) The sum of each row cannot be more than 1
# because we cannot go to multiple places at once
for i in range(m):
    for j in range(n+1):
        model.constraints.add(sum(model.roots[i, j, k] for k in range(n+1)) <= 1)

# 2) The sum of no column can be more than 1
# because we don't want subtours
for i in range(m):
    for k in range(n+1):
        model.constraints.add(sum(model.roots[i, j, k] for j in range(n+1)) <= 1)

# 3) Enforce the constraint that the sum of the first column of each matrix is exactly 1
# because we're obligated to come back to the point 0
for i in range(m):
    model.constraints.add(sum(model.roots[i, j, n] for j in range(n+1)) == 1)
    model.constraints.add(sum(model.roots[i, n, j] for j in range(n+1)) == 1)

########
# Add the MTZ constraint
########
for i in range(m):
    for j in range(n+1):
        for k in range(n+1):
            if j != n and j != k:
                model.constraints.add(model.u[i, j] - model.u[i, k] + n * model.roots[i, j, k] <= n - 1)

# 5) Set diagonal elements of roots to 0
for i in range(m):
    for j in range(n+1):
        model.constraints.add(model.roots[i, j, j] == 0)


# Define the capacity constraint, which ensures that each item is assigned to exactly one courier
for j in range(n):
    # The summation of the number of couriers responsible for taking one package shouldn't surpass 1
    model.constraints.add(sum(model.x[i, j] for i in range(m)) == 1)

# Define the capacity constraint, which ensures that each courier does not exceed their capacity
for i in range(m):
    model.constraints.add(
        sum(model.x[i, j] * s[j] for j in range(n)) <= l[i]
    )

# Connecting roots to x: every x which is assigned should be visited by roots
for i in range(m):
    for j in range(n):
        model.constraints.add(sum(model.roots[i, j, k] for k in range(n+1)) == model.x[i, j])
        model.constraints.add(sum(model.roots[i, k, j] for k in range(n+1)) == model.x[i, j])


###################################################################################################
##### Symmetry Breaking ###########################################################################
###################################################################################################
####model.m = RangeSet(0, m-1)
# model.n = RangeSet(0, n-1)
# def constraint_rule(model, i, j):
#   if (#model.x[i, j].is_fixed()and
#       model.x[i, j].value == 1
#       and l_reap[l[i]]['before_indexes'][i] is not None
#       ):
#       return summation(model.x[l_reap[l[i]]['before_indexes'][i], k] for k in range(0, j)) >= 1
#   else:
#       return Constraint.Skip

# model.symmetryBreak = Constraint(model.m, model.n, rule=constraint_rule)
#model.constraints.add(model.m, model.n, rule=constraint_rule)
###################################################################################################
###################################################################################################
###################################################################################################


# Define the objective function to minimize the total distance traveled by the couriers
for i in range(m):
    model.constraints.add(sum(model.roots[i, j, k] * d[j][k] for j in range(n+1) for k in range(n+1)) <= model.max_distance)

model.constraints.add(sum(model.roots[i, j, k] * d[j][k] for i in range(m) for j in range(n+1) for k in range(n+1)) == model.distance)

# Define the objective function to minimize the total distance traveled by the couriers
# model.objective = Objective(expr=model.max_distance, sense=minimize)
# model.objective = Objective(expr=model.distance, sense=minimize)
model.objective = Objective(expr=model.max_distance, sense=minimize)

### Options ###
# Set solver options
# solver.options['timelimit'] = 60  # Set a time limit of 3600 seconds
# solver.options['mipgap'] = 0.001  # Set the MIP optimality gap tolerance to 1%
# # Set solver options
# solver.options['MIPFocus'] = 2  # Focus on finding good feasible solutions
# solver.options['Heuristics'] = 0.8  # Allow more time for heuristics to find better solutions
# solver.options['Threads'] = 2  # Use multiple threads for parallel processing

solver.options['timeLimit'] = 5*60 # Time limit in seconds
# Set the time limit
#solver.options['tmlim'] = 200
# solver.options['mipgap'] = 0.0001  # Set the MIP gap tolerance
# solver.options['max_iter'] = 1000000  # Set the maximum number of iterations to 1000
# solver.options['mipfocus'] = 3  # Increase focus on finding optimal solutions
# solver.options['heuristics'] = 0  # Reduce the emphasis on heuristics
solver.options['solutionpool'] = 10  # Collect up to 10 solutions in the solution pool

# Solve the optimization problem
results = solver.solve(model, tee=True)


# Check the solver status and termination condition
# if (results.solver.termination_condition == TerminationCondition.optimal or
#         results.solver.termination_condition == TerminationCondition.locallyOptimal):

# Check that we actually computed an optimal solution, load results
if (results.solver.status != pyomo.opt.SolverStatus.ok):
    logging.warning('Check solver not ok?')
if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):
    logging.warning('Check solver optimality?')

items = [[] for _ in range(m)]
X = []

for i in range(m):
    for j in range(n):
        X.append(value(model.x[i, j]))
        if value(model.x[i, j]):
            items[i].append(j)

print("Items:", items)
print("Assignment Matrix:")
print(np.array(X).astype(int).reshape([m, n]))

courier_matrices = []
for i in range(m):
    courier_matrix = []
    print(f"Courier {i}:")
    for j in range(n+1):
        row = []
        for k in range(n+1):
            row.append(value(model.roots[i, j, k]))
        print(np.array(row).astype(int))
        courier_matrix.append(np.array(row).astype(int))
    courier_matrix = np.concatenate(courier_matrix, axis=0).reshape([n+1, n+1])
    courier_matrices.append(courier_matrix)

rows = m
cols = n+1
MTZ = np.zeros((rows, cols))
print("MTZ Matrix:")
for i in range(rows):
    for j in range(cols):
        MTZ[i, j] = model.u[i, j].value

for i in range(m):
    row = []
    for j in range(n+1):
        row.append(int(value(model.u[i, j])))
    print(row)

print("Objective function value:", value(model.max_distance))

print("The sum of distances is:", value(model.distance))
end_time = timer()

# else:
#     print("No solution found.")

#print(model.display())
print(results)
print(f"[INFO] Total execution time: {end_time-start_time:.3f} seconds")
#print("PPRINT:")
#model.pprint()


n_items_per_courier = []
# coordinates of items per courier
coordinates_per_courier = []
for i in range(m):
    coordinates_per_courier.append(dict())
    n_items_per_courier.append(0)
    for j in range(n + 1):
        for k in range(n):
            if courier_matrices[i][j][k] > 0:
                n_items_per_courier[i] += 1
                # I save the coordinates of the item in the dictionary, where the key is the starting node and the value is the ending node
                coordinates_per_courier[i][j] = k
print(n_items_per_courier)
print(coordinates_per_courier)
best_paths_items = [[] for i in range(m)]
for i in range(m):
    best_paths_items[i].append(coordinates_per_courier[i][n])
    while len(best_paths_items[i]) < n_items_per_courier[i]:
        best_paths_items[i].append(coordinates_per_courier[i][best_paths_items[i][-1]])


import json

# Create a dictionary with your values
data = {
    "time": f"{end_time-start_time}",
    "optimal": f"true",
    "obj": f"{int(model.max_distance.value)}",
    "sol": f"{best_paths_items}"
}

# Specify the file name where you want to save the JSON data
file_name = f"{InstanceNumber}.json"

# Write the data to the JSON file
with open(file_name, "w") as json_file:
    json.dump(data, json_file)

print(f"JSON data has been saved to {file_name}")