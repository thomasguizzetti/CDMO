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

print(l)
# This is used in SB below:
l_reap = get_element_info(l)
l_reap


import numpy as np
from timeit import default_timer as timer
from z3 import *


m, n, l, s, d = import_variables(f"../Instances/inst{InstanceNumber}.dat")

start_time = timer()
# Define a function to handle the on_model event
def on_model(model):
    items = [[] for i in range(m)]
    X = []
    for i in range(m):
        for j in range(n):
            X.append(is_true(model.eval(x[i][j])))
            if is_true(model.eval(x[i][j])):
                items[i].append(j)
    print("Items assigned to couriers:")
    print(items)
    print("Assignment Matrix:")
    print(np.array(X).astype(int).reshape([m, n]))
    global courier_matrices
    courier_matrices = []
    for i in range(m):
        courier_matrix = []
        print(f"Courier {i}:")
        for j in range(n+1):
            row = []
            for k in range(n+1):
                row.append(is_true(model[roots[i][j][k]]))
            print(np.array(row).astype(int))
            courier_matrix.append(np.array(row).astype(int))
        courier_matrix = np.concatenate(courier_matrix, axis=0).reshape([n+1, n+1])
        courier_matrices.append(courier_matrix)

    global model_max_distance
    model_max_distance = model.evaluate(max_distance)
    print("Objective function value: ", model.evaluate(max_distance))

    distance = 0
    for i in range(m):
        distance += np.sum(courier_matrices[i] * d)
    print(f"The sum of distances is:    {distance}")
    print("##########################################")


# Set the on_model handler
# solver.set("on_model", on_model)

# Define the decision variable matrix
# i is the courier and j is the item
x = [[Bool('x[%i,%i]' % (i,j)) for j in range(n)] for i in range(m)]
roots = [ [ [ Bool("courier%i[%i,%i]" % (i, j, k)) for k in range(n+1) ] for j in range(n+1) ] for i in range(m) ]

# Define the solver
solver = Optimize()
# Set the function that needs to be called when a solution is found (on_model prints the intermediate solutions) to the solver
solver.set_on_model(on_model) 

###################################################################################################
#### Options ######################################################################################
###################################################################################################

# Set the rlimit parameter to a large value
#solver.set("rlimit", 10**9)  # Example: 1 billion steps

# Set the timeout to 5 minutes (300,000 milliseconds)
solver.set("timeout", 5*60*1000)
# Sets the logic option for the solver
solver.set("maxsat_engine",'core_maxsat')

###################################################################################################
###################################################################################################
###################################################################################################


###################################################################################################
#### The assignment Matrix Rules ##################################################################
###################################################################################################
# Define the assignement constraint, which ensures that each item is assigned to exactly one courier
for j in range(n):
    # the summation of the number of couriers responsible for taking one package shouldn't surpass 1
    solver.add(Sum([If(x[i][j], 1, 0) for i in range(m)]) == 1)

# Define the capacity constraint, which ensures that each courier does not exceed their capacity
for i in range(m):
    solver.add(Sum([If(x[i][j], s[j], 0) for j in range(n)]) <= l[i])
    #solver.add(Sum([If(x[i][j], s[j], 0) for j in range(n)]) <= IntVal(l[i]))
###################################################################################################
###################################################################################################
###################################################################################################



###################################################################################################
#### Courier Matrices' Rules ######################################################################
###################################################################################################

# 1) The sum of each row cannot be more than 1
# because we cannot go to multiple places at once
for i in range(m):
    for j in range(n+1):
        solver.add(Sum([roots[i][j][k] for k in range(n+1)]) <= 1)

# 2) The sum of no column can be more than 1
# because we don't want to go to the same place twice
for i in range(m):
    for k in range(n+1):
        solver.add(Sum([roots[i][j][k] for j in range(n+1)]) <= 1)

########  CHANGED
# 3) enforce the constraint that the sum of the last column of each matrix is exactly 1
# because we're obligated to come back to the point 0
# 3. Every vehicle leaves the depot
for i in range(m):
  solver.add(Sum([roots[i][j][n] for j in range(n+1)]) == 1)
  solver.add(Sum([roots[i][n][j] for j in range(n+1)]) == 1)

#4) No subtours
#Add the MTZ constraint
u = [[Int(f"u[{i},{j}]") for j in range(n+1)] for i in range(m)]

# All the values in U should be greater or equal to zero. 
for i in range(m):
    for j in range(n+1):
        for k in range(n+1):
          solver.add(u[i][j] >= 0)

for i in range(m):
    for j in range(n+1):
        for k in range(n+1):
            if j != n and j != k:
              solver.add(u[i][j] - u[i][k] + n * roots[i][j][k] <= n - 1)

# # 5)
# All the diagonal elements in the routes matrix are zero. 
for i in range(m):
   for j in range(n+1):
      solver.add(roots[i][j][j] == BoolVal(0))


#### Implied Constraints ###############################################################################
#6)
# 1. Vehicle leaves node that it enters
for i in range(m):
    for j in range(n):
        sum_expr = Sum([roots[i][j][k] for k in range(n+1)])
        solver.add(Implies(sum_expr == 1, Sum(roots[i][j]) == 1))

###################################################################################################
###################################################################################################
###################################################################################################

###################################################################################################
#### Connection Rules #############################################################################
###################################################################################################

#???
# connecting roots to x
# every x which is assigned should be visited by roots
# one edge coming and one going for the ones which are assigned
# 3. Every vehicle leaves the depot
for i in range(m):
    for j in range(n):
        solver.add(Sum([roots[i][j][k] for k in range(n+1)]) == x[i][j])
        # redundant constraint
        solver.add(Sum([roots[i][k][j] for k in range(n+1)]) == x[i][j])


###################################################################################################
#### Symmetry Breaking ############################################################################
###################################################################################################

# Lexographical ordering of the couriers with the same load capacity
for i in range(m):
    for j in range(n):
        if (l_reap[l[i]]['before_indexes'][i] is not None):
            before_indexes_i = l_reap[l[i]]['before_indexes'][i]
            if (x[i][j] == True):
                summation_expr = Sum([If(x[before_indexes_i][k], 1, 0) for k in range(j)])
                solver.add(summation_expr >= 1)
###################################################################################################
#### Solver Objective and configurations ##########################################################
###################################################################################################

# Define the objective function to minimize the total distance traveled by the couriers
max_distance = Int("max_distance")
for i in range(m):
  solver.add(Sum([roots[i][j][k] * d[j][k] for j in range(n+1) for k in range(n+1)]) <= max_distance)

solver.minimize(max_distance)


distance = Int("distance")
solver.add(distance == Sum([roots[i][j][k] * d[j][k] for j in range(n+1) for k in range(n+1) for j in range(m)]))

# # Minimize distance if max_distance is already optimized
#solver.minimize(distance)

# Solve the optimization problem
if solver.check() == sat:
    print("Solution(s) found.")
else:
    print("No solution found.")

end_time = timer()

# # Solve the optimization problem
# #while True:
# if solver.check() == sat:
#   model = solver.model()
#   items = [[] for i in range(m)]
#   X = []
#   for i in range(m):
#       for j in range(n):
#         X.append(is_true(model.eval(x[i][j])))
#         if is_true(model.eval(x[i][j])):
#           items[i].append(j)
#   print(items)
#   print("Assignment Matrix:")
#   print(np.array(X).astype(int).reshape([m,n]))
#   courier_matrices = []
#   for i in range(m):
#       courier_matrix = []
#       print(f"Courier {i}:")
#       for j in range(n+1):
#         row = []
#         for k in range(n+1):
#             row.append(is_true(model[roots[i][j][k]]))
#         print(np.array(row).astype(int))
#         courier_matrix.append(np.array(row).astype(int))
#       courier_matrix = np.concatenate(courier_matrix, axis=0).reshape([n+1, n+1])
#       courier_matrices.append(courier_matrix)

#   ################################################################
#   ######## MTZ
#   print("####################################################")
#   print("MTZ Matrix")
#   for i in range(m):
#     row = []
#     for j in range(n+1):
#         row.append(model.evaluate(u[i][j]))
#     print(row)
#   print("####################################################")
#   ################################################################

#   print("Objective function value: ", model.evaluate(max_distance))

#   distance = 0
#   for i in range(m):
#     distance += np.sum(courier_matrices[i]*d)
#   print(f"The sum of distances is:    {distance}")

#       #break
# else:
#    print("No solution found.")

# How we print the Output. 

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
                
best_paths_items = [[] for i in range(m)]
for i in range(m):
    best_paths_items[i].append(coordinates_per_courier[i][n])
    while len(best_paths_items[i]) < n_items_per_courier[i]:
        best_paths_items[i].append(coordinates_per_courier[i][best_paths_items[i][-1]])
print(best_paths_items)


import json

# Create a dictionary with your values
data = {
    "time": f"{end_time-start_time}",
    "optimal": f"true",
    "obj": f"{model_max_distance}",
    "sol": f"{best_paths_items}"
}

# Specify the file name where you want to save the JSON data
file_name = f"{InstanceNumber}.json"

# Write the data to the JSON file
with open(file_name, "w") as json_file:
    json.dump(data, json_file)

print(f"JSON data has been saved to {file_name}")