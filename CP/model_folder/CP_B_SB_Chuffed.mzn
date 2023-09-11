include "globals.mzn";

% Parameters
int: m; % couriers
int: n; % items
array[1..m] of int: max_load;
array[1..n] of int: size;
array[1..n+1, 1..n+1] of int: distance;

% More Parameters

set of int: COURIERS = 1..m;
set of int: ITEM_LOCATIONS = 1..n;
set of int: LOAD = 0..max(max_load);
set of int: NODES = 1..n+2*m;
set of int: ORIGIN_NODES = n+1..n+2*m;
set of int: ORIGIN_NODES_START = n+1..n+m;
set of int: ORIGIN_NODES_END = n+m+1..n+2*m;

array[NODES] of int: item_location_size = [  % Size of each nodes is the corresponding item size
  if i <= n then 
    size[i]
  else 
    0
  endif
| i in NODES]; 

array[NODES, NODES] of int: distance_nodes = array2d(NODES,NODES,[           
  if i<=n /\ j <= n then % from one item location to the other
    distance[i,j]
  elseif i<=n /\ j in ORIGIN_NODES then % from item location to the origin
    distance[i,n+1]
  elseif i in ORIGIN_NODES /\ j <= n then % from origin to item location
    distance[n+1, j]
  else  
    0 % from origin to origin 
  endif
   | i,j in NODES ]);  

% Decision Variables

array[NODES] of var NODES: successor; 
array[NODES] of var NODES: predecessor; % for redundant constraints
array[NODES] of var COURIERS: courier_assignment; % specifies which courier visits which item location
array[NODES] of var LOAD: intload; % specifies the intermediate load of each courier while travelling to their assigned item locations

% Constraints

% Initialization constraint
% Predecessor of start nodes are end nodes
constraint
   forall(i in (n+2..n+m)) (
     predecessor[i] = i + m -1
   );

constraint
   predecessor[n+1] = n+2*m;

% Successors of end nodes are start nodes
constraint 
   forall(i in (n+m+1..n+2*m-1)) (
     successor[i] = i-m+1 
   );
constraint
   successor[n+2*m] = n+1;

% Associate each start/end nodes with a vehicle
constraint 
   forall(i in ORIGIN_NODES_START) (
     courier_assignment[i] = i-n
   );
   
constraint 
   forall(i in ORIGIN_NODES_END) (
     courier_assignment[i] = i-n-m
   );


% Vehicle load when starting at the origin
constraint 
   forall(i in ORIGIN_NODES_START) (
     intload[i] = 0
   );
   

% Predecessor/successor constraints

constraint
   forall(i in NODES) (
      successor[predecessor[i]] = i
   );

constraint
   forall(i in NODES) (
      predecessor[successor[i]] = i
   );

% Alldiff + subtour elimination constraints
constraint 
   circuit(successor);
   
constraint
   circuit(predecessor);


% Vehicle constraints 

% Vehicle of node i is the same as the vehicle for the predecessor
constraint
   forall(i in ITEM_LOCATIONS) (
      courier_assignment[predecessor[i]] = courier_assignment[i]
   );
constraint 
   forall(i in ITEM_LOCATIONS) (
      courier_assignment[successor[i]] = courier_assignment[i]
   );

% Load constraints

constraint 
   forall(i in ITEM_LOCATIONS) (
      intload[i] + item_location_size[i] = intload[successor[i]]
   );
constraint
   forall(i in ORIGIN_NODES_START) (
      intload[i] = intload[successor[i]] 
   );

% Check that partial load is always less or equal then the capacity for each vehicle.
constraint
    forall(i in ITEM_LOCATIONS)(
        intload[i] <= max_load[courier_assignment[i]]
    );

% Check that final load is less or equal then the capacity for each vehicle.
constraint forall(i in COURIERS)(
    intload[i+n+m] <= max_load[i]
);

% Symmetry-breaking constraint

function var int: equality(var int: assignment_1, int:assignment_2) = if assignment_1 = assignment_2 then 1 else 0 endif;

constraint forall(i, j in COURIERS where i < j) (
    if max_load[i] == max_load[j] then
        lex_less(
            [equality(courier_assignment[k], i) | k in NODES where k <= n], 
            [equality(courier_assignment[k], j) | k in NODES where k <= n]
        )
    endif
);


% Objective

var int: total_distance = sum([distance_nodes[i, successor[i]] |i in NODES]);
array[COURIERS] of var int: distance_per_courier;
constraint forall(j in COURIERS)(
    distance_per_courier[j] = sum([distance_nodes[i, successor[i]] | i in NODES where courier_assignment[i] = j]));
var int : max_distance_per_courier = max(distance_per_courier);

solve :: seq_search([int_search(courier_assignment, first_fail, indomain_split),
                int_search(successor, first_fail, indomain_split),               
               int_search(intload, first_fail, indomain_min),
               relax_and_reconstruct(successor, 88),
               restart_linear(n*n)
              ])   

minimize max_distance_per_courier; % distance


% Output

output 
   [ "maximum distance among couriers = "] ++ [show(max_distance_per_courier)] ++
   [ ";\ntotal distance of all couriers = "] ++ [show(total_distance)] ++
   [ ";\ncourier_assignment = " ] ++ [ show(courier_assignment) ]++
   [ ";\nsuccessor = "] ++          [ show(successor) ] ++
   [ ";\npredecessor = "] ++ [show(predecessor)] ++
   [ ";\nintload = "] ++ [show(intload)] ++  
   [ ";\n"]
;