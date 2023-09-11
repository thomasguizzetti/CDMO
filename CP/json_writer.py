import os
import json
from collections import defaultdict
import re

def split_list(lst, delimiter):
    sublists = []
    sublist = []
    
    for item in lst:
        if item == delimiter:
            if sublist:
                sublists.append(sublist)
            sublist = []
        else:
            sublist.append(item)
    
    if sublist:
        sublists.append(sublist)
        
    return sublists

def parse_data(data_list):

    result = {}
    
    for line in data_list:
        if "maximum distance among couriers" in line:
            result['obj'] = int(re.findall(r"\d+", line)[0])
        elif "Time taken:" in line:
            result['time'] = int(float(re.findall(r"\d+\.\d+|\d+", line)[0])) if int(float(re.findall(r"\d+\.\d+|\d+", line)[0])) < 300 else 300
        elif "courier_assignment =" in line:
            result['courier_assignment'] = list(map(int, re.findall(r"\d+", line)))
        elif "successor =" in line:
            result['successor'] = list(map(int, re.findall(r"\d+", line)))

    try:
        result['optimal'] = result.get('time', float('inf')) < 300
    except:
        return None

    try:
        result['sol'] = find_routes(result['courier_assignment'], result['successor'])
    except:
        return None

    result.pop('courier_assignment', None)
    result.pop('successor', None)

    return result

def find_routes(courier_assignment, successor):
    # Number of couriers.
    num_couriers = max(courier_assignment)
    
    # Extract start and end nodes for each courier.
    start_nodes = successor[-num_couriers*2:][:-num_couriers]
    end_nodes = successor[-num_couriers*2:][-num_couriers:]

    # Remove the last 'number of couriers' * 2 entries.
    courier_assignment = courier_assignment[:-num_couriers*2]
    successor = successor[:-num_couriers*2]

    # Associate each node with its successor and its assigned courier.
    node_info = {node: {"courier": courier, "successor": succ} 
                for node, (courier, succ) in enumerate(zip(courier_assignment, successor), 1)}

    # Create routes for each courier using successors.
    courier_routes = {courier: [] for courier in range(1, num_couriers + 1)}
    for courier in range(1, num_couriers + 1):
        current_node = start_nodes[courier - 1]
        while True:
            try:
                successor[current_node-1]
                #print(f"courier: {courier}, node: {current_node}")
                courier_routes[courier].append(current_node)
                next_node = node_info[current_node]["successor"]
                # Appending the successor, which is the next_node
                current_node = next_node
            except:
                break

    # Convert dict to list, keeping the routes in courier order.
    sorted_routes = [courier_routes[courier] for courier in range(1, num_couriers + 1)]
    
    return sorted_routes


def extract_info(lines, solver_model_name):
    data = {
        "time": 300,
        "optimal": False,
        "obj": "n/a",
        "sol": "n/a"
    }
    
    for line in lines:
        if "maximum distance among couriers" in line:
            data["obj"] = int(line.split("=")[1].strip()[:-1])
        elif "courier_assignment" in line:
            line = line.split("=")[1].strip()[1:-1]
            assignment = list(map(int, line.replace(']', '').replace('[', '').split(", ")))
            
            # Transform assignment into a list of lists
            couriers = defaultdict(list)
            for idx, courier in enumerate(assignment):
                couriers[courier].append(idx + 1)
            
            data["sol"] = list(couriers.values())

    return {solver_model_name: data}


def main():
    txt_files_folder = './solutions_txt/'  # Replace this with the correct path
    output_folder = './res/CP'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    aggregated_data = {"inst01": {}, "inst02": {}, "inst03": {}, "inst04": {}, "inst05": {} ,"inst06": {}, "inst07": {}, "inst08": {}, "inst09": {}, "inst10": {}, \
                       "inst11": {}, "inst12": {}, "inst13": {}, "inst14": {}, "inst15": {}, "inst16": {}, "inst17": {}, "inst18": {}, "inst19": {}, "inst20": {}, "inst21": {}}
    
    try:
        for filename in os.listdir(txt_files_folder):
            #print(f"Processing file: {filename}")  # Debugging statement
            if filename.endswith(".txt"):
                with open(os.path.join(txt_files_folder, filename), 'r') as f:
                    lines = f.readlines()
                
                solver = lines[0].split(":")[1].strip()
                model = lines[1].split(":")[1].strip()

                solver_model_name = f"{model}_{solver}"

                instance_blocks = split_list(lines, '\n')[1:]
                #print(instance_blocks[1])
                #instance_blocks = "\n".join(lines).split("\n")[3]
                #print(instance_blocks)
                
                for block in instance_blocks:
                    instance_name = block[0].split(": ")[1][:6]
                    parsed_data = parse_data(block)
                    if parsed_data is not None:
                        aggregated_data[instance_name][solver_model_name] = parsed_data
                    #instance_name = block_lines[0].split(": ")[1][4:]
                    
                    
                    #print(f"Processing instance: {instance_name}")  # Debugging statement
                    #instance_data = extract_info(block_lines[1:], solver_model_name)
                    
                    #if instance_name in aggregated_data:
                    #    aggregated_data[instance_name].update(instance_data)
                    #else:
                    #    aggregated_data[instance_name] = instance_data
        
        print(aggregated_data)
        for instance_name, instance_data in aggregated_data.items():
            json_filename = os.path.join(output_folder, f"{instance_name}.json")
            with open(json_filename, 'w') as json_file:
                json.dump(instance_data, json_file)
    except Exception as e:
        print("An error occurred:", e)
        
if __name__ == "__main__":
    main()
