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
    # Initialize an empty dictionary to store the routes for each courier.
    courier_routes = {}
    slice_number = max(courier_assignment) * 2
    courier_assignment = courier_assignment[:-slice_number]
    successor = successor[:-slice_number]

    # Associate each node with its successor and its assigned courier.
    node_info = {}
    for i, (courier, succ) in enumerate(zip(courier_assignment, successor)):
        
        node = i + 1
        if courier not in courier_routes:
            courier_routes[courier] = []
        node_info[node] = {"courier": courier, "successor": succ}
    
    # Sort nodes by successor order within each courier's list.
    for node, info in node_info.items():
        courier = info["courier"]
        route = courier_routes[courier]
        route.append((node, info["successor"]))
        
    # Sort and clean up each courier's route.
    for courier, route in courier_routes.items():
        route.sort(key=lambda x: x[1])  # Sort by successor.
        courier_routes[courier] = [node for node, _ in route]
    
    # Initialize a list with None for placeholder.
    max_courier = max(courier_routes.keys())
    sorted_routes = [None] * max_courier
    
    # Populate the list using courier numbers as indices.
    for courier, route in courier_routes.items():
        sorted_routes[courier - 1] = route
    
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
                json.dump(instance_data, json_file, indent=4)
    except Exception as e:
        print("An error occurred:", e)
        
if __name__ == "__main__":
    main()
