README

1. To run run_one_instance you need to input solver, model name, filename.txt and instance. For example:  ./run_one_instance.sh gecode CP_A CP_MODEL_B_gecode.txt inst01

2. To run all instances on all models, simply run ./run_instances_and_models_gecode.sh (for gecode) and ./run_instances_and_models_chuffed.sh (for chuffed)

Once all solutions are printed, to get them in JSON form one must json_writer.py, making sure that the solutions that we want in json format are in a solutions_txt/ folder. 

