# CDMO
Combinatorial Optimization - Vehicle Routing Problem


In order to run the MIP and SMT solvers in the you should enter their respective folders and write a command like the following:

    python3 SMT.py 21 

or

    python3 MIP.py 01 


Warning: The number provided must be a two digit integer


The Notebooke files for the project are also included in the Notebooks folder which can be run using Google Colab or Jupyter Notebook. The needed license for using Gurobi has been also included but only in the zip file.

For CP, head to the CP folder and:

1. To run run_one_instance you need to input solver, model name, filename.txt and instance. For example:  ./run_one_instance.sh gecode CP_A CP_MODEL_B_gecode.txt inst01

2. To run all instances on all models, simply run ./run_instances_and_models_gecode.sh (for gecode) and ./run_instances_and_models_chuffed.sh (for chuffed)

Once all solutions are printed, to get them in JSON form one must json_writer.py, making sure that the solutions that we want in json format are in a solutions_txt/ folder. 


