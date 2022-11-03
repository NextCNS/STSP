import os
import itertools
import click
import math
import dimod
import pandas as pd
from dwave.system import LeapHybridCQMSampler, DWaveSampler,FixedEmbeddingComposite, EmbeddingComposite, LeapHybridSampler
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel, QuadraticModel
from pyqubo import Array, Constraint, solve_qubo, Binary
from datetime import datetime
# # P4 - C_MAX = 20
# PROFIT = [0,8,3,0]
# P5 - C_MAX = 25
# PROFIT = [0,8,3,9,0]
# p6 
PROFIT = [0,8,3,9,1,0]
N = len(PROFIT)
def parse_inputs(data_file, capacity):
    """Parse user input and files for data to build CQM.

    Args:
        data_file (csv file):
            File of connection between arcs
        capacity (int):
            Max prefefined weight.

    Returns:
        profits, weights, and capacity.
    """
    df = pd.read_csv(data_file, names=['vertex1','vertex2', 'weight'])

    if not capacity:
        capacity = int(0.8 * sum(df['weight']))
        print("\nSetting weight capacity to 80% of total: {}".format(str(capacity)))

    return df['vertex1'], df['vertex2'], df['weight'], capacity

def build_knapsack_cqm(profits, vertex1, vertex2,weight, max_weight):
    """Construct a CQM for the knapsack problem.

    Args:
        profits (array-like):
            Array of profits for the items.
        weights (array-like):
            Array of weights for the items.
        max_weight (int):
            Maximum allowable weight for the knapsack.

    Returns:
        Constrained quadratic model instance that represents the knapsack problem.
    """
    num_items = len(profits)
    print("\nBuilding a CQM for {} items.".format(str(num_items)))

    cqm = ConstrainedQuadraticModel()
    obj = BinaryQuadraticModel(vartype='BINARY')
    constraint1 = QuadraticModel()
    constraint2 = QuadraticModel()
    constraint3 = QuadraticModel()
    constraint4 = QuadraticModel()
    constraint5 = QuadraticModel()

    #Maximize the total profit
    for i in range(1, num_items-1):
        for j in range(1, num_items):
            if j != i:
                # Objective is to maximize the total profit
                obj.add_variable("x[{}][{}]".format(i,j))
                obj.set_linear("x[{}][{}]".format(i,j), -profits[i])
    cqm.set_objective(obj)
    #constraint to start from node 1          
    for j in range(1, num_items):            
        constraint1.add_variable('BINARY', "x[{}][{}]".format(0,j))
        constraint1.set_linear("x[{}][{}]".format(0,j), 1)
    
    cqm.add_constraint(constraint1, sense="==", rhs=1, weight=None, label='c1')
    #constraint to end at node n          
    for j in range(num_items-1):            
        constraint2.add_variable('BINARY', "x[{}][{}]".format(j,num_items-1))
        constraint2.set_linear("x[{}][{}]".format(j,num_items-1), 1)
    cqm.add_constraint(constraint2, sense="==", rhs=1, weight=None, label='c2')
    
    #Visit at most 1
    for k in range(1, num_items-1):
        for i in range(num_items-1):
            if i != k:            
                constraint3.add_variable('BINARY', "x[{}][{}]".format(i,k))
                constraint3.set_linear("x[{}][{}]".format(i,k), 1)
    cqm.add_constraint(constraint3, sense="<=", rhs=1, weight=None, label='c3')
    
    #leave at most 1
    for k in range(1, num_items-1):
        for j in range(1,num_items):
            if j != k:            
                constraint4.add_variable('BINARY', "x[{}][{}]".format(k,j))
                constraint4.set_linear("x[{}][{}]".format(k,j), 1)
    cqm.add_constraint(constraint4, sense="<=", rhs=1, weight=None, label='c4')

    #smaller than C_Max
    for from_v,to_v, cos in zip(vertex1, vertex2, weight):            
        constraint5.add_variable('BINARY', "x[{}][{}]".format(from_v,to_v))
        constraint5.set_linear("x[{}][{}]".format(from_v,to_v), cos)
    cqm.add_constraint(constraint5, sense="<=", rhs=20, weight=None, label='c5')
    # Check
    # print(obj.variables, obj.linear)
    # print(constraint1.variables, constraint1.linear)
    # print(constraint2.variables, constraint2.linear)
    # print(constraint3.variables, constraint3.linear)
    # print(constraint4.variables, constraint4.linear)
    # print(constraint5.variables, constraint5.linear)
    return cqm

def parse_solution(sampleset):
    """Translate the best sample returned from solver to shipped items.

    Args:

        sampleset (dimod.Sampleset):
            Samples returned from the solver.
        costs (array-like):
            Array of costs for the items.
        weights (array-like):
            Array of weights for the items.
    """
    # feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    # if not len(feasible_sampleset):
    #     raise ValueError("No feasible solution found")

    best = sampleset.first

    # selected_item_indices = [key for key, val in best.sample.items() if val==1.0]
    # selected_weights = list(weights.loc[selected_item_indices])
    # selected_costs = list(costs.loc[selected_item_indices])

    print("\nFound best solution at energy {}".format(best.energy))
    # print("\nSelected item numbers (0-indexed):", selected_item_indices)
    # print("\nSelected item weights: {}, total = {}".format(selected_weights, sum(selected_weights)))
    # print("\nSelected item costs: {}, total = {}".format(selected_costs, sum(selected_costs)))
    print("\n Feasible solution: {}".format(best.first.sample))

def datafile_help(max_files=5):
    """Provide content of input file names and total weights for click()'s --help."""

    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        datafiles = os.listdir(data_dir)
        # "\b" enables newlines in click() help text
        help = """
\b
Name of data file (under the 'data/' folder) to run on.
One of:
File Name \t Total weight
"""
        for file in datafiles[:max_files]:
            _, weights, _ = parse_inputs(os.path.join(data_dir, file), 1234)
            help += "{:<20} {:<10} \n".format(str(file), str(sum(weights)))
        help += "\nDefault is to run on data/large.csv."
    except:
        help = """
\b
Name of data file (under the 'data/' folder) to run on.
Default is to run on data/large.csv.
"""

    return help

filename_help = datafile_help()     # Format the help string for the --filename argument

@click.command()
@click.option('--filename', type=click.File(), default='data/large.csv',
              help=filename_help)
@click.option('--capacity', default=None,
              help="Maximum weight for the container. By default sets to 80% of the total.")

def main(filename, capacity):
    """Solve a knapsack problem using a CQM solver."""

    vertex1, vertex2, weight, C_MAX = parse_inputs(filename, capacity)
    cqm = build_knapsack_cqm(PROFIT,vertex1, vertex2, weight, C_MAX)
    bqm, invert = dimod.cqm_to_bqm(cqm)

    # -------Dimod--------
    # start_time = datetime.now()
    # sampleset = dimod.ExactSolver().sample(bqm)
    # end_time =  datetime.now()
    
    # -------QPU--------
    qpu = DWaveSampler()
    sampleset_1 = EmbeddingComposite(qpu).sample(bqm,return_embedding=True,
                                             answer_mode="raw",
                                             num_reads=2000,
                                             annealing_time=1)
    embedding = sampleset_1.info["embedding_context"]["embedding"]  
    sampleset_25 = FixedEmbeddingComposite(qpu, embedding).sample(bqm,
                                                              answer_mode="raw",
                                                              num_reads=2000,
                                                              annealing_time=25)
    print(f'first: {sampleset_25.first.sample}')
    print(f'info: {sampleset_25.info}')
    print(f'first: {sampleset_25.first.energy}')
    
    #-------Hybrid---------
    # hybrid = LeapHybridSampler(solver={'category': 'hybrid'})
    # hybrid_sampleset = hybrid.sample(bqm)
    # print(hybrid_sampleset.first)
    # print(hybrid_sampleset.info)
if __name__ == '__main__':
    main()
