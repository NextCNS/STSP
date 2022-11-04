import os
import itertools
import click
import math
import pandas as pd
from dwave.system import LeapHybridCQMSampler, DWaveSampler,FixedEmbeddingComposite, EmbeddingComposite, LeapHybridSampler
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel, QuadraticModel
from pyqubo import Array, Constraint, solve_qubo, Binary
from datetime import datetime
import numpy as np
import networkx as nx
import dwave_networkx as dnx
from dwave.system.composites import EmbeddingComposite
from collections import defaultdict
import itertools


# # P4 - C_MAX = 20
# PROFIT = [0,8,3,0]
# P5 - C_MAX = 25
# PROFIT = [0,8,3,9,0]
# p6 
PROFIT = [0,8,3,9,1,0]
N = len(PROFIT)

def selective_traveling_salesperson_qubo(G, lagrange=None, weight='weight',profits=PROFIT, CMax = 13):
    """Return the QUBO with ground states corresponding to a minimum TSP route.
    -------
    QUBO : dict
       The QUBO with ground states corresponding to a minimum travelling
       salesperson route. The QUBO variables are labelled `(c, t)` where `c`
       is a node in `G` and `t` is the time index. For instance, if `('a', 0)`
       is 1 in the ground state, that means the node 'a' is visted first.

    """
    N = G.number_of_nodes()
    # Slack variables used for C_max
    s1 = int(1 + math.log(CMax,2))

    
    if lagrange is None:
        if G.number_of_edges()>0:
            lagrange = G.size(weight='weight')*G.number_of_nodes()/G.number_of_edges()
        else:
            lagrange = 2

    # some input checking
    if N in (1, 2) or len(G.edges) != N*(N-1)//2:
        msg = "graph must be a complete graph with at least 3 nodes or empty"
        raise ValueError(msg)
    
    
    # Creating the QUBO
    Q = defaultdict(float)
    print(f'larange is {lagrange}')
    
    # #Constraint must leave the node 0
    for i in range(1,N):
        Q[((0, i), (0, i))] -= lagrange
        for j in range(i+1, N):
            Q[((0, i), (0, j))] += 2.0*lagrange
    
    # # Constrant must end at node n
    for i in range(N-1):
        Q[((i, N-1), (i, N-1))] -= lagrange
        for j in range(i+1,N-1):
            Q[((i, N-1), (j, N-1))] += 2.0*lagrange
    
    # #Constrant each node must be visited at most once
    for k in range(1,N-1):
        for i in range(N-1):
            if i != k:
                for j in range(N-1):
                    if j != i and j!=k:
                        Q[((i, k), (j, k))] += lagrange
    
    # #Constrant each node must leaft at most once
    for k in range(1,N-1):
        for i in range(1,N):
            if i != k:
                for j in range(1,N):
                    if j != i and j!=k:
                        Q[((k, i), (k, j))] += lagrange
    
    print(f'Number of slack varable s1 {s1}')
    #Constrant to make sure the cost is smaller than pre-defined value
    # print(weight)
    for i in range(0,N-1):
        for j in range(1,N):
            if j != i:
                weight_ij = weight[(i,j)] if (i,j) in weight else weight[(j,i)]
                Q[((i,j), (i,j))] += lagrange*(math.pow(weight_ij,2) - CMax* weight_ij)
                for k in range(i, N-1):
                    for h in range(1,N):
                        if (i,j) != (k,h) and h!= k:
                            weight_kh = weight[(k,h)] if (k,h) in weight else weight[(h,k)]
                            Q[((i,j), (k,h))] += lagrange*(weight_ij * weight_kh)
    # ----Slack of C_max----
    for i in range(N, N+s1):
        Q[((i,i), (i,i))] += lagrange*(pow(4,i-N) - CMax)
        for j in range (N, N+s1):
            if j!=i:
                Q[((i,i), (j,j))] += lagrange*(pow(2,i-N)*pow(2,j-N))
        for h in range(0,N-1):
            for k in range(1,N):
                if k != h:
                    weight_hk = weight[(h,k)] if (h,k) in weight else weight[(k,h)]
                    Q[((i,i), (h,k))] += lagrange*(math.pow(2,i-N + 1)*weight_hk)
    return Q

def main():
    # import data
    data = pd.read_csv('data/five_d.txt', sep='\s+', header=None)
    # G = nx.from_pandas_dataframe(data) 
    seed = 1
    np.random.seed(seed)
    G = nx.from_pandas_adjacency(data)
    # pos = nx.random_layout(G) 
    pos = nx.spring_layout(G, seed=seed)

    # get characteristics of graph
    nodes = G.nodes()
    edges = G.edges()
    weights = nx.get_edge_attributes(G,'weight')
    STSP_QUBO = selective_traveling_salesperson_qubo(G, weight=weights)
    # print(STSP_QUBO)

    # -------QPU--------
    qpu = DWaveSampler()
    bqm = BinaryQuadraticModel.from_qubo(STSP_QUBO)
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
