import numpy as np
import networkx as nx
import dimod
import dwave_networkx as dnx
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dwave.system.composites import EmbeddingComposite
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
import pandas as pd
import math

profit = [3,4,5,6,7]
