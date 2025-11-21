from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

edges = [("O", "H"), ("O", "W"), ("W", "R"), ("H", "R"), ("H", "E"), ("R", "C")]
model = DiscreteBayesianNetwork(edges)

cpd_O = TabularCPD('O', 2, [[0.3], [0.7]])

cpd_H = TabularCPD(
    'H', 2,
    [[0.8, 0.1],
     [0.2, 0.9]],
    evidence=['O'],
    evidence_card=[2]
)
cpd_W = TabularCPD(
    'W', 2,
    [[0.4, 0.9],
     [0.6, 0.1]],
    evidence=['O'],
    evidence_card=[2]
)
cpd_R = TabularCPD(
    'R', 2,
    [[0.5, 0.7, 0.1, 0.4],
     [0.5, 0.3, 0.9, 0.6]],
    evidence=['H', 'W'],
    evidence_card=[2, 2]
)
cpd_E = TabularCPD(
    'E', 2,
    [[0.8, 0.2],
     [0.2, 0.8]],
    evidence=['H'],
    evidence_card=[2]
)
cpd_C = TabularCPD(
    'C', 2,
    [[0.6, 0.15],
     [0.4, 0.85]],
    evidence=['R'],
    evidence_card=[2]
)

model.add_cpds(cpd_O,cpd_H,cpd_W,cpd_R,cpd_E,cpd_C)
graph = nx.DiGraph()
graph.add_edges_from(edges)
plt.figure(figsize=(5, 5))
nx.draw(graph, with_labels=True, arrows=True, node_size=1000)
plt.show()

infer = VariableElimination(model)

print("P(H|C) = %.3f" % infer.query(['H'], evidence={'C':1}).values[1])
print("P(E|C) = %.3f" % infer.query(['E'], evidence={'C':1}).values[1])
mapq = infer.map_query(['H','W'], evidence={'C':1})
print(f"MAP(H,W|C) = {mapq}")

print("\nIndependencies:")
print(model.get_independencies())
