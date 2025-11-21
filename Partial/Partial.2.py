import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from hmmlearn.hmm import MultinomialHMM

st = ["W","R","S"]
out = ["Low", "Medium", "High"]
oidx = {g: i for i, g in enumerate(out)}

startp = np.array([0.4, 0.3, 0.3])

transp =  np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.2, 0.5],
])

emip = np.array([
    [0.1,0.7,0.2],
    [0.05,0.25,0.7],
    [0.8,0.15,0.05]
])

model = MultinomialHMM(n_components=3, init_params="")
model.startprob_ = startp
model.transmat_ = transp
model.emissionprob_ = emip

obs = ["Medium","High","Low"]
obsidx = np.array([[oidx[g]]] for g in obs)

logobs = model.score(obsidx)
p_obs = np.exp(logobs)
print(f"P(Observations) = {p_obs}")

logprob_vit, state_path = model.decode(obsidx, algorithm="viterbi")
state_names = [st[s] for s in state_path]
p_joint = np.exp(logprob_vit)
p_conditional = p_joint / p_obs

print("Log P(states*, observations) =", logprob_vit)
print("P(states*, observations) =", p_joint)
print("P(states* | observations) =", p_conditional)