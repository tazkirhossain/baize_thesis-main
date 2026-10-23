"""Production hyperparameters for the E1–E9 scripts.

Frozen from DECISIONS.md D-017 (learning rate / epoch budget after
convergence), D-020 (federated local training), and D-024 (60 rounds).

`ExperimentConfig` keeps smaller defaults so unit tests stay fast. Every
script that produces a reported number must import from here instead of
re-stating magic numbers — that is how the under-trained lr=2 path in
`scripts/phase2_baseline.py` survived next to the converged lr=160 path.
"""

N_FEATURES = 2**18
NGRAM_RANGE = (3, 5)
LR = 160.0
BATCH = 512
L2 = 1e-7
CENTRALIZED_EPOCHS = 35
ROUNDS = 60
LOCAL_EPOCHS = 3
N_CLIENTS = 10
SPLIT = "domain_aware"
