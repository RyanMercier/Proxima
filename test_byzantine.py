#!/usr/bin/env python3
"""Byzantine strategy checks for the flat protocol (assertions, not prints).

Each strategy runs with 20 txs and no partial observation so the Byzantine
effect is isolated. Large-deviation strategies must be excluded by the
distance filter; small-deviation ones may sit inside the cluster, which is
harmless because finality needs valid signatures on the proposal's hash.
Run: python test_byzantine.py
"""

import numpy as np

import blockchain
blockchain.USE_REAL_BLS = False

from blockchain import (
    BLSKeyPair, Blockchain, make_validators, calibrate_threshold,
    vector_consensus,
)

np.random.seed(11)

EXCLUDED = {"drop_half", "random_vector", "coalition"}
MAY_STAY = {"replace_one_tx", "mimic_honest"}
FAILS = 0


def run(strategy, n_txs=20, n_honest=4, n_byz=1):
    BLSKeyPair._counter = 1
    all_v, honest, _byz = make_validators(n_honest, n_byz, strategy)
    chain = Blockchain(all_v)
    for i in range(5):
        chain.register_account(f"A{i}", 100_000.0)
    for i in range(n_txs):
        tx = chain.make_tx(f"A{i % 5}", f"A{(i + 1) % 5}", 1.0)
        if tx:
            chain.submit_tx(tx)
    block = chain.propose_block(honest[0])
    return vector_consensus(all_v, block, calibrate_threshold(), partial_obs={})


def expect(name, cond, detail=""):
    global FAILS
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    FAILS += 0 if cond else 1


for strategy in sorted(EXCLUDED | MAY_STAY):
    r = run(strategy)
    excluded_byz = [e for e in r["excluded"] if e[1]]
    excluded_honest = [e for e in r["excluded"] if not e[1]]
    expect(f"{strategy}: finalizes", r["finalized"])
    expect(f"{strategy}: no honest validator excluded", not excluded_honest,
           str(excluded_honest))
    if strategy in EXCLUDED:
        expect(f"{strategy}: Byzantine excluded", len(excluded_byz) == 1,
               str(r["distances"]))

print(f"\n{'all passed' if not FAILS else f'{FAILS} failed'}")
raise SystemExit(1 if FAILS else 0)
