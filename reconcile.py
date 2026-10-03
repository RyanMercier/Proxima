"""
reconcile.py -- Exact set reconciliation with IBLTs.

Used wherever two parties hold sets but no shared list of identifiers:
mempool sync between peers, partition healing, fork healing, and decoding
dangling cross-shard transactions. (Inside consensus a validator that holds
the proposal already knows which ids it lacks and fetches them by id; no
sketch is needed there.)

Estimate, then decode: a sketch estimate (dpmh) sizes the table; an
Invertible Bloom Lookup Table (Goodrich-Mitzenmacher) recovers exactly
which keys differ; on decode failure the table doubles. The same
estimate-then-BCH pattern appears in PBS (Gong et al., VLDB 2021); rateless
IBLTs (Yang et al., SIGCOMM 2024) and CertainSync remove the need for an
estimate at some cost in rounds. Our only addition is that the estimate is
already on the wire for other reasons.

Keys are 8-byte short ids salted per session (as in Erlay), so an adversary
cannot grind two transactions with colliding keys ahead of time. The
decoder only knows its own items: keys that decode to the other side are
fetched in one extra round trip, and that cost is counted.
"""

import hashlib

CELL_BYTES = 16          # count (4) + key_sum (8) + check_sum (4)
KEY_BYTES = 8
ITEM_BYTES = 200         # transaction body pushed after decode
HASH_COUNT = 3
ALPHA = 1.5
MIN_CELLS = 16


def _key64(item: str, session: int = 0) -> int:
    h = hashlib.sha256(b"PXM/sid" + session.to_bytes(8, "big") + item.encode())
    return int.from_bytes(h.digest()[:8], "big")


def _check(key: int) -> int:
    return int.from_bytes(
        hashlib.sha256(b"chk" + key.to_bytes(8, "big")).digest()[:4], "big")


def _cells_for(key: int, m: int, salt: int) -> list:
    idxs, i = [], 0
    while len(idxs) < HASH_COUNT:
        h = hashlib.sha256(salt.to_bytes(4, "big") + i.to_bytes(2, "big")
                           + key.to_bytes(8, "big")).digest()
        idx = int.from_bytes(h[:8], "big") % m
        if idx not in idxs:
            idxs.append(idx)
        i += 1
    return idxs


class IBLT:
    def __init__(self, m: int, salt: int = 0):
        self.m = m
        self.salt = salt
        self.count = [0] * m
        self.key_sum = [0] * m
        self.chk_sum = [0] * m

    def insert_key(self, key: int, sign: int = 1) -> None:
        chk = _check(key)
        for idx in _cells_for(key, self.m, self.salt):
            self.count[idx] += sign
            self.key_sum[idx] ^= key
            self.chk_sum[idx] ^= chk

    def insert(self, item: str, sign: int = 1, session: int = 0) -> None:
        self.insert_key(_key64(item, session), sign)

    @classmethod
    def of(cls, items, m: int, salt: int = 0, session: int = 0) -> "IBLT":
        t = cls(m, salt)
        for it in items:
            t.insert(it, 1, session)
        return t

    def subtract(self, other: "IBLT") -> "IBLT":
        assert self.m == other.m and self.salt == other.salt
        out = IBLT(self.m, self.salt)
        out.count = [a - b for a, b in zip(self.count, other.count)]
        out.key_sum = [a ^ b for a, b in zip(self.key_sum, other.key_sum)]
        out.chk_sum = [a ^ b for a, b in zip(self.chk_sum, other.chk_sum)]
        return out

    def peel(self):
        """Return (keys only in self-side, keys only in other-side, ok)."""
        count, key_sum, chk_sum = (list(self.count), list(self.key_sum),
                                   list(self.chk_sum))
        a_keys, b_keys = set(), set()
        progress = True
        while progress:
            progress = False
            for idx in range(self.m):
                c = count[idx]
                if c not in (1, -1):
                    continue
                key = key_sum[idx]
                if key == 0 or _check(key) != chk_sum[idx]:
                    continue
                (a_keys if c == 1 else b_keys).add(key)
                for j in _cells_for(key, self.m, self.salt):
                    count[j] -= c
                    key_sum[j] ^= key
                    chk_sum[j] ^= _check(key)
                progress = True
        ok = not any(count) and not any(key_sum) and not any(chk_sum)
        return a_keys, b_keys, ok


def size_for(d_hint: float) -> int:
    """Cell count for an expected difference of d_hint elements."""
    return max(MIN_CELLS, int(ALPHA * (d_hint + 6)) + 1)


def reconcile(set_a, set_b, d_hint: float = None, max_rounds: int = 8,
              session: int = 0) -> dict:
    """Requester A learns A delta B from responder B.

    B sends an IBLT of its short ids sized by d_hint; A subtracts its own
    table and peels. Keys on A's side map to A's items directly. Keys on
    B's side are unknown to A, so A requests them (8 B each) and B returns
    the bodies: one extra round trip, counted in bytes and rounds. On
    decode failure the table doubles. d_hint=None means no estimate was
    available; the table then starts at the minimum size.
    """
    set_a, set_b = set(set_a), set(set_b)
    true_d = len(set_a ^ set_b)
    m = size_for(1.0 if d_hint is None else d_hint)
    bytes_used, rounds = 0, 0
    for attempt in range(max_rounds):
        sess = session * 1009 + attempt          # fresh short ids per attempt
        keys_a = {_key64(x, sess): x for x in set_a}
        keys_b = {_key64(x, sess): x for x in set_b}   # responder-side only
        if len(keys_a) != len(set_a) or len(keys_b) != len(set_b):
            continue                             # short-id collision: rekey
        rounds += 1
        t_a, t_b = IBLT(m, attempt), IBLT(m, attempt)
        for k in keys_a:
            t_a.insert_key(k)
        for k in keys_b:
            t_b.insert_key(k)
        bytes_used += m * CELL_BYTES
        a_keys, b_keys, ok = t_a.subtract(t_b).peel()
        if ok and a_keys <= keys_a.keys():
            only_a = {keys_a[k] for k in a_keys}
            only_b = {keys_b[k] for k in b_keys if k in keys_b}
            fetch = 0
            if b_keys:
                rounds += 1
                fetch = len(b_keys) * KEY_BYTES + len(only_b) * ITEM_BYTES
                bytes_used += fetch
            return {"only_a": only_a, "only_b": only_b, "ok": True,
                    "bytes": bytes_used, "fetch_bytes": fetch,
                    "rounds": rounds, "cells": m, "true_d": true_d}
        m *= 2
    return {"only_a": set(), "only_b": set(), "ok": False,
            "bytes": bytes_used, "fetch_bytes": 0, "rounds": rounds,
            "cells": m, "true_d": true_d}


def sync_cost(d_hint: float) -> int:
    """Happy-path sketch bytes for a one-shot table sized by d_hint."""
    return size_for(d_hint) * CELL_BYTES
