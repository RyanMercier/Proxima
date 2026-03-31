#!/usr/bin/env python3
"""
wallet.py -- Interactive wallet client.

Connects to a running node. Provides a REPL for sending transactions,
checking balances, and viewing chain status. Run one per user.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
import readline  # enables arrow keys and history in REPL


class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


class WalletClient:
    def __init__(self, name: str, node_url: str):
        self.name = name
        self.node = node_url.rstrip("/")

    def _get(self, path: str) -> dict:
        url = f"{self.node}{path}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"error": body}
        except urllib.error.URLError as e:
            return {"error": f"cannot reach node: {e.reason}"}

    def _post(self, path: str, data: dict) -> dict:
        url = f"{self.node}{path}"
        payload = json.dumps(data).encode()
        try:
            req = urllib.request.Request(url, data=payload,
                                        headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"error": body}
        except urllib.error.URLError as e:
            return {"error": f"cannot reach node: {e.reason}"}

    def register(self, balance: float = 0.0) -> dict:
        return self._post("/register", {"name": self.name, "balance": balance})

    def balance(self) -> dict:
        return self._get(f"/balance/{self.name}")

    def send(self, receiver: str, amount: float) -> dict:
        return self._post("/tx", {
            "sender": self.name,
            "receiver": receiver,
            "amount": amount,
        })

    def history(self) -> dict:
        return self._get(f"/history/{self.name}")

    def status(self) -> dict:
        return self._get("/status")

    def mempool(self) -> dict:
        return self._get("/mempool")

    def chain(self) -> dict:
        return self._get("/chain")

    def block(self, height: int) -> dict:
        return self._get(f"/block/{height}")


def print_help():
    cmds = [
        ("send <name> <amount>", "Send coins to another account"),
        ("balance", "Check your balance"),
        ("history", "Transaction history"),
        ("status", "Chain and node status"),
        ("mempool", "View pending transactions"),
        ("chain", "Chain info (height, supply)"),
        ("block <height>", "View block details"),
        ("help", "Show this help"),
        ("quit", "Exit wallet"),
    ]
    print(f"\n{C.BOLD}Commands:{C.RESET}")
    for cmd, desc in cmds:
        print(f"  {C.CYAN}{cmd:<25}{C.RESET} {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Interactive wallet client")
    parser.add_argument("--name", required=True, help="Account name")
    parser.add_argument("--node", default="http://localhost:8545", help="Node URL")
    args = parser.parse_args()

    client = WalletClient(args.name, args.node)

    # Register if not already registered
    reg = client.register()
    if "error" in reg:
        print(f"{C.RED}Error: {reg['error']}{C.RESET}")
        sys.exit(1)

    # Show welcome
    bal = client.balance()
    balance = bal.get("balance", 0.0)
    print(f"\n{C.BOLD}Welcome {args.name}!{C.RESET} Balance: {C.GREEN}{balance:.2f} PROX{C.RESET}")
    print(f"Connected to {args.node}")
    print(f"Type 'help' for commands.\n")

    prompt = f"{C.CYAN}{args.name.lower()}{C.RESET}> "

    while True:
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}Goodbye.{C.RESET}")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "quit" or cmd == "exit":
            print(f"{C.DIM}Goodbye.{C.RESET}")
            break

        elif cmd == "help":
            print_help()

        elif cmd == "balance" or cmd == "bal":
            resp = client.balance()
            if "error" in resp:
                print(f"{C.RED}Error: {resp['error']}{C.RESET}")
            else:
                bal = resp["balance"]
                pending = resp.get("pending_out", 0)
                avail = resp.get("available", bal)
                print(f"{C.GREEN}{bal:.2f} PROX{C.RESET}", end="")
                if pending > 0:
                    print(f" ({C.YELLOW}{avail:.2f} after pending txs{C.RESET})", end="")
                print()

        elif cmd == "send":
            if len(parts) < 3:
                print(f"{C.YELLOW}Usage: send <name> <amount>{C.RESET}")
                continue
            receiver = parts[1]
            try:
                amount = float(parts[2])
            except ValueError:
                print(f"{C.RED}Invalid amount{C.RESET}")
                continue
            resp = client.send(receiver, amount)
            if "error" in resp:
                print(f"{C.RED}Error: {resp['error']}{C.RESET}")
            else:
                h = resp.get("tx_hash", "")[:12]
                print(f"TX submitted: {C.GREEN}{h}...{C.RESET} (pending)")

        elif cmd == "history":
            resp = client.history()
            if "error" in resp:
                print(f"{C.RED}Error: {resp['error']}{C.RESET}")
                continue
            hist = resp.get("history", [])
            pending = resp.get("pending", [])
            if not hist and not pending:
                print(f"{C.DIM}No transactions yet.{C.RESET}")
                continue
            for h in hist:
                direction = h["direction"]
                if direction == "sent":
                    print(f"  Block #{h['block']}: {C.YELLOW}sent{C.RESET} {h['amount']:.2f} to {h['other']}")
                elif direction == "received":
                    print(f"  Block #{h['block']}: {C.GREEN}received{C.RESET} {h['amount']:.2f} from {h['other']}")
                elif direction == "mined":
                    print(f"  Block #{h['block']}: {C.CYAN}mined{C.RESET} {h['amount']:.2f} (coinbase)")
            if pending:
                print(f"  {C.YELLOW}Pending:{C.RESET}")
                for p in pending:
                    print(f"    {p['sender']} -> {p['receiver']} {p['amount']:.2f}")

        elif cmd == "status":
            resp = client.status()
            if "error" in resp:
                print(f"{C.RED}Error: {resp['error']}{C.RESET}")
            else:
                h = resp.get("height", 0)
                mp = resp.get("mempool_size", 0)
                supply = resp.get("supply", 0)
                print(f"Chain height: {h} | Mempool: {mp} txs | Supply: {supply:.2f} PROX")
                last = resp.get("last_consensus")
                if last:
                    fp = "fast path" if last.get("fast_path") else "full protocol"
                    print(f"Last block: {fp} | {last['msgs']} msgs | {last['time']:.3f}s")

        elif cmd == "mempool":
            resp = client.mempool()
            if "error" in resp:
                print(f"{C.RED}Error: {resp['error']}{C.RESET}")
            else:
                txs = resp.get("transactions", [])
                if not txs:
                    print(f"{C.DIM}Mempool empty.{C.RESET}")
                else:
                    print(f"{len(txs)} pending:")
                    for tx in txs:
                        print(f"  {tx['sender']} -> {tx['receiver']} {tx['amount']:.2f}")

        elif cmd == "chain":
            resp = client.chain()
            if "error" in resp:
                print(f"{C.RED}Error: {resp['error']}{C.RESET}")
            else:
                print(f"Height: {resp['height']} | Tip: {resp['tip']} | Supply: {resp['supply']:.2f}")

        elif cmd == "block":
            if len(parts) < 2:
                print(f"{C.YELLOW}Usage: block <height>{C.RESET}")
                continue
            try:
                height = int(parts[1])
            except ValueError:
                print(f"{C.RED}Invalid height{C.RESET}")
                continue
            resp = client.block(height)
            if "error" in resp:
                print(f"{C.RED}Error: {resp['error']}{C.RESET}")
            else:
                print(f"Block #{resp['height']} | Hash: {resp['block_hash']}")
                print(f"Proposer: {resp['proposer']} | Txs: {resp['n_transactions']}")
                for tx in resp.get("transactions", []):
                    if tx.get("type") == "coinbase":
                        print(f"  {C.CYAN}coinbase{C.RESET} -> {tx['receiver']} {tx['amount']:.2f}")
                    else:
                        print(f"  {tx['sender']} -> {tx['receiver']} {tx['amount']:.2f}")

        else:
            print(f"{C.YELLOW}Unknown command: {cmd}. Type 'help' for commands.{C.RESET}")


if __name__ == "__main__":
    main()
