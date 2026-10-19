"""Talk to the service from a terminal.

    export ANALYST_KEY=...                                # from: python keys.py create <user>
    python chat.py "¿Cuánto contrató la Gobernación del Cauca en 2025?"
    python chat.py --thread a1b2c3d4 "¿Y en obras?"      # continue a conversation

Prints the answer as it arrives and the tools the agent calls, and asks before an expensive query runs.
"""

import argparse
import json
import os
import sys
import urllib.request
import uuid

BASE = os.environ.get("ANALYST_URL", "http://localhost:8000")
KEY = os.environ.get("ANALYST_KEY", "")


def post(path: str, body: dict) -> dict | None:
    """Sends a request, prints its event stream and returns the approval it is waiting for, if any."""
    request = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json",
                                                                         "Authorization": f"Bearer {KEY}"})
    waiting, name = None, ""
    with urllib.request.urlopen(request) as response:
        for line in response:
            line = line.decode().rstrip("\n")
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if name == "token":
                    print(data, end="", flush=True)
                elif name == "tool":
                    print(f"\n[{data['name']}] {json.dumps(data['args'], ensure_ascii=False)}", flush=True)
                elif name == "approval":
                    waiting = data
                elif name == "error":
                    print(f"\nerror: {data['message']} (trace {data['trace_id']})")
    return waiting


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("message")
    parser.add_argument("--thread", default=uuid.uuid4().hex[:8], help="conversation to continue; a new one by default")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not KEY:
        sys.exit("Set ANALYST_KEY to a key from: python keys.py create <user>")
    who = {"thread_id": args.thread}
    waiting = post("/chat", {**who, "message": args.message})
    while waiting:
        print(f"\nConsulta costosa (costo estimado {waiting['cost']:.0f}, límite {waiting['limit']:.0f}):\n  {waiting['sql']}")
        waiting = post("/resume", {**who, "approve": input("¿Ejecutarla? [s/N] ").strip().lower().startswith("s")})
    print(f"\n\n(hilo {args.thread})")
