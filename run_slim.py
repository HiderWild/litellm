"""LiteLLM slim gateway launcher for NSSM service.

Avoids shell/argument-quoting issues: the service runs
  python.exe run_slim.py --config config.local.yaml --port 9374 --slim
No embedded quotes anywhere. All CLI args are forwarded to litellm.run_server.
"""
import sys
from litellm import run_server

if __name__ == "__main__":
    sys.exit(run_server())
