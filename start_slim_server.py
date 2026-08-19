"""Start the LiteLLM slim gateway (service entry point).

Used by the NSSM service instead of `-c "from litellm import run_server;
run_server()"` because nssm strips quotes from AppParameters when stored in the
registry, breaking the inline `-c` code. A script file has a space-free path and
needs no quotes, so it survives nssm's argument handling untouched.
"""
from litellm import run_server

if __name__ == "__main__":
    run_server()
