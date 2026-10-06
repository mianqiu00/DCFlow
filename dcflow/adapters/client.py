"""JSONL client isolating PyTorch generation from the JAX folding backend."""

import json
import os
import subprocess

PREFIX = "DCFLOW_JSON\t"


class AFCycClient:
    def __init__(self, command, env=None):
        environment = {
            **os.environ,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            **(env or {}),
        }
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.counter = 0
        try:
            response = self._read()
            if not response.get("ready"):
                raise RuntimeError(str(response))
        except BaseException:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            if line.startswith(PREFIX):
                return json.loads(line[len(PREFIX) :])
            print(line.rstrip(), flush=True)
        raise RuntimeError(f"AFCyc worker exited with code {self.process.poll()}")

    def request(self, command, **kwargs):
        self.counter += 1
        self.process.stdin.write(
            json.dumps(dict(id=self.counter, command=command, **kwargs)) + "\n"
        )
        self.process.stdin.flush()
        response = self._read()
        if response.get("id") != self.counter:
            raise RuntimeError("Worker response ID mismatch")
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def evaluate(self, sequences):
        return self.request("evaluate", sequences=sequences)["results"]

    def close(self):
        if self.process.poll() is not None:
            return
        try:
            self.process.stdin.write(json.dumps(dict(command="shutdown")) + "\n")
            self.process.stdin.flush()
            self.process.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
