"""
vLLM inference server lifecycle management.

Handles start/stop/restart of the vLLM process so that GPU memory can be
released for training and reclaimed with a (potentially new) model afterwards.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from cegsr.utils.logging import get_logger
from cegsr.utils.modeling import render_model_path_template

logger = get_logger(__name__)


class VLLMServerManager:
    """Manage a vLLM OpenAI-compatible server as a child process."""

    def __init__(self, serving_config: dict[str, Any], backend_config: dict[str, Any] | None = None) -> None:
        self._serving = serving_config
        self._backend = backend_config or {}
        self._process: subprocess.Popen[str] | None = None

        # Derived settings
        self.host = str(serving_config.get("host", "127.0.0.1"))
        self.port = int(serving_config.get("port", 8000))
        self.api_key = str(serving_config.get("api_key", "") or "EMPTY")
        self.base_url = f"http://{self.host}:{self.port}/v1"

    # ------------------------------------------------------------------
    # Build the vllm serve command
    # ------------------------------------------------------------------

    def _build_command(self, model_path: str | None = None) -> list[str]:
        s = self._serving
        model = model_path or s.get("model_name_or_path") or s.get("model")
        if not model:
            raise ValueError("No model path provided for vLLM server")

        model = render_model_path_template(model, s.get("model_size"))

        gpu_list = self._normalize_gpus(s.get("gpu_ids"))
        tp = int(s.get("tensor_parallel_size", max(1, len(gpu_list)) or 1))

        cmd = [
            "vllm", "serve", model,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", str(s.get("dtype", "auto")),
            "--tensor-parallel-size", str(tp),
            "--gpu-memory-utilization", str(s.get("gpu_memory_utilization", 0.9)),
            "--api-key", self.api_key,
        ]
        for flag, key in [("--max-model-len", "max_model_len"), ("--max-num-seqs", "max_num_seqs")]:
            val = s.get(key)
            if val is not None:
                cmd.extend([flag, str(val)])
        for extra in s.get("extra_args", []):
            cmd.append(str(extra))
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        gpu_list = self._normalize_gpus(self._serving.get("gpu_ids"))
        if gpu_list:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
        return env

    @staticmethod
    def _normalize_gpus(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        if isinstance(value, (list, tuple)):
            return [str(x).strip() for x in value]
        return [str(value)]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    def health_check(self, timeout: float = 5.0) -> bool:
        """Return True if the server responds to /v1/models."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            resp = requests.get(f"{self.base_url}/models", headers=headers, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def start(self, model_path: str | None = None, wait: bool = True, timeout: float = 300) -> None:
        """Start the vLLM server. Blocks until healthy or *timeout* seconds."""
        if self.is_running:
            logger.warning("vLLM server already running (pid %d), skipping start", self._process.pid)
            return

        cmd = self._build_command(model_path)
        env = self._build_env()
        logger.info("Starting vLLM server: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logger.info("vLLM server started (pid %d)", self._process.pid)

        if wait:
            self._wait_healthy(timeout)

    def stop(self, timeout: float = 30) -> None:
        """Gracefully stop the vLLM server, releasing GPU memory."""
        if self._process is None:
            # Try to stop an externally started server via pkill
            self._stop_external(timeout)
            return

        if not self.is_running:
            logger.info("vLLM server process already exited")
            self._process = None
            return

        pid = self._process.pid
        logger.info("Stopping vLLM server (pid %d) ...", pid)

        # SIGTERM → wait → SIGKILL
        self._process.terminate()
        try:
            self._process.wait(timeout=timeout)
            logger.info("vLLM server stopped gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("vLLM did not stop in %ds, sending SIGKILL", timeout)
            self._process.kill()
            self._process.wait(timeout=10)
        self._process = None

        # Wait for GPU memory to be freed
        self._wait_gpu_release()

    def _stop_external(self, timeout: float = 30) -> None:
        """Stop an externally started vLLM server (not our child process)."""
        if not self.health_check(timeout=3):
            logger.info("No reachable vLLM server to stop")
            return

        logger.info("Stopping external vLLM server on port %d ...", self.port)
        try:
            # Find vLLM processes by port
            result = subprocess.run(
                ["lsof", "-ti", f":{self.port}"],
                capture_output=True, text=True, timeout=5,
            )
            pids = [int(p.strip()) for p in result.stdout.strip().split("\n") if p.strip()]
            for pid in pids:
                logger.info("Sending SIGTERM to pid %d", pid)
                os.kill(pid, signal.SIGTERM)
        except Exception as exc:
            logger.warning("Failed to find/kill vLLM process: %s", exc)
            # Fallback: pkill
            try:
                subprocess.run(["pkill", "-f", f"vllm.*--port.*{self.port}"], timeout=5)
            except Exception:
                pass

        # Wait for the server to actually go away
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.health_check(timeout=2):
                logger.info("External vLLM server stopped")
                self._wait_gpu_release()
                return
            time.sleep(2)
        logger.warning("External vLLM server may still be running after %ds", timeout)

    def restart(self, model_path: str | None = None, timeout: float = 300) -> None:
        """Stop, then start with (optionally) a new model."""
        self.stop()
        self.start(model_path=model_path, wait=True, timeout=timeout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_healthy(self, timeout: float) -> None:
        logger.info("Waiting for vLLM server to become healthy (up to %.0fs) ...", timeout)
        deadline = time.time() + timeout
        interval = 5
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited unexpectedly (code {self._process.returncode})"
                )
            if self.health_check(timeout=3):
                logger.info("vLLM server is healthy")
                return
            time.sleep(interval)
        raise TimeoutError(f"vLLM server not healthy after {timeout}s")

    @staticmethod
    def _wait_gpu_release(max_wait: float = 15) -> None:
        """Brief pause to let CUDA free GPU memory after process exit."""
        logger.info("Waiting %.0fs for GPU memory release ...", max_wait)
        time.sleep(max_wait)


def create_server_manager(config: dict[str, Any]) -> VLLMServerManager | None:
    """Create a VLLMServerManager from a full experiment config, or None if not applicable."""
    serving = config.get("serving", {})
    backend = config.get("backend", {})
    if backend.get("kind") not in ("vllm", "sglang"):
        return None
    if not serving.get("enabled", False):
        return None
    return VLLMServerManager(serving, backend)
