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

        self.host = str(serving_config.get("host", "127.0.0.1"))
        self.port = int(serving_config.get("port", 8000))
        self.api_key = str(serving_config.get("api_key", "") or "EMPTY")
        self.base_url = f"http://{self.host}:{self.port}/v1"

    # ------------------------------------------------------------------
    # Command building
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
        """Start the vLLM server.  Blocks until healthy or *timeout* seconds."""
        if self.is_running:
            logger.warning("vLLM already running (pid %d), skipping start", self._process.pid)
            return

        # Make sure the port is free before attempting to bind
        self._ensure_port_free()

        cmd = self._build_command(model_path)
        env = self._build_env()
        logger.info("Starting vLLM: %s", " ".join(cmd))

        # start_new_session=True → vLLM + its TP workers get their own
        # process group so we can kill the whole tree in one shot later.
        self._process = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        logger.info("vLLM started (pid %d, pgid %d)", self._process.pid, os.getpgid(self._process.pid))

        if wait:
            self._wait_healthy(timeout)

    def stop(self, timeout: float = 30) -> None:
        """Stop the vLLM server and release GPU memory."""
        if self._process is not None:
            self._stop_child(timeout)
        else:
            self._stop_external(timeout)

        # After killing, still make sure no straggler holds the port / GPU
        self._kill_port_holders()
        self._wait_gpu_release()

    def restart(self, model_path: str | None = None, timeout: float = 300) -> None:
        """Stop, then start with (optionally) a new model."""
        self.stop()
        self.start(model_path=model_path, wait=True, timeout=timeout)

    # ------------------------------------------------------------------
    # Internal: stop a child we started
    # ------------------------------------------------------------------

    def _stop_child(self, timeout: float) -> None:
        if not self.is_running:
            logger.info("vLLM child already exited")
            self._process = None
            return

        pid = self._process.pid
        pgid = os.getpgid(pid)
        logger.info("Stopping vLLM (pid %d, pgid %d) ...", pid, pgid)

        # 1) SIGTERM the whole process group (main + TP workers)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            self._process.wait(timeout=timeout)
            logger.info("vLLM stopped gracefully")
        except subprocess.TimeoutExpired:
            # 2) SIGKILL the whole group
            logger.warning("vLLM did not stop in %ds, SIGKILL-ing process group %d", timeout, pgid)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._process.wait(timeout=10)

        self._process = None

    # ------------------------------------------------------------------
    # Internal: stop a server we did NOT start (e.g. manual bash launch)
    # ------------------------------------------------------------------

    def _stop_external(self, timeout: float = 30) -> None:
        if not self.health_check(timeout=3):
            logger.info("No reachable vLLM server to stop")
            return

        logger.info("Stopping external vLLM on port %d ...", self.port)
        pids = self._find_port_pids()

        if pids:
            # Find all parent PIDs and kill their process trees
            for pid in pids:
                self._kill_tree(pid)
        else:
            # Fallback: pkill by command pattern
            self._pkill_vllm()

        # Wait until health check fails (server is truly gone)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.health_check(timeout=2):
                logger.info("External vLLM stopped")
                return
            time.sleep(2)

        # Absolute last resort
        logger.warning("Server still alive after %ds, force-killing all port holders", timeout)
        self._kill_port_holders()

    # ------------------------------------------------------------------
    # Process-tree utilities
    # ------------------------------------------------------------------

    def _find_port_pids(self) -> list[int]:
        """Find PIDs that are listening on / connected to self.port."""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{self.port}"],
                capture_output=True, text=True, timeout=5,
            )
            return [int(p) for p in result.stdout.split() if p.strip().isdigit()]
        except Exception:
            return []

    @staticmethod
    def _kill_tree(pid: int) -> None:
        """SIGTERM then SIGKILL an entire process tree rooted at *pid*."""
        # Try to get pgid — if the target set its own session, this is cleaner
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(3)
            os.killpg(pgid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
        # Fallback: use pkill --parent
        try:
            subprocess.run(["pkill", "-TERM", "-P", str(pid)], timeout=3)
            os.kill(pid, signal.SIGTERM)
            time.sleep(3)
            subprocess.run(["pkill", "-KILL", "-P", str(pid)], timeout=3)
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def _pkill_vllm(self) -> None:
        """Broad pkill matching vllm processes on our port."""
        for sig in ["-TERM", "-KILL"]:
            try:
                subprocess.run(
                    ["pkill", sig, "-f", f"vllm.*--port.*{self.port}"],
                    timeout=5,
                )
            except Exception:
                pass
            if sig == "-TERM":
                time.sleep(3)

    def _kill_port_holders(self) -> None:
        """SIGKILL every process still holding self.port."""
        pids = self._find_port_pids()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
                logger.info("SIGKILL sent to straggler pid %d", pid)
            except (ProcessLookupError, PermissionError):
                pass

    # ------------------------------------------------------------------
    # Port & GPU readiness
    # ------------------------------------------------------------------

    def _ensure_port_free(self, timeout: float = 60) -> None:
        """Wait until self.port is no longer in use, killing stragglers if needed."""
        deadline = time.time() + timeout
        first = True
        while self._find_port_pids():
            if first:
                logger.info("Port %d still in use, cleaning up ...", self.port)
                self._kill_port_holders()
                first = False
            if time.time() > deadline:
                raise RuntimeError(f"Port {self.port} still occupied after {timeout}s")
            time.sleep(2)

    @staticmethod
    def _wait_gpu_release(max_wait: float = 15) -> None:
        """Pause to let CUDA release GPU memory after process exit."""
        logger.info("Waiting %.0fs for GPU memory release ...", max_wait)
        time.sleep(max_wait)

    def _wait_healthy(self, timeout: float) -> None:
        logger.info("Waiting for vLLM to become healthy (up to %.0fs) ...", timeout)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited unexpectedly (code {self._process.returncode})"
                )
            if self.health_check(timeout=3):
                logger.info("vLLM is healthy")
                return
            time.sleep(5)
        raise TimeoutError(f"vLLM not healthy after {timeout}s")


def create_server_manager(config: dict[str, Any]) -> VLLMServerManager | None:
    """Create a VLLMServerManager if the config uses a vLLM/SGLang backend with serving enabled."""
    serving = config.get("serving", {})
    backend = config.get("backend", {})
    if backend.get("kind") not in ("vllm", "sglang"):
        return None
    if not serving.get("enabled", False):
        return None
    return VLLMServerManager(serving, backend)
