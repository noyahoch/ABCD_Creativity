"""vLLM server lifecycle, owned by the pipeline.

``managed_vllm(cfg)`` returns a context manager that:

  * skips entirely when no judge stage uses a vLLM backend, or when a healthy
    OpenAI-compatible server already answers at the configured ``base_url``
    (``reuse_existing: true``);
  * otherwise spawns ``python -m vllm.entrypoints.openai.api_server`` with the
    flags from ``cfg.vllm_server``, waits until ``GET {base_url}/models``
    returns 200, yields to the caller, and on exit sends ``SIGTERM`` to the
    whole process group (falling back to ``SIGKILL`` after
    ``terminate_timeout_s``).

It does **not** decide *when* to start the server — ``pipeline.run_pipeline``
opens the context only around the contiguous slice of stages that need vLLM,
so GPU-heavy inference stages still get the full device.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

from .config import RunConfig, VllmJudgeBackend, VllmServerSpec


def judge_uses_vllm(cfg: RunConfig) -> bool:
    """True iff at least one configured judge stage targets a vLLM backend."""
    for jc in (cfg.aut_judge, cfg.macgyver_judge):
        if jc is not None and isinstance(jc.backend, VllmJudgeBackend):
            return True
    return False


def vllm_judge_base_url(cfg: RunConfig) -> Optional[str]:
    """Pick a base_url from the configured vLLM judge backends (any wins).

    Used only for the health probe / reuse-existing check; the per-stage
    backend.base_url is what actually talks to the server.
    """
    for jc in (cfg.aut_judge, cfg.macgyver_judge):
        if jc is not None and isinstance(jc.backend, VllmJudgeBackend):
            return jc.backend.base_url
    return None


def _is_healthy(base_url: str, timeout_s: float = 2.0) -> bool:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError,
            socket.timeout, ConnectionError, OSError):
        return False


def _build_argv(spec: VllmServerSpec) -> list[str]:
    argv: list[str] = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", spec.model,
        "--host", spec.host,
        "--port", str(spec.port),
        "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
        "--tensor-parallel-size", str(spec.tensor_parallel_size),
    ]
    if spec.quantization:
        argv += ["--quantization", spec.quantization]
    if spec.max_model_len is not None:
        argv += ["--max-model-len", str(spec.max_model_len)]
    if spec.dtype:
        argv += ["--dtype", spec.dtype]
    argv += list(spec.extra_args)
    return argv


@contextlib.contextmanager
def managed_vllm(cfg: RunConfig) -> Iterator[Optional[subprocess.Popen]]:
    """Spawn vLLM for the duration of the ``with`` block, then tear it down.

    Yields the ``subprocess.Popen`` handle, or ``None`` when no spawn occurred
    (no vLLM judge configured, server disabled, or an existing server was
    reused).
    """
    spec = cfg.vllm_server
    if spec is None or not spec.enable or not judge_uses_vllm(cfg):
        yield None
        return

    base_url = vllm_judge_base_url(cfg) or f"http://localhost:{spec.port}/v1"

    if spec.reuse_existing and _is_healthy(base_url):
        print(f"[vllm] reusing existing healthy server at {base_url}", flush=True)
        yield None
        return

    log_path = Path(spec.log_to) if spec.log_to is not None else (
        cfg.output_root / cfg.run_id / "vllm_server.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    argv = _build_argv(spec)
    print(f"[vllm] starting: {' '.join(argv)}", flush=True)
    print(f"[vllm] logs → {log_path}", flush=True)

    log_fh = open(log_path, "ab", buffering=0)

    # VLLM_CUDA_VISIBLE_DEVICES pins the judge to its own GPU(s), independent of
    # the GPUs the HF model is sharded over. Needed when a judge must be LIVE
    # during HF generation (denial reviewer) and the HF model is too large to
    # leave room beside it — e.g. Mistral-24B (~48 GB): sharing a card starves
    # the judge's KV cache, since the Qwen-32B-AWQ weights alone are 18.1 GiB.
    # Unset (the common case) → inherit the parent's CUDA_VISIBLE_DEVICES.
    env = None
    pin = os.environ.get("VLLM_CUDA_VISIBLE_DEVICES")
    if pin:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=pin)
        print(f"[vllm] pinned to CUDA_VISIBLE_DEVICES={pin}", flush=True)

    proc = subprocess.Popen(
        argv,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group → clean teardown
        env=env,
    )

    try:
        deadline = time.monotonic() + spec.startup_timeout_s
        while True:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited with code {proc.returncode} before becoming "
                    f"healthy; see {log_path}"
                )
            if _is_healthy(base_url):
                print(f"[vllm] healthy at {base_url}", flush=True)
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"vLLM not healthy after {spec.startup_timeout_s:.0f}s at "
                    f"{base_url}; see {log_path}"
                )
            time.sleep(spec.poll_interval_s)

        yield proc

    finally:
        # NB: never use ``return`` here — a ``return`` inside ``finally``
        # silently suppresses any exception raised in the ``with`` body.
        if spec.keep_alive and proc.poll() is None:
            # The api_server spawns an internal EngineCore worker whose
            # cmdline is ``VLLM::EngineCore`` (not ``vllm.entrypoints``), so
            # ``pkill -f vllm.entrypoints`` orphans it. Killing the whole
            # process group cleans up both. We started the server with
            # ``start_new_session=True`` so ``proc.pid`` is the pgid.
            print(
                f"[vllm] keep_alive=true — leaving server running at {base_url} "
                f"(PID {proc.pid}). Stop the whole server (api_server + "
                f"EngineCore worker) with:\n"
                f"        kill -- -{proc.pid}",
                flush=True,
            )
            log_fh.close()
        else:
            if proc.poll() is None:
                print("[vllm] stopping server …", flush=True)
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=spec.terminate_timeout_s)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=30)
            log_fh.close()
            print("[vllm] stopped", flush=True)
