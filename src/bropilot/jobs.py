from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .utils.common import stable_hash

JOB_FILE = "job.json"
LOG_FILE = "job.log"
RESULT_FILE = "job_result.json"

STAGES = ["ingest", "asr", "boards", "board_reading", "segmentation", "compose"]
STAGE_TITLES = {
    "ingest": "Загрузка видео и аудио",
    "asr": "Распознавание речи",
    "boards": "Поиск кадров доски",
    "board_reading": "Чтение доски (VLM)",
    "segmentation": "Разбиение на разделы",
    "compose": "Генерация конспекта и PDF",
}

_STAGE_EVENT = re.compile(r"\[(\w+)\] (running|done in|cache hit)")
_STAGE_SECONDS = re.compile(r"\[(\w+)\] done in ([\d.]+)s")
_SRC_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _SRC_ROOT.parent

_PROCS: dict[str, subprocess.Popen] = {}


def workdir_for(runs_root: Path, source: str) -> Path:
    return Path(runs_root) / stable_hash(source)


def start_job(
    runs_root: Path,
    source: str,
    *,
    config_name: str = "default",
    overrides: list[str] | None = None,
    until: str | None = None,
    force: bool = False,
    env_extra: dict[str, str] | None = None,
) -> Path:
    runs_root = Path(runs_root).resolve()
    wd = workdir_for(runs_root, source)
    wd.mkdir(parents=True, exist_ok=True)
    (wd / RESULT_FILE).unlink(missing_ok=True)

    cli_args = ["--config", config_name, "run", source,
                f"paths.workdir={runs_root.as_posix()}", *(overrides or [])]
    if until:
        cli_args += ["--until", until]
    if force:
        cli_args.append("--force")

    env = os.environ.copy()
    env.update(env_extra or {})
    env["PYTHONPATH"] = str(_SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [sys.executable, "-m", "bropilot.jobs", str(wd), "--", *cli_args]
    log = open(wd / LOG_FILE, "w", encoding="utf-8")  # noqa: SIM115
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(_PROJECT_ROOT), env=env, **kwargs
    )
    log.close()
    _PROCS[str(wd)] = proc
    (wd / JOB_FILE).write_text(
        json.dumps(
            {"pid": proc.pid, "source": source, "started": time.time(),
             "until": until, "args": cli_args},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return wd


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, errors="replace",
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def is_running(wd: Path) -> bool:
    wd = Path(wd)
    proc = _PROCS.get(str(wd))
    if proc is not None:
        return proc.poll() is None
    job = wd / JOB_FILE
    if not job.exists() or (wd / RESULT_FILE).exists():
        return False
    try:
        return _pid_alive(int(json.loads(job.read_text(encoding="utf-8"))["pid"]))
    except (ValueError, KeyError, json.JSONDecodeError):
        return False


def stop_job(wd: Path) -> None:
    wd = Path(wd)
    proc = _PROCS.get(str(wd))
    if proc is not None and proc.poll() is None:
        proc.terminate()
    else:
        try:
            pid = int(json.loads((wd / JOB_FILE).read_text(encoding="utf-8"))["pid"])
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            else:
                os.kill(pid, 15)
        except (OSError, ValueError, KeyError, FileNotFoundError, json.JSONDecodeError):
            pass
    (wd / RESULT_FILE).write_text(
        json.dumps({"returncode": -15, "error": "остановлено пользователем"}, ensure_ascii=False),
        encoding="utf-8",
    )


def status(wd: Path) -> dict:
    wd = Path(wd)
    info: dict = {"exists": (wd / JOB_FILE).exists(), "running": is_running(wd),
                  "stages": {}, "stage_seconds": {}, "current": None,
                  "log_tail": "", "result": None, "started": None,
                  "finished": None, "elapsed": None}
    if (wd / JOB_FILE).exists():
        info.update(json.loads((wd / JOB_FILE).read_text(encoding="utf-8")))
    log_path = wd / LOG_FILE
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for stage, event in _STAGE_EVENT.findall(text):
            state = "running" if event == "running" else "done"
            info["stages"][stage] = state
            if state == "running":
                info["current"] = stage
        for stage, seconds in _STAGE_SECONDS.findall(text):
            info["stage_seconds"][stage] = float(seconds)
        lines = text.splitlines()
        info["log_tail"] = "\n".join(lines[-40:])
        progress = [ln for ln in lines if "progress" in ln]
        info["progress_line"] = progress[-1].split("|")[-1].strip() if progress else ""
    if (wd / RESULT_FILE).exists():
        info["result"] = json.loads((wd / RESULT_FILE).read_text(encoding="utf-8"))
        info["finished"] = info["result"].get("finished")

    started = info.get("started")
    if started:
        end = info["finished"] if not info["running"] and info.get("finished") else time.time()
        info["elapsed"] = max(0.0, float(end) - float(started))
    return info


def delete_run(wd: Path) -> None:
    wd = Path(wd)
    if is_running(wd):
        stop_job(wd)
        for _ in range(40):
            if not is_running(wd):
                break
            time.sleep(0.25)
    _PROCS.pop(str(wd), None)
    shutil.rmtree(wd, ignore_errors=True)
    if wd.exists():
        raise OSError(f"не удалось полностью удалить {wd}")


def _main() -> int:
    wd = Path(sys.argv[1])
    cli_args = sys.argv[sys.argv.index("--") + 1 :]
    result: dict = {"returncode": 1, "error": None}
    try:
        from .cli import main as cli_main

        result["returncode"] = int(cli_main(cli_args) or 0)
    except SystemExit as exc:
        result["returncode"] = exc.code if isinstance(exc.code, int) else 1
        if not isinstance(exc.code, int):
            result["error"] = str(exc.code)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["finished"] = time.time()
        if result["returncode"] and not result["error"]:
            result["error"] = f"процесс завершился с кодом {result['returncode']} — подробности в журнале"
        (wd / RESULT_FILE).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        sys.stdout.flush()
    return int(result["returncode"] or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
