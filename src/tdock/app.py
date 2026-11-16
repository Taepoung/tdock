"""Docker 관리 TUI.

컨테이너/이미지/GPU/정리 탭. 마운트·GPU·DooD를 지정해 컨테이너를 만들거나,
기존 컨테이너를 commit 후 설정을 바꿔 다시 만들 수 있다.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from rich.text import Text

from . import services as svc
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.suggester import SuggestFromList
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

LOCALTIME = "/etc/localtime"
TZ_LOCALTIME = "TZ=:/etc/localtime"   # 이미지의 TZ 설정을 덮어쓰고 마운트한 파일을 쓰게 함 (glibc, musl 모두)
ZONEINFO = "/usr/share/zoneinfo"      # 데몬 호스트의 시간대 파일. 이미지에 tzdata가 없어도 된다
TZ_SERVER, TZ_IMAGE, TZ_OTHER = "server", "image", "other"
COMMON_ZONES = ["Asia/Seoul", "UTC"]


@functools.cache
def all_zones() -> list[str]:
    try:
        import zoneinfo
        return sorted(zoneinfo.available_timezones())
    except Exception:
        return COMMON_ZONES

DEFAULT_SHM = 64 * 1024 * 1024
DOOD_SOCKET_DEST = "/var/run/docker.sock"
DOOD_PLUGINS = "/usr/libexec/docker/cli-plugins"


# ---------------------------------------------------------------- docker 호출

class DockerError(Exception):
    pass


def docker(*args: str, check: bool = True) -> str:
    p = subprocess.run(["docker", *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise DockerError((p.stderr or p.stdout).strip() or f"docker {args[0]} failed")
    return p.stdout


def docker_json_lines(*args: str) -> list[dict]:
    return [json.loads(line) for line in docker(*args).splitlines() if line.strip()]


def inspect(*names: str) -> list[dict]:
    if not names:
        return []
    return json.loads(docker("inspect", *names))


def container_exists(name: str) -> bool:
    return subprocess.run(["docker", "container", "inspect", name],
                          capture_output=True).returncode == 0


def host_socket() -> str:
    """호스트 Docker 소켓 경로 (DOCKER_HOST > 현재 context)."""
    host = os.environ.get("DOCKER_HOST", "")
    if not host:
        try:
            host = docker("context", "inspect", "--format", "{{.Endpoints.docker.Host}}").strip()
        except DockerError:
            host = ""
    if host.startswith("unix://"):
        return host[len("unix://"):]
    return "/var/run/docker.sock"


def docker_binary() -> str:
    return os.path.realpath(shutil.which("docker") or "/usr/bin/docker")


def dood_mounts() -> list[str]:
    mounts = [f"{host_socket()}:{DOOD_SOCKET_DEST}", f"{docker_binary()}:/usr/bin/docker:ro"]
    if os.path.isdir(DOOD_PLUGINS):
        mounts.append(f"{DOOD_PLUGINS}:{DOOD_PLUGINS}:ro")
    return mounts


@functools.cache
def is_rootless() -> bool:
    try:
        return "rootless" in docker("info", "--format", "{{json .SecurityOptions}}")
    except DockerError:
        return False


@functools.cache
def docker_caps() -> dict:
    """데몬 호스트의 CPU/메모리 용량과 제한 지원 여부."""
    try:
        info = json.loads(docker("info", "--format", "{{json .}}"))
    except (DockerError, ValueError):
        info = {}
    return {
        "ncpu": info.get("NCPU") or os.cpu_count() or 1,
        "mem_gib": round((info.get("MemTotal") or 0) / (1 << 30)),
        "cpu_limit": bool(info.get("CPUCfsQuota")),
        "mem_limit": bool(info.get("MemoryLimit", True)),
    }


def size_bytes(v: str) -> int:
    m = re.fullmatch(r"(\d+)([bkmg]?)", v.strip(), re.I)
    if not m:
        return 0
    return int(m[1]) * {"": 1, "b": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30}[m[2].lower()]


def list_gpus() -> list[tuple[str, str, int]]:
    """(index, 이름, 메모리 GiB)"""
    if not shutil.which("nvidia-smi"):
        return []
    p = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    out = []
    for line in p.stdout.strip().splitlines():
        idx, name, mem = [x.strip() for x in line.split(",")]
        out.append((idx, name.replace("NVIDIA ", ""), round(int(mem) / 1024)))
    return out


VIRTUAL_FS = {"overlay", "tmpfs", "proc", "sysfs", "devpts", "mqueue", "cgroup", "cgroup2", "devtmpfs",
              "securityfs", "debugfs", "tracefs", "fusectl", "configfs", "pstore", "bpf", "autofs",
              "hugetlbfs", "binfmt_misc", "nsfs", "rpc_pipefs", "efivarfs"}
PATHMAP_CACHE = os.path.expanduser("~/.config/tdock/pathmap.json")
STATE_FILE = os.path.expanduser("~/.config/tdock/state.json")


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(**kw) -> None:
    state = load_state() | kw
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=1)
    except OSError:
        pass


def _unescape(s: str) -> str:
    return s.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


class PathMap:
    """이 TUI가 컨테이너 안에서 돌 때 로컬 경로를 Docker 데몬 호스트 경로로 바꾼다.

    같은 디스크가 여러 경로에 마운트돼 있으면(예: /data 와 /home/user/data) 데몬이
    실제로 보는 경로를 작은 컨테이너로 확인해 고르고, 결과를 캐시한다.
    컨테이너 자체 파일시스템(/tmp 등)은 데몬에서 보이지 않으므로 None.
    """

    def __init__(self) -> None:
        self.in_container = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
        self.mounts: list[dict] = []
        try:
            with open("/proc/self/mountinfo") as f:
                for line in f:
                    left, _, right = line.partition(" - ")
                    a, b = left.split(), right.split()
                    self.mounts.append({"mp": _unescape(a[4]), "dev": a[2], "root": _unescape(a[3]), "fs": b[0]})
        except OSError:
            pass
        self.host_of: dict[str, str | None] = {}
        self.unverified = False
        self.ready = not self.in_container
        self.lock = threading.Lock()

    def real_mounts(self) -> list[dict]:
        return [m for m in self.mounts if m["fs"] not in VIRTUAL_FS and os.path.isdir(m["mp"])]

    def mount_for(self, path: str) -> dict | None:
        best = None
        for m in self.mounts:
            mp = m["mp"]
            if mp == "/" or path == mp or path.startswith(mp.rstrip("/") + "/"):
                if best is None or len(mp) >= len(best["mp"]):
                    best = m
        return best

    def ensure(self) -> None:
        if self.ready:
            return
        with self.lock:
            if self.ready:
                return
            real = self.real_mounts()
            key = hashlib.sha1(json.dumps(sorted((m["mp"], m["dev"], m["root"]) for m in real)).encode()).hexdigest()
            try:
                with open(PATHMAP_CACHE) as f:
                    cache = json.load(f)
                if cache.get("key") == key:
                    self.host_of, self.ready = cache["map"], True
                    return
            except (OSError, ValueError, KeyError):
                pass
            image = self._probe_image()
            for m in real:
                self.host_of[m["mp"]] = self._probe(m, real, image) if image else m["mp"]
            self.unverified = image is None
            if image:
                os.makedirs(os.path.dirname(PATHMAP_CACHE), exist_ok=True)
                with open(PATHMAP_CACHE, "w") as f:
                    json.dump({"key": key, "map": self.host_of}, f, indent=1)
            self.ready = True

    @staticmethod
    def _probe_image() -> str | None:
        try:
            imgs = [i for i in docker_json_lines("images", "--format", "{{json .}}") if i["Repository"] != "<none>"]
        except DockerError:
            return None
        small = [i for i in imgs if i["Repository"].split("/")[-1] in ("alpine", "busybox", "ubuntu", "debian")]
        for i in small + imgs:
            return f"{i['Repository']}:{i['Tag']}"
        return None

    def _probe(self, m: dict, real: list[dict], image: str) -> str | None:
        try:
            want = sorted(os.listdir(m["mp"]))[:200]
        except OSError:
            return m["mp"]
        cands = [m["mp"]]
        for n in real:
            if n is not m and n["dev"] == m["dev"] and (
                    m["root"] == n["root"] or m["root"].startswith(n["root"].rstrip("/") + "/")):
                cands.append(os.path.normpath(n["mp"] + "/" + m["root"][len(n["root"].rstrip("/")):]))
        for cand in dict.fromkeys(cands):
            p = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "ls",
                                "--mount", f"type=bind,src={cand},dst=/m,readonly", image, "-1A", "/m"],
                               capture_output=True, text=True)
            if p.returncode == 0 and sorted(x for x in p.stdout.split("\n") if x)[:200] == want:
                return cand
        return None

    def translate(self, path: str) -> str | None:
        """로컬 경로 → 데몬 호스트 경로. 데몬에서 안 보이면 None."""
        path = os.path.realpath(os.path.expanduser(path))
        if not self.in_container:
            return path
        self.ensure()
        m = self.mount_for(path)
        if m is None or m["fs"] in VIRTUAL_FS:
            return None
        host = self.host_of.get(m["mp"], m["mp"])
        if host is None:
            return None
        rel = path[len(m["mp"].rstrip("/")):]
        return (host.rstrip("/") + rel) or "/"

    def summary(self) -> list[str]:
        if not self.in_container:
            return []
        self.ensure()
        out = [f"{mp} → {h}" for mp, h in self.host_of.items() if h and h != mp]
        if self.unverified:
            out.append("(no image available to verify path translation)")
        return out


def detect_server_tz() -> str:
    """데몬 호스트의 시간대 약어와 오프셋 (예: 'UTC +0000'). 확인용 이미지가 없으면 빈 문자열."""
    image = PathMap._probe_image()
    if not image:
        return ""
    p = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "date",
                        "-v", f"{LOCALTIME}:{LOCALTIME}:ro", "-e", TZ_LOCALTIME, image, "+%Z %z"],
                       capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def human_size(n: int) -> str:
    for unit, k in (("g", 1 << 30), ("m", 1 << 20), ("k", 1 << 10)):
        if n % k == 0:
            return f"{n // k}{unit}"
    return str(n)


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return str(n)


# ---------------------------------------------------------------- 컨테이너 설정

@dataclass
class RunSpec:
    name: str = ""
    image: str = ""
    gpus: str = "all"            # "" | "all" | "0,1"
    workdir: str = ""            # 호스트 작업 폴더: 같은 경로 + /workspace 로 마운트
    mounts: list[str] = field(default_factory=list)   # host:container[:ro]
    ports: list[str] = field(default_factory=list)    # [ip:]host:container[/udp]
    env: list[str] = field(default_factory=list)      # KEY=VAL
    dood: bool = False
    restart: bool = False        # --restart unless-stopped
    shm: str = "16g"             # --shm-size, 비우면 기본값(64m)
    cpus: str = ""               # --cpus (데몬이 지원할 때만 폼에 나옴)
    memory: str = ""             # --memory
    ipc_host: bool = False       # --ipc host
    ulimits: bool = True         # NVIDIA 권장 ulimit
    ptrace: bool = False         # --cap-add SYS_PTRACE
    tz: str = TZ_SERVER          # "server" | "image" | 시간대 이름 (예: Asia/Seoul)
    extra: str = ""              # 추가 docker run 인자
    command: str = ""            # 비우면 이미지 기본 CMD
    hook: bool = True            # 서비스 자동 시작 훅 (/etc/tdock/startup.sh)
    base_argv: Optional[list] = None   # 다시 만들 때 기존 컨테이너의 원래 명령

    def args(self, argv: Optional[list] = None) -> list[str]:
        """argv: 훅으로 감쌀 원래 명령 (resolve_argv). None 이면 훅 없이 만든다."""
        a = ["docker", "run", "-d"]
        if self.name:
            a += ["--name", self.name]
        if self.gpus == "all":
            a += ["--gpus", "all"]
        elif self.gpus:
            a += ["--gpus", f'"device={self.gpus}"']
        if self.restart:
            a += ["--restart", "unless-stopped"]
        if self.ipc_host:
            a += ["--ipc", "host"]
        elif self.shm:
            a.append(f"--shm-size={self.shm}")
        if self.cpus:
            a.append(f"--cpus={self.cpus}")
        if self.memory:
            a.append(f"--memory={self.memory}")
        if self.ulimits:
            if not is_rootless():
                a += ["--ulimit", "memlock=-1"]
            a += ["--ulimit", "stack=67108864"]
        if self.ptrace:
            a += ["--cap-add", "SYS_PTRACE"]
        if self.dood:
            for m in dood_mounts():
                a += ["-v", m]
        # 직접 /etc/localtime 마운트나 TZ 환경변수를 넣었으면 그쪽을 따른다
        tz_on = self.tz != TZ_IMAGE and not any(m.split(":")[1:2] == [LOCALTIME] for m in self.mounts) \
            and not any(e.startswith("TZ=") for e in self.env)
        if tz_on:
            src = LOCALTIME if self.tz == TZ_SERVER else f"{ZONEINFO}/{self.tz}"
            a += ["--mount", f"type=bind,src={src},dst={LOCALTIME},readonly"]
        mounts = list(self.mounts)
        if self.workdir:
            wd = os.path.abspath(os.path.expanduser(self.workdir))
            mounts += [f"{wd}:{wd}", f"{wd}:/workspace"]
        for m in mounts:
            parts = m.split(":")
            if parts[0].startswith("/"):
                # --mount: 호스트에 경로가 없으면 빈 폴더를 만들지 않고 에러
                opt = f"type=bind,src={parts[0]},dst={parts[1]}"
                a += ["--mount", opt + (",readonly" if parts[2:] == ["ro"] else "")]
            else:
                a += ["-v", m]
        for p in self.ports:
            a += ["-p", p]
        env = list(self.env)
        if tz_on:
            env.append(TZ_LOCALTIME)
        for e in env:
            a += ["-e", e]
        if self.extra:
            a += shlex.split(self.extra)
        if self.hook and argv is not None:
            pre, post = svc.hook_args(argv)
            return a + pre + [self.image] + post
        a.append(self.image)
        if self.command:
            a += shlex.split(self.command)
        return a


    def validate(self) -> list[str]:
        errs = []
        if not self.image:
            errs.append("Enter an image.")
        if self.gpus not in ("", "all") and not all(x.strip().isdigit() for x in self.gpus.split(",")):
            errs.append("GPU must be empty, all, or a list like 0,1.")
        for m in self.mounts:
            parts = m.split(":")
            if len(parts) < 2 or not parts[0] or not parts[1]:
                errs.append(f"Invalid mount: {m}  (host_path:container_path[:ro])")
            elif "," in m:
                errs.append(f"Paths cannot contain commas: {m}")
        dests = [m.split(":")[1] for m in self.mounts if len(m.split(":")) >= 2]
        if self.dood:
            dests += [m.split(":")[1] for m in dood_mounts()]
        for d in sorted({d for d in dests if dests.count(d) > 1}):
            errs.append(f"Duplicate container path: {d}")
        if self.shm and not self.ipc_host and not re.fullmatch(r"\d+[bkmg]?", self.shm, re.I):
            errs.append("Shared memory size must look like 16g or 512m.")
        caps = docker_caps()
        if self.cpus:
            try:
                c = float(self.cpus)
                if not 0 < c <= caps["ncpu"]:
                    errs.append(f"CPUs must be greater than 0 and at most {caps['ncpu']}.")
            except ValueError:
                errs.append("CPUs must be a number (e.g. 16, 0.5).")
        if self.memory:
            mem = size_bytes(self.memory)
            if not mem:
                errs.append("Memory must look like 64g or 512m.")
            elif mem < 6 << 20:
                errs.append("Memory limit must be at least 6m.")
            elif caps["mem_gib"] and mem > caps["mem_gib"] << 30:
                errs.append(f"Memory can be at most {caps['mem_gib']}GiB.")
            elif self.shm and not self.ipc_host and size_bytes(self.shm) >= mem:
                errs.append("Shared memory must be smaller than the memory limit. It counts toward the limit and can cause OOM.")
        if self.workdir and not os.path.isdir(os.path.expanduser(self.workdir)):
            errs.append(f"Work folder not found: {self.workdir}")
        if self.tz not in (TZ_SERVER, TZ_IMAGE) and self.tz not in all_zones():
            errs.append(f"Unknown timezone: {self.tz or '(empty)'}  (e.g. Asia/Seoul, Europe/London)")
        try:
            shlex.split(self.extra)
            shlex.split(self.command)
        except ValueError as e:
            errs.append(f"Argument quoting error: {e}")
        return errs


def resolve_argv(spec: RunSpec) -> Optional[list]:
    """훅으로 감쌀 원래 명령 (entrypoint + cmd). 이미지가 로컬에 없으면 None."""
    if spec.command and spec.base_argv is not None:
        return shlex.split(spec.command)
    if not spec.command and spec.base_argv is not None:
        return list(spec.base_argv)
    try:
        cfg = inspect(spec.image)[0].get("Config") or {}
    except (DockerError, IndexError, ValueError):
        return None
    ep, cmd = cfg.get("Entrypoint") or [], cfg.get("Cmd") or []
    inner = svc.unwrap_hook(ep, cmd)
    if inner is not None:              # tdock 이 commit 한 이미지
        return shlex.split(spec.command) if spec.command else inner
    return ep + (shlex.split(spec.command) if spec.command else cmd)


def has_sh(image: str) -> bool:
    return subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/sh", image,
                           "-c", "true"], capture_output=True).returncode == 0


def container_argv(c: dict) -> list:
    """컨테이너의 원래 명령 (훅으로 감쌌으면 벗겨서)."""
    cfg = c.get("Config") or {}
    inner = svc.unwrap_hook(cfg.get("Entrypoint"), cfg.get("Cmd"))
    return inner if inner is not None else (cfg.get("Entrypoint") or []) + (cfg.get("Cmd") or [])


def has_hook_or_script(name: str) -> bool:
    return subprocess.run(["docker", "exec", name, "test", "-f", svc.SCRIPT], capture_output=True).returncode == 0


def has_hook(c: dict) -> bool:
    return ((c.get("Config") or {}).get("Labels") or {}).get(svc.HOOK_LABEL) == "1"


def gpu_desc(c: dict) -> str:
    for req in (c.get("HostConfig") or {}).get("DeviceRequests") or []:
        if ["gpu"] in (req.get("Capabilities") or []) or req.get("Driver") == "nvidia":
            if req.get("Count") == -1:
                return "all"
            if req.get("DeviceIDs"):
                return ",".join(req["DeviceIDs"])
            if req.get("Count"):
                return f"{req['Count']} GPUs"
    return ""


def mount_str(m: dict) -> str | None:
    if m.get("Type") == "bind":
        s = f"{m['Source']}:{m['Destination']}"
    elif m.get("Type") == "volume":
        s = f"{m['Name']}:{m['Destination']}"
    else:
        return None
    return s if m.get("RW", True) else s + ":ro"


def ports_of(c: dict) -> list[str]:
    out = []
    for cport, binds in ((c.get("HostConfig") or {}).get("PortBindings") or {}).items():
        num, _, proto = cport.partition("/")
        for b in binds or []:
            s = f"{b.get('HostPort', '')}:{num}"
            if b.get("HostIp"):
                s = f"{b['HostIp']}:{s}"
            if proto and proto != "tcp":
                s += f"/{proto}"
            out.append(s)
    return out


def is_dood(c: dict) -> bool:
    return any(m.get("Destination") == DOOD_SOCKET_DEST for m in c.get("Mounts") or [])


def tz_of(c: dict) -> str:
    for m in c.get("Mounts") or []:
        if m.get("Destination") == LOCALTIME:
            src = m.get("Source", "")
            return src[len(ZONEINFO) + 1:] if src.startswith(ZONEINFO + "/") else TZ_SERVER
    return TZ_IMAGE


def spec_from_container(c: dict) -> tuple[RunSpec, list[str]]:
    """기존 컨테이너 설정을 RunSpec으로. 옮기지 못한 설정은 경고로 돌려준다."""
    hc = c.get("HostConfig") or {}
    cfg = c.get("Config") or {}
    dood_dests = {DOOD_SOCKET_DEST, "/usr/bin/docker", DOOD_PLUGINS, LOCALTIME}
    mounts = [s for m in c.get("Mounts") or []
              if m.get("Destination") not in dood_dests and (s := mount_str(m))]
    try:
        image_env = set((inspect(c["Image"])[0].get("Config") or {}).get("Env") or [])
    except DockerError:
        image_env = set()
    extra, warns = [], []
    net = hc.get("NetworkMode", "")
    if net and net not in ("default", "bridge"):
        extra += ["--network", net]
    # 폼에 없는 옵션은 추가 인자로 옮겨 그대로 유지
    if hc.get("Init"):
        extra.append("--init")
    if hc.get("Privileged"):
        extra.append("--privileged")
    if hc.get("AutoRemove"):
        extra.append("--rm")
    if cfg.get("User"):
        extra += ["--user", cfg["User"]]
    caps = hc.get("CapAdd") or []
    for cap in caps:
        if cap.upper() not in ("SYS_PTRACE", "CAP_SYS_PTRACE"):
            extra += ["--cap-add", cap]
    ulimits = {u["Name"]: u for u in hc.get("Ulimits") or []}
    for name, u in ulimits.items():
        if name not in ("stack", "memlock"):
            extra += ["--ulimit", f"{name}={u['Soft']}:{u['Hard']}"]
    if hc.get("Devices"):
        warns.append("--device settings are not carried over. Add them to extra args if needed.")
    if any(n not in ("bridge",) for n in ((c.get("NetworkSettings") or {}).get("Networks") or {})) \
            and net in ("default", "bridge"):
        warns.append("Additional Docker networks must be reconnected (docker network connect).")
    env = [e for e in cfg.get("Env") or [] if e not in image_env]
    cpus = f"{hc['NanoCpus'] / 1e9:g}" if hc.get("NanoCpus") else ""
    spec = RunSpec(
        name=c["Name"].lstrip("/"),
        image=cfg.get("Image", ""),
        gpus="all" if gpu_desc(c).endswith(" GPUs") else gpu_desc(c),
        mounts=mounts,
        ports=ports_of(c),
        env=[e for e in env if e != TZ_LOCALTIME],
        dood=is_dood(c),
        restart=(hc.get("RestartPolicy") or {}).get("Name") == "unless-stopped",
        shm=human_size(hc["ShmSize"]) if hc.get("ShmSize") and hc["ShmSize"] != DEFAULT_SHM else "",
        cpus=cpus,
        memory=human_size(hc["Memory"]) if hc.get("Memory") else "",
        ipc_host=hc.get("IpcMode") == "host",
        ulimits="stack" in ulimits,
        ptrace=any(cap.upper() in ("SYS_PTRACE", "CAP_SYS_PTRACE") for cap in caps),
        tz=tz_of(c),
        extra=shlex.join(extra),
        base_argv=container_argv(c),
    )
    return spec, warns


# ---------------------------------------------------------------- 공용 위젯

# 기본 체크박스는 꺼져 있어도 흐린 X가 보여 구분이 어렵다: 꺼짐은 빈 칸, 켜짐은 채운 칸
Checkbox.BUTTON_INNER = "✔"


class GpuToggle(Checkbox):
    """선택하면 카드 전체가 채워지는 GPU 선택 버튼."""


# ---------------------------------------------------------------- 공용 모달

class Confirm(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "dismiss(False)", "Cancel"), Binding("y", "dismiss(True)", "Yes")]

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__()
        self.message, self.detail = message, detail

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(Text(self.message, style="bold"))
            if self.detail:
                with VerticalScroll(id="confirm-detail"):
                    yield Static(Text(self.detail))
            with Horizontal(classes="buttons"):
                yield Button("Yes (y)", variant="error", id="yes")
                yield Button("Cancel (esc)", id="no")

    @on(Button.Pressed)
    def pressed(self, e: Button.Pressed) -> None:
        self.dismiss(e.button.id == "yes")


class TaskLog(ModalScreen[None]):
    """여러 단계 작업의 진행 로그."""

    BINDINGS = [Binding("escape,q", "close", "Close")]

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title_text = title
        self.done = False

    def compose(self) -> ComposeResult:
        with Vertical(id="tasklog"):
            yield Static(Text(self.title_text, style="bold"))
            yield RichLog(wrap=True, markup=False)
            yield Button("Close (esc)", id="close", disabled=True)

    def write(self, line: str, style: str = "") -> None:
        self.query_one(RichLog).write(Text(line, style=style))

    def finish(self) -> None:
        self.done = True
        self.query_one("#close", Button).disabled = False
        self.query_one("#close", Button).focus()

    def action_close(self) -> None:
        if self.done:
            self.dismiss()

    @on(Button.Pressed, "#close")
    def close_pressed(self) -> None:
        self.action_close()


class LogsView(ModalScreen[None]):
    BINDINGS = [Binding("escape,q", "dismiss", "Close"), Binding("r", "reload", "Refresh")]

    def __init__(self, name: str) -> None:
        super().__init__()
        self.cname = name

    def compose(self) -> ComposeResult:
        with Vertical(id="tasklog"):
            yield Static(Text(f"Logs: {self.cname}  (last 1000 lines, r refresh, esc close)", style="bold"))
            yield RichLog(wrap=True, markup=False, max_lines=5000)

    def on_mount(self) -> None:
        self.action_reload()

    def action_reload(self) -> None:
        log = self.query_one(RichLog)
        log.clear()
        p = subprocess.run(["docker", "logs", "--tail", "1000", self.cname],
                           capture_output=True, text=True)
        out = (p.stdout + p.stderr).rstrip()
        log.write(Text(out or "(no logs)"))


# ---------------------------------------------------------------- 마운트/포트 편집기

def _ancestor(widget, cls):
    node = widget.parent
    while node is not None and not isinstance(node, cls):
        node = node.parent
    return node


class DirTree(DirectoryTree):
    """폴더만 보여주는 호스트 탐색기. → 마운트 추가, Backspace 상위, o 들어가기."""

    BINDINGS = [
        Binding("right", "add_mount", "Add mount"),
        Binding("backspace", "go_up", "Parent folder"),
        Binding("o", "go_into", "Open folder"),
        Binding("r", "refresh_tree", "Refresh folders"),
    ]

    def filter_paths(self, paths):
        out = []
        for p in paths:
            try:
                if p.is_dir() and not p.name.startswith("."):
                    out.append(p)
            except OSError:
                pass
        return out

    def selected_path(self) -> str | None:
        node = self.cursor_node
        return str(node.data.path) if node is not None and node.data is not None else None

    def action_add_mount(self) -> None:
        _ancestor(self, MountEditor).add_selected()

    def action_go_up(self) -> None:
        _ancestor(self, MountEditor).go_up()

    def action_go_into(self) -> None:
        _ancestor(self, MountEditor).go_into()

    def action_refresh_tree(self) -> None:
        _ancestor(self, MountEditor).refresh_tree()


class MountTable(DataTable):
    BINDINGS = [Binding("left,delete", "remove", "Remove mount"), Binding("r", "toggle_ro", "Read-only")]

    def action_remove(self) -> None:
        _ancestor(self, MountEditor).remove_selected()

    def action_toggle_ro(self) -> None:
        _ancestor(self, MountEditor).toggle_ro()


class MountEditor(Vertical):
    """왼쪽 호스트 폴더 트리 → 오른쪽 컨테이너 마운트 목록."""

    class Changed(Message):
        pass

    def __init__(self, mounts: list[str], root: str) -> None:
        super().__init__(classes="editor-box")
        self.items: list[list] = []          # [host, dest, ro]
        for m in mounts:
            parts = m.split(":")
            if len(parts) >= 2:
                self.items.append([parts[0], parts[1], len(parts) > 2 and parts[2] == "ro"])
        self.root = root
        self._syncing = False

    def compose(self) -> ComposeResult:
        with Horizontal(classes="editor"):
            with Vertical(classes="pane-left"):
                yield Label("Host (outside)  pick a folder, then →   (Backspace up, o open, r refresh)", classes="pane-title")
                with Horizontal(classes="pathbar"):
                    yield Button("↑ Up", id="m-up", tooltip="Go to the parent folder (Backspace)")
                    yield Button("↓ Open", id="m-into", tooltip="Open the selected folder (o)")
                    yield Input(self.root, id="m-root", placeholder="Type a path and press Enter")
                    yield Button("Refresh", id="m-refresh", tooltip="Refresh folders, e.g. after creating one (r)")
                yield DirTree(self.root, id="m-tree")
            with Vertical(classes="mid"):
                yield Button("→", id="m-add", variant="primary", tooltip="Mount the selected folder at the same path")
                yield Button("⇉Workdir", id="m-addwork", tooltip="Mount at the same path and at /workspace")
                yield Button("←", id="m-del", variant="error", tooltip="Remove the selected mount")
            with Vertical(classes="pane-right"):
                yield Label("Container (inside)  ← remove, r read-only", classes="pane-title")
                yield MountTable(id="m-table", cursor_type="row")
                with Horizontal(classes="editrow"):
                    yield Input(id="m-dest", placeholder="Container path of the selected mount")
                    yield Checkbox("ro", id="m-ro")

    def on_mount(self) -> None:
        t = self.query_one(MountTable)
        t.add_column("Outside (host)", key="host")
        t.add_column("", key="arrow")
        t.add_column("Inside (container)", key="dest")
        t.add_column("ro", key="ro")
        self.redraw()

    def mount_strings(self) -> list[str]:
        return [f"{h}:{d}" + (":ro" if ro else "") for h, d, ro in self.items]

    def redraw(self, select: int | None = None) -> None:
        t = self.query_one(MountTable)
        keep = select if select is not None else t.cursor_row
        t.clear()
        for i, (h, d, ro) in enumerate(self.items):
            t.add_row(Text(h, style="cyan"), "→", Text(d, style="green"), "✓" if ro else "", key=str(i))
        if self.items:
            t.move_cursor(row=min(max(keep, 0), len(self.items) - 1))
        self.sync_edit()
        self.post_message(self.Changed())

    def current(self) -> int | None:
        t = self.query_one(MountTable)
        return t.cursor_row if self.items and 0 <= t.cursor_row < len(self.items) else None

    def sync_edit(self) -> None:
        i = self.current()
        self._syncing = True
        dest, ro = self.query_one("#m-dest", Input), self.query_one("#m-ro", Checkbox)
        dest.value = self.items[i][1] if i is not None else ""
        ro.value = self.items[i][2] if i is not None else False
        dest.disabled = ro.disabled = i is None
        self._syncing = False

    def add(self, host: str, dest: str) -> None:
        if any(d == dest for _, d, _ in self.items):
            self.app.notify(f"{dest} is already mounted.", severity="warning")
            return
        self.items.append([host, dest, False])
        self.redraw(select=len(self.items) - 1)

    def add_selected(self, work: bool = False) -> None:
        local = self.query_one(DirTree).selected_path()
        if not local:
            self.app.notify("Select a folder on the left.", severity="warning")
            return
        path = self.app.pathmap.translate(local)
        if path is None:
            self.app.notify(f"{local} is not visible to the Docker host (it is inside this environment).\n"
                            "Use a folder under /data or /home.", severity="error", timeout=8)
            return
        self.add(path, path)
        if work:
            self.add(path, "/workspace")

    def remove_selected(self) -> None:
        i = self.current()
        if i is not None:
            del self.items[i]
            self.redraw(select=i)

    def toggle_ro(self) -> None:
        i = self.current()
        if i is not None:
            self.items[i][2] = not self.items[i][2]
            self.redraw(select=i)

    def set_root(self, path: str, focus_child: str | None = None) -> None:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(path):
            self.app.notify(f"Folder not found: {path}", severity="error")
            return
        self.root = path
        self.query_one("#m-root", Input).value = path
        tree = self.query_one(DirTree)
        tree.path = path
        if focus_child:
            self._cursor_to(focus_child, tries=20)

    def _cursor_to(self, child: str, tries: int) -> None:
        """트리가 다 읽힌 뒤 child 폴더에 커서를 둔다 (↑ 직후 ↓ 로 바로 돌아오도록)."""
        tree = self.query_one(DirTree)
        for node in tree.root.children:
            if node.data is not None and str(node.data.path) == child:
                tree.move_cursor(node)
                return
        if tries > 0:
            self.set_timer(0.1, lambda: self._cursor_to(child, tries - 1))

    def go_up(self) -> None:
        if self.root != "/":
            self.set_root(str(Path(self.root).parent), focus_child=self.root)

    def refresh_tree(self) -> None:
        # 펼친 폴더와 커서 위치는 reload()가 유지한다
        self.query_one(DirTree).reload()
        self.app.notify("Folders refreshed", timeout=2)

    def go_into(self) -> None:
        path = self.query_one(DirTree).selected_path()
        if not path or path == self.root:
            self.app.notify("Select a folder in the tree to open.", severity="warning")
            return
        self.set_root(path)

    @on(Button.Pressed, "#m-add")
    def _add(self) -> None:
        self.add_selected()

    @on(Button.Pressed, "#m-addwork")
    def _addwork(self) -> None:
        self.add_selected(work=True)

    @on(Button.Pressed, "#m-del")
    def _del(self) -> None:
        self.remove_selected()

    @on(Button.Pressed, "#m-up")
    def _up(self) -> None:
        self.go_up()

    @on(Button.Pressed, "#m-into")
    def _into(self) -> None:
        self.go_into()

    @on(Button.Pressed, "#m-refresh")
    def _refresh(self) -> None:
        self.refresh_tree()

    @on(Input.Submitted, "#m-root")
    def _root(self, e: Input.Submitted) -> None:
        self.set_root(e.value)

    @on(DataTable.RowHighlighted, "#m-table")
    def _row(self) -> None:
        self.sync_edit()

    @on(Input.Changed, "#m-dest")
    def _dest(self, e: Input.Changed) -> None:
        i = self.current()
        if self._syncing or i is None or not e.value.strip():
            return
        self.items[i][1] = e.value.strip()
        self.query_one(MountTable).update_cell(str(i), "dest", Text(self.items[i][1], style="green"))
        self.post_message(self.Changed())

    @on(Checkbox.Changed, "#m-ro")
    def _ro(self, e: Checkbox.Changed) -> None:
        i = self.current()
        if self._syncing or i is None:
            return
        self.items[i][2] = e.value
        self.query_one(MountTable).update_cell(str(i), "ro", "✓" if e.value else "")
        self.post_message(self.Changed())


class PortTable(DataTable):
    BINDINGS = [Binding("delete,backspace", "remove", "Remove port")]

    def action_remove(self) -> None:
        _ancestor(self, PortEditor).remove_selected()


class PortEditor(Vertical):
    """왼쪽에서 외부/내부 포트를 입력해 + 로 추가, 오른쪽 목록에서 빼기."""

    class Changed(Message):
        pass

    def __init__(self, ports: list[str]) -> None:
        super().__init__(classes="editor-box")
        self.items: list[list[str]] = []     # [host, container]  host는 ip:port 일 수 있음
        for p in ports:
            h, _, c = p.rpartition(":")
            self.items.append([h, c])

    def compose(self) -> ComposeResult:
        with Horizontal(classes="editor ports"):
            with Vertical(classes="port-left"):
                yield Label("Outside (host port)", classes="pane-title")
                yield Input(id="p-host", placeholder="e.g. 2222", type="integer")
                yield Label("Inside (container port)  empty = same as outside", classes="pane-title")
                yield Input(id="p-cont", placeholder="e.g. 22  (UDP: 53/udp)")
                yield Button("+ Add", id="p-add", variant="primary")
            with Vertical(classes="pane-right"):
                yield Label("Ports  (select one, then Delete or the remove button)", classes="pane-title")
                yield PortTable(id="p-table", cursor_type="row")
                yield Button("- Remove selected", id="p-del", variant="error")

    def on_mount(self) -> None:
        t = self.query_one(PortTable)
        t.add_column("Outside (host)")
        t.add_column("")
        t.add_column("Inside (container)")
        self.redraw()

    def port_strings(self) -> list[str]:
        return [f"{h}:{c}" for h, c in self.items]

    def redraw(self) -> None:
        t = self.query_one(PortTable)
        keep = t.cursor_row
        t.clear()
        for i, (h, c) in enumerate(self.items):
            t.add_row(Text(h, style="cyan"), "→", Text(c, style="green"), key=str(i))
        if self.items:
            t.move_cursor(row=min(max(keep, 0), len(self.items) - 1))
        self.query_one("#p-del", Button).disabled = not self.items
        self.post_message(self.Changed())

    @on(Input.Submitted, "#p-host")
    def _next(self) -> None:
        self.query_one("#p-cont", Input).focus()

    @on(Button.Pressed, "#p-add")
    @on(Input.Submitted, "#p-cont")
    def add(self) -> None:
        hi, ci = self.query_one("#p-host", Input), self.query_one("#p-cont", Input)
        host, cont = hi.value.strip(), ci.value.strip() or hi.value.strip()
        num, _, proto = cont.partition("/")
        if not host.isdigit() or not num.isdigit() or proto not in ("", "tcp", "udp"):
            self.app.notify("Ports must be numbers (the inside port may be like 53/udp).", severity="warning")
            return
        if any(h.rpartition(":")[2] == host and c.partition("/")[2] == proto for h, c in self.items):
            self.app.notify(f"Host port {host} is already used.", severity="warning")
            return
        self.items.append([host, cont])
        hi.value = ci.value = ""
        hi.focus()
        self.redraw()

    @on(Button.Pressed, "#p-del")
    def remove_selected(self) -> None:
        t = self.query_one(PortTable)
        if self.items and 0 <= t.cursor_row < len(self.items):
            del self.items[t.cursor_row]
            self.redraw()


# ---------------------------------------------------------------- 이미지 선택

def image_exists(ref: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", ref], capture_output=True).returncode == 0


@functools.cache
def hub_username() -> str:
    for line in docker("info", check=False).splitlines():
        if line.strip().startswith("Username:"):
            return line.split(":", 1)[1].strip()
    return ""


def hub_get(path: str) -> dict:
    url = "https://hub.docker.com/v2/" + path
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.load(r)


def hub_repo_path(name: str) -> str:
    """ubuntu → library/ubuntu, docker.io/x/y → x/y"""
    name = name.removeprefix("docker.io/")
    return name if "/" in name else f"library/{name}"


def ago(iso: str) -> str:
    if not iso:
        return ""
    try:
        t = time.mktime(time.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
    except ValueError:
        return iso[:10]
    d = (time.time() - t) / 86400
    if d < 1:
        return "today"
    if d < 30:
        return f"{int(d)}d ago"
    if d < 365:
        return f"{int(d / 30)}mo ago"
    return f"{int(d / 365)}y ago"


class ImagePicker(ModalScreen[Optional[str]]):
    """로컬 이미지 / Docker Hub 검색 / 이름으로 받기 중에서 이미지를 고른다."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, current: str = "", pull_mode: bool = False) -> None:
        super().__init__()
        self.current = current
        self.pull_mode = pull_mode     # True면 고른 이미지를 받는 용도 (이미지 탭의 p)
        self.repo = ""
        self.tags: list[dict] = []
        self.local: list[dict] = []
        self.busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Static(Text("Pull image" if self.pull_mode else "Select image", style="bold"))
            with TabbedContent(initial="pk-hub" if self.pull_mode else "pk-local", id="pk-tabs"):
                with TabPane("Local images", id="pk-local"):
                    yield Input(placeholder="Filter (part of the name)", id="pk-local-filter")
                    yield DataTable(id="pk-local-table", cursor_type="row", zebra_stripes=True)
                with TabPane("Docker Hub", id="pk-hub"):
                    with Horizontal(id="pk-hub-body"):
                        with Vertical():
                            yield Input(placeholder="Search and press Enter (empty = my repositories)", id="pk-query")
                            yield DataTable(id="pk-repos", cursor_type="row", zebra_stripes=True)
                        with Vertical():
                            yield Input(placeholder="Tag filter (e.g. cuda12 runtime)", id="pk-tag-filter")
                            yield DataTable(id="pk-tags", cursor_type="row", zebra_stripes=True)
                with TabPane("Pull by name", id="pk-name"):
                    yield Label("Image name (e.g. ubuntu:22.04, pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime)")
                    yield Input(self.current, id="pk-name-input")
                    with Horizontal(classes="buttons"):
                        yield Button("Pull now", variant="primary", id="pk-pull")
                        if not self.pull_mode:
                            yield Button("Select only (pull on run)", id="pk-choose")
                    yield RichLog(id="pk-log", wrap=True, markup=False)
            yield Static(id="pk-status")

    def on_mount(self) -> None:
        self.query_one("#pk-local-table", DataTable).add_columns("Image", "Size", "Created")
        self.query_one("#pk-repos", DataTable).add_columns("Repository", "★", "Description")
        self.query_one("#pk-tags", DataTable).add_columns("Tag", "Size", "Updated")
        self.load_local()
        self.search("")

    def status(self, msg: str, style: str = "dim") -> None:
        self.query_one("#pk-status", Static).update(Text(msg, style=style))

    def action_cancel(self) -> None:
        if not self.busy:
            self.dismiss(None)

    # ---- 로컬
    @work(thread=True)
    def load_local(self) -> None:
        try:
            imgs = docker_json_lines("images", "--format", "{{json .}}")
        except DockerError as e:
            self.app.call_from_thread(self.status, str(e), "red")
            return
        self.local = [i for i in imgs if i["Repository"] != "<none>"]
        self.app.call_from_thread(self.fill_local)

    @on(Input.Changed, "#pk-local-filter")
    def fill_local(self) -> None:
        q = self.query_one("#pk-local-filter", Input).value.strip().lower()
        t = self.query_one("#pk-local-table", DataTable)
        t.clear()
        seen = set()
        for i in self.local:
            ref = f"{i['Repository']}:{i['Tag']}"
            if q in ref.lower() and ref not in seen:
                seen.add(ref)
                t.add_row(Text(ref), i.get("Size", ""), i.get("CreatedSince", ""), key=ref)
        if not self.local:
            self.status("No local images. Try the Docker Hub tab.")

    @on(DataTable.RowSelected, "#pk-local-table")
    def local_selected(self, e: DataTable.RowSelected) -> None:
        self.dismiss(e.row_key.value)

    # ---- Docker Hub
    @on(Input.Submitted, "#pk-query")
    def query_submitted(self, e: Input.Submitted) -> None:
        self.search(e.value.strip())

    @work(thread=True, exclusive=True, group="hub-search")
    def search(self, q: str) -> None:
        call = self.app.call_from_thread
        rows = []
        try:
            if q:
                call(self.status, f"Searching '{q}'...")
                for r in docker_json_lines("search", "--limit", "50", "--no-trunc", "--format", "{{json .}}", q):
                    off = r.get("IsOfficial") in (True, "true", "[OK]")
                    rows.append((r["Name"], int(r.get("StarCount") or 0), r.get("Description", ""), off))
            else:
                user = hub_username()
                if not user:
                    call(self.status, "Type a search term and press Enter.")
                    return
                call(self.status, f"Loading repositories of {user}...")
                for r in hub_get(f"repositories/{user}/?page_size=100").get("results", []):
                    rows.append((f"{user}/{r['name']}", r.get("star_count", 0), r.get("description") or "", False))
        except Exception as e:   # 네트워크/파싱 오류
            call(self.status, f"Search failed: {e}", "red")
            return
        call(self.fill_repos, rows, q)

    def fill_repos(self, rows: list, q: str) -> None:
        t = self.query_one("#pk-repos", DataTable)
        t.clear()
        rows.sort(key=lambda r: (not r[3], -r[1]))
        for name, stars, desc, off in rows:
            label = Text(name, style="bold" if off else "")
            if off:
                label.append(" official", style="green")
            t.add_row(label, str(stars), Text(desc[:80], style="dim"), key=name)
        what = f"Results for '{q}'" if q else "My repositories"
        self.status(f"{what}: {len(rows)}. Pick a repository and press Enter, then pick a tag on the right")
        if len(rows) == 1:
            self.load_tags(rows[0][0])

    @on(DataTable.RowSelected, "#pk-repos")
    def repo_selected(self, e: DataTable.RowSelected) -> None:
        self.load_tags(e.row_key.value)

    @work(thread=True, exclusive=True, group="hub-tags")
    def load_tags(self, repo: str) -> None:
        call = self.app.call_from_thread
        call(self.status, f"Loading tags of {repo}...")
        try:
            data = hub_get(f"repositories/{hub_repo_path(repo)}/tags?page_size=100&ordering=last_updated")
        except Exception as e:
            call(self.status, f"Failed to load tags: {e}", "red")
            return
        self.repo, self.tags = repo, data.get("results", [])
        call(self.fill_tags)
        more = f" (latest 100 of {data.get('count')})" if data.get("count", 0) > 100 else ""
        call(self.status, f"{repo}: {len(self.tags)} tags{more}. Pick a tag and press Enter")
        call(self.query_one("#pk-tags", DataTable).focus)

    @on(Input.Changed, "#pk-tag-filter")
    def fill_tags(self) -> None:
        words = self.query_one("#pk-tag-filter", Input).value.lower().split()
        t = self.query_one("#pk-tags", DataTable)
        t.clear()
        seen = set()
        for tag in self.tags:
            name = tag["name"]
            if all(w in name.lower() for w in words) and name not in seen:
                seen.add(name)
                size = fmt_bytes(tag["full_size"]) if tag.get("full_size") else ""
                t.add_row(Text(name), size, ago(tag.get("last_updated", "")), key=name)

    @on(DataTable.RowSelected, "#pk-tags")
    def tag_selected(self, e: DataTable.RowSelected) -> None:
        self.dismiss(f"{self.repo}:{e.row_key.value}")

    # ---- 이름으로 받기
    @on(Button.Pressed, "#pk-choose")
    @on(Input.Submitted, "#pk-name-input")
    def choose(self) -> None:
        ref = self.query_one("#pk-name-input", Input).value.strip()
        if not ref:
            return
        if self.pull_mode:
            self.pull_now()
        else:
            self.dismiss(ref)

    @on(Button.Pressed, "#pk-pull")
    def pull_now(self) -> None:
        ref = self.query_one("#pk-name-input", Input).value.strip()
        if ref and not self.busy:
            self.busy = True
            self.query_one("#pk-pull", Button).disabled = True
            self.do_pull(ref, self.query_one("#pk-log", RichLog), self.query_one("#pk-pull", Button))

    @work(thread=True)
    def do_pull(self, ref: str, log: RichLog, button: Button) -> None:
        call = self.app.call_from_thread
        call(self.status, f"Pulling {ref}... (esc is disabled until it finishes)")
        rc = stream_cmd(["docker", "pull", ref], lambda line: call(log.write, Text(line)))
        self.busy = False
        if rc == 0:
            call(self.dismiss, ref)
        else:
            call(self.status, f"Pull failed (exit {rc}). Check the name and tag.", "red")
            call(setattr, button, "disabled", False)


def stream_cmd(cmd: list[str], on_line) -> int:
    """명령 출력을 줄 단위로 넘긴다. docker pull의 같은 레이어 진행 줄은 몇 개만 보낸다."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = {}
    for line in p.stdout:
        line = line.rstrip()
        key = line.split(":")[0] if ": Downloading" in line or ": Extracting" in line else None
        if key:
            now = time.time()
            if now - last.get(key, 0) < 2:
                continue
            last[key] = now
        on_line(line)
    return p.wait()


# ---------------------------------------------------------------- 실행 폼

def option_help() -> dict[str, str]:
    rl = is_rootless()
    return {
        "dood": "Use the host Docker from inside the container with the docker command. "
                "Needed for execution-based evaluation (SWE-bench, etc.).",
        "restart": "Start the container again automatically after a server reboot or Docker restart.",
        "shm_on": "Size of /dev/shm used by PyTorch DataLoader (num_workers>0). The default 64MB is small and "
                  "often causes bus errors. Ignored when sharing host IPC.",
        "cpu_on": "Limit the number of CPU cores the container can use (--cpus).",
        "mem_on": "Upper bound of memory the container can use (--memory). Processes inside are OOM-killed "
                  "above it. Shared memory (/dev/shm) usage counts toward it.",
        "ipc_host": "Share shared memory with the host. Removes the shm size limit and helps multi-GPU "
                    "communication (NCCL). The shared memory size setting is ignored when on.",
        "ulimits": "NVIDIA recommended settings: stack 64MB" + (
            ". Rootless Docker cannot use memlock=-1, so only stack is applied." if rl else ", unlimited memlock."),
        "ptrace": "Let debuggers and profilers such as gdb and py-spy dump/top attach to running processes.",
    }


class RunForm(Screen[None]):
    """새 컨테이너 만들기 / 기존 컨테이너 다시 만들기."""

    BINDINGS = [Binding("ctrl+r", "submit", "Run"), Binding("escape", "app.pop_screen", "Cancel")]

    def __init__(self, spec: RunSpec, recreate: dict | None = None, warns: list[str] | None = None) -> None:
        super().__init__()
        self.spec = spec
        self.recreate = recreate     # 다시 만들 기존 컨테이너의 inspect 결과
        self.warns = warns or []

    def compose(self) -> ComposeResult:
        s = self.spec
        images = self.app.image_names
        title = f"Recreate: {s.name}" if self.recreate else "New container"
        yield Header()
        with VerticalScroll(id="form"):
            yield Static(Text(title, style="bold"))
            for w in self.warns:
                yield Static(Text(f"⚠ {w}", style="yellow"))
            yield Label("Name")
            yield Input(s.name, id="name", placeholder="Empty = random name from Docker")
            yield Label("Image")
            with Horizontal(classes="image-row"):
                yield Input(s.image, id="image", suggester=SuggestFromList(images, case_sensitive=False))
                yield Button("Find / Pull", id="find-image")
            yield Static(id="image-info")
            if self.recreate:
                yield Checkbox("Commit the current state and continue from it (keeps what was installed inside)",
                               value=True, id="commit")
            h = option_help()
            caps = docker_caps()
            yield Label("Resources")
            with Vertical(classes="res-box"):
                yield Label("GPU  (turn all off to run without GPU)", classes="pane-title")
                if self.app.gpus:
                    sel = {g[0] for g in self.app.gpus} if s.gpus == "all" else set(filter(None, s.gpus.split(",")))
                    with Horizontal(classes="gpu-row"):
                        for idx, gname, mem in self.app.gpus:
                            yield GpuToggle(f"GPU {idx} · {gname} · {mem}GiB", value=idx in sel, id=f"gpu-{idx}")
                else:
                    yield Input(s.gpus, id="gpus", placeholder="no nvidia-smi: all / 0,1 / empty")
                if caps["cpu_limit"]:
                    with Horizontal(classes="res-row"):
                        yield Checkbox("CPU limit", value=bool(s.cpus), id="cpu_on", tooltip=h["cpu_on"])
                        yield Input(s.cpus or str(min(16, caps["ncpu"])), id="cpus")
                        yield Static(Text(f"max {caps['ncpu']} cores", style="dim"), classes="res-note")
                if caps["mem_limit"]:
                    with Horizontal(classes="res-row"):
                        yield Checkbox("Memory limit", value=bool(s.memory), id="mem_on", tooltip=h["mem_on"])
                        yield Input(s.memory or "64g", id="memory")
                        yield Static(Text(f"max {caps['mem_gib']}GiB", style="dim"), classes="res-note")
                with Horizontal(classes="res-row", id="shm-row"):
                    yield Checkbox("Shared memory (/dev/shm)", value=bool(s.shm), id="shm_on", tooltip=h["shm_on"])
                    yield Input(s.shm or "16g", id="shm")
                    yield Static(Text("For DataLoader. Off = Docker default 64m", style="dim"), classes="res-note")
            yield Label("Timezone")
            known = s.tz in (TZ_SERVER, TZ_IMAGE, *COMMON_ZONES)
            with Horizontal(classes="tz-row"):
                yield Select(
                    [(f"Server ({self.app.server_tz})" if self.app.server_tz else "Server", TZ_SERVER)]
                    + [(z, z) for z in COMMON_ZONES]
                    + [("Image default (don't change)", TZ_IMAGE), ("Other…", TZ_OTHER)],
                    value=s.tz if known else TZ_OTHER, allow_blank=False, id="tz")
                yield Input("" if known else s.tz, id="tz-other", placeholder="e.g. Europe/London",
                            suggester=SuggestFromList(all_zones(), case_sensitive=False))
                yield Static(Text("Uses the server's zone files, so it works even if the image has no tzdata.",
                                  style="dim"), classes="res-note")
            yield Label("Mounts")
            if notes := self.app.pathmap.summary():
                yield Static(Text("Auto-translated to Docker host paths: " + ", ".join(notes)
                                  + "   (paths inside this environment, like /tmp, cannot be mounted)", style="dim"))
            yield MountEditor(s.mounts, self.tree_root(s))
            yield Label("Ports")
            yield PortEditor(s.ports)
            yield Label("Environment variables  (one per line: KEY=VALUE)")
            yield TextArea("\n".join(s.env), id="env")
            yield Label("Options  (focus an item to see its description)")
            rl = is_rootless()
            # 중요한 순서. ulimits는 GPU가 있을 때, 호스트 IPC는 GPU 2개 이상일 때만 보인다 (update_preview)
            with Grid(id="opts"):
                yield Checkbox("Auto-restart (--restart unless-stopped)", value=s.restart, id="restart",
                               tooltip=h["restart"])
                if os.path.exists(host_socket()):
                    yield Checkbox("Use Docker inside the container (DooD)", value=s.dood, id="dood",
                                   tooltip=h["dood"])
                yield Checkbox("NVIDIA recommended ulimits" + (" (stack)" if rl else " (memlock, stack)"),
                               value=s.ulimits, id="ulimits", tooltip=h["ulimits"])
                yield Checkbox("Share host IPC (--ipc host) · multi-GPU", value=s.ipc_host, id="ipc_host",
                               tooltip=h["ipc_host"])
                yield Checkbox("Allow debuggers (gdb, py-spy)", value=s.ptrace, id="ptrace", tooltip=h["ptrace"])
            yield Static(id="opt-help")
            yield Label("Extra docker run args  (e.g. --device /dev/fuse --hostname dev)")
            yield Input(s.extra, id="extra")
            yield Label("Command  (empty = image default)")
            yield Input(s.command, id="command")
            yield Label("Command to run")
            yield Static(id="preview")
            yield Static(id="errors")
            with Horizontal(classes="buttons"):
                yield Button("Run (ctrl+r)", variant="primary", id="run")
                yield Button("Cancel (esc)", id="cancel")
        yield Footer()

    def on_mount(self) -> None:
        for ta in self.query(TextArea):
            ta.show_line_numbers = False
        self.update_preview()
        self.image_info()

    @staticmethod
    def tree_root(s: RunSpec) -> str:
        """트리 시작 위치: 첫 호스트 마운트의 상위 폴더, 없으면 ~/data 또는 홈."""
        for m in s.mounts:
            parent = os.path.dirname(m.split(":")[0].rstrip("/"))
            if m.startswith("/") and os.path.isdir(parent):
                return parent
        home = os.path.expanduser("~")
        return os.path.join(home, "data") if os.path.isdir(os.path.join(home, "data")) else home

    def lines(self, wid: str) -> list[str]:
        return [x.strip() for x in self.query_one(f"#{wid}", TextArea).text.splitlines() if x.strip()]

    def val(self, wid: str) -> str:
        return self.query_one(f"#{wid}", Input).value.strip()

    def commit_on(self) -> bool:
        return bool(self.recreate) and self.query_one("#commit", Checkbox).value

    def snapshot_tag(self) -> str:
        if not hasattr(self, "_snap"):
            self._snap = f"{self.spec.name or 'container'}-snapshot:{time.strftime('%Y%m%d-%H%M%S')}".lower()
        return self._snap

    def has(self, selector: str) -> bool:
        return bool(self.query(selector))

    def shown_cb(self, wid: str) -> bool:
        """화면에 보이는(사용 가능한) 체크박스만 값으로 친다."""
        found = self.query(f"#{wid}")
        return bool(found) and found.first().display and found.first().value

    def tz_value(self) -> str:
        v = self.query_one("#tz", Select).value
        return self.val("tz-other") if v == TZ_OTHER else v

    def gpu_count(self) -> int:
        g = self.selected_gpus()
        if not g:
            return 0
        if g == "all":
            return len(self.app.gpus) or 2
        return len(g.split(","))

    def cb(self, wid: str) -> bool:
        return self.query_one(f"#{wid}", Checkbox).value

    def selected_gpus(self) -> str:
        if not self.app.gpus:
            return self.val("gpus").replace(" ", "")
        chosen = [idx for idx, _, _ in self.app.gpus if self.cb(f"gpu-{idx}")]
        return "all" if len(chosen) == len(self.app.gpus) else ",".join(chosen)

    def collect(self) -> RunSpec:
        return RunSpec(
            name=self.val("name"),
            image=self.snapshot_tag() if self.commit_on() else self.val("image"),
            gpus=self.selected_gpus(),
            mounts=self.query_one(MountEditor).mount_strings(),
            ports=self.query_one(PortEditor).port_strings(),
            env=self.lines("env"),
            dood=self.shown_cb("dood"),
            restart=self.cb("restart"),
            shm=self.val("shm") if self.cb("shm_on") and not self.shown_cb("ipc_host") else "",
            cpus=self.val("cpus") if self.has("#cpu_on") and self.cb("cpu_on") else "",
            memory=self.val("memory") if self.has("#mem_on") and self.cb("mem_on") else "",
            ipc_host=self.shown_cb("ipc_host"),
            ulimits=self.shown_cb("ulimits"),
            ptrace=self.cb("ptrace"),
            tz=self.tz_value(),
            extra=self.val("extra"),
            command=self.val("command"),
        )

    def on_descendant_focus(self, event) -> None:
        wid = event.widget.id or ""
        help_text = option_help().get(wid)
        self.query_one("#opt-help", Static).update(Text(help_text or "", style="italic"))

    @on(Input.Changed)
    @on(Select.Changed)
    @on(TextArea.Changed)
    @on(MountEditor.Changed)
    @on(PortEditor.Changed)
    @on(Checkbox.Changed)
    def update_preview(self) -> None:
        # 보이는 옵션을 먼저 정한 뒤 값을 모은다
        n = self.gpu_count()
        self.query_one("#ulimits", Checkbox).display = n >= 1
        self.query_one("#ipc_host", Checkbox).display = n >= 2
        self.query_one("#shm-row").display = not self.shown_cb("ipc_host")
        self.query_one("#tz-other").display = self.query_one("#tz", Select).value == TZ_OTHER
        spec = self.collect()
        if self.recreate:
            self.query_one("#image", Input).disabled = self.commit_on()
            self.query_one("#find-image", Button).disabled = self.commit_on()
        self.query_one("#shm", Input).disabled = not self.cb("shm_on")
        if self.has("#cpu_on"):
            self.query_one("#cpus", Input).disabled = not self.cb("cpu_on")
        if self.has("#mem_on"):
            self.query_one("#memory", Input).disabled = not self.cb("mem_on")
        try:
            cmd = " \\\n  ".join(_group_args(spec.args(self.preview_argv(spec))))
        except ValueError as e:
            cmd = f"(argument error: {e})"
        self.query_one("#preview", Static).update(Text(cmd, style="green"))
        errs = self.errors(spec)
        self.query_one("#errors", Static).update(Text("\n".join(errs), style="red"))

    def preview_argv(self, spec: RunSpec) -> Optional[list]:
        if not spec.hook:
            return None
        key = (spec.image, spec.command, tuple(spec.base_argv or ()))
        cache = self.__dict__.setdefault("_argv_cache", {})
        if key not in cache:
            argv = resolve_argv(spec)
            # 이미지가 아직 없으면 실행할 때 결정된다
            cache[key] = argv if argv is not None else ["<image default command>"]
        return cache[key]

    def errors(self, spec: RunSpec) -> list[str]:
        errs = spec.validate()
        old = self.recreate["Name"].lstrip("/") if self.recreate else None
        if spec.name and spec.name != old and spec.name in self.app.containers:
            errs.append(f"A container with this name already exists: {spec.name}")
        return errs

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#find-image")
    def find_image(self) -> None:
        def chosen(ref: str | None) -> None:
            if ref:
                self.query_one("#image", Input).value = ref
                self.app.action_refresh_all()
        self.app.push_screen(ImagePicker(self.val("image")), chosen)

    @on(Input.Changed, "#image")
    def image_info(self) -> None:
        ref = self.val("image")
        info = self.query_one("#image-info", Static)
        if not ref or self.commit_on():
            info.update("")
        elif ref in self.app.image_names or image_exists(ref):
            info.update(Text("✓ available locally", style="green"))
        else:
            info.update(Text("Not available locally: it will be pulled from Docker Hub on run (with progress)", style="yellow"))

    @on(Button.Pressed, "#run")
    def action_submit(self) -> None:
        spec = self.collect()
        errs = self.errors(spec)
        if errs:
            self.notify("\n".join(errs), severity="error")
            return
        if self.recreate:
            old = self.recreate["Name"].lstrip("/")
            steps = []
            if self.commit_on():
                steps.append(f"1. Commit {old} to image {spec.image}")
            steps += [f"{len(steps) + 1}. Stop {old} and rename it to {old}-old-<time>",
                      f"{len(steps) + 2}. Run {spec.name or '(random name)'} with the new settings",
                      "   If it fails, the original container is restored.",
                      "   If it succeeds, the original container is kept, stopped."]
            self.app.push_screen(Confirm("Recreate the container?", "\n".join(steps)),
                                 lambda ok: ok and self.app.start_recreate(self.recreate, spec, self.commit_on()))
        else:
            self.app.start_run(spec)


def _group_args(args: list[str]) -> list[str]:
    """미리보기용: 옵션과 값을 한 줄로, 이미지와 명령을 한 줄로 묶는다."""
    takes_value = {"--name", "--gpus", "--restart", "-v", "-p", "-e", "--network", "--ipc", "--user",
                   "--ulimit", "--cap-add", "--mount", "--entrypoint", "--label"}
    q = ["'<tdock startup hook>'" if a == svc.HOOK else shlex.quote(a) for a in args]
    out, i = [" ".join(q[:3])], 3
    while i < len(args):
        if args[i] in takes_value and i + 1 < len(args):
            out.append(f"{q[i]} {q[i + 1]}")
            i += 2
        elif args[i].startswith("-"):
            out.append(q[i])
            i += 1
        else:
            out.append(" ".join(q[i:]))   # 이미지 + 명령
            break
    return out


# ---------------------------------------------------------------- 서비스 (자동 시작)

def host_public_keys() -> str:
    """SSH 공개키 기본값: 호스트의 authorized_keys, 없으면 ~/.ssh/*.pub"""
    ssh = Path.home() / ".ssh"
    for f in [ssh / "authorized_keys", *sorted(ssh.glob("*.pub"))]:
        try:
            text = f.read_text().strip()
        except OSError:
            continue
        if text:
            return text
    return ""


def host_port_for(c: dict, cport: int) -> Optional[str]:
    for p in ports_of(c):
        hostpart, _, cp = p.rpartition(":")
        if cp.split("/")[0] == str(cport):
            return hostpart.rpartition(":")[2]
    return None


class ServiceLog(ModalScreen[None]):
    BINDINGS = [Binding("escape,q", "dismiss", "Close"), Binding("r", "reload", "Refresh")]

    def __init__(self, container: str, sid: str) -> None:
        super().__init__()
        self.container, self.sid = container, sid

    def compose(self) -> ComposeResult:
        with Vertical(id="tasklog"):
            yield Static(Text(f"Service log: {self.sid} in {self.container}  "
                              f"({svc.LOG_DIR}/{self.sid}.log, r refresh, esc close)", style="bold"))
            yield RichLog(wrap=True, markup=False, max_lines=5000)

    def on_mount(self) -> None:
        self.action_reload()

    def action_reload(self) -> None:
        log = self.query_one(RichLog)
        log.clear()
        log.write(Text(svc.read_log(self.container, self.sid)))


class ServiceRow(Horizontal):
    """서비스 한 줄: 이름 | 설치 여부 | 상태 | Start with container | Restart if it dies | Start/Stop | Log"""

    def __init__(self, sid: str, label: str, entry: dict, installed: Optional[bool], custom: bool = False) -> None:
        super().__init__(classes="svc-row", id=f"row-{sid}")
        self.sid, self.label, self.entry, self.inst, self.custom = sid, label, entry, installed, custom

    def compose(self) -> ComposeResult:
        usable = self.inst is not False
        if self.custom:
            yield Input(self.entry.get("command", ""), id=f"cmdtext-{self.sid}", classes="svc-cmd",
                        placeholder="command, run in the foreground (e.g. python serve.py)")
        else:
            yield Static(Text(self.label, style="bold"), classes="svc-name")
        if not self.custom:
            inst = {True: Text("✓ installed", style="green"), False: Text("✗ not installed", style="dim"),
                    None: Text("?", style="dim")}[self.inst]
            yield Static(inst, classes="svc-inst")
        yield Static(Text("-", style="dim"), id=f"st-{self.sid}", classes="svc-status")
        auto = bool(self.entry.get("autostart")) and usable
        yield Checkbox("Start with container", value=auto, id=f"auto-{self.sid}", disabled=not usable)
        yield Checkbox("Restart if it dies", value=auto and bool(self.entry.get("keep")),
                       id=f"keep-{self.sid}", disabled=not auto)
        yield Button("Start", id=f"run-{self.sid}", classes="svc-btn", disabled=not usable)
        yield Button("Log", id=f"log-{self.sid}", classes="svc-btn")
        if self.custom:
            yield Button("✕", id=f"del-{self.sid}", variant="error", classes="svc-btn")


class ServicesScreen(Screen[None]):
    BINDINGS = [Binding("ctrl+s", "apply", "Apply"), Binding("escape", "close", "Close")]

    def __init__(self, c: dict, cfg: dict, installed: dict, statuses: dict) -> None:
        super().__init__()
        self.c = c
        self.cname = c["Name"].lstrip("/")
        self.saved = json.loads(json.dumps(cfg))     # 컨테이너에 저장된 설정
        self.cfg = cfg
        self.inst = installed
        self.statuses = statuses
        self.running = bool(c["State"].get("Running"))
        self.hook = has_hook(c)
        hc = c.get("HostConfig") or {}
        self.mem_total = docker_caps()["mem_gib"] << 30
        self.cur_mem = hc.get("Memory") or 0
        self.cur_restart = (hc.get("RestartPolicy") or {}).get("Name") in ("unless-stopped", "always")
        self.cur_cpus = f"{hc['NanoCpus'] / 1e9:g}" if hc.get("NanoCpus") else ""
        self.auto_remove = bool(hc.get("AutoRemove"))   # --rm 컨테이너는 재시작 정책을 바꿀 수 없다

    # ---- 화면
    def compose(self) -> ComposeResult:
        s = self.cfg["services"]
        caps = docker_caps()
        yield Header()
        with VerticalScroll(id="form"):
            yield Static(Text(f"Services: {self.cname}", style="bold"))
            if not self.hook:
                with Horizontal(classes="svc-banner"):
                    yield Static(Text("⚠ This container was not created by tdock, so it cannot start services "
                                      "automatically. Recreate it once to add the startup hook "
                                      "(what you installed inside is kept).", style="yellow"))
                    yield Button("Recreate…", id="svc-recreate-hook", variant="warning")
            if not self.running:
                yield Static(Text("The container is stopped: settings are saved and take effect at its next start.",
                                  style="yellow"))
            yield Label("Services")
            yield Static(Text("Service                Installed        Status       ", style="dim"))
            for sid, label, _, _ in svc.BUILTIN:
                yield ServiceRow(sid, label, s[sid], self.inst.get(sid) if self.running else None)
            with Vertical(id="custom-rows"):
                for cm in self.cfg["custom"]:
                    yield ServiceRow(cm["id"], cm.get("name") or cm["id"], cm, None, custom=True)
            yield Button("+ Add command", id="add-cmd")

            yield Label("SSH authorized keys  (one public key per line)")
            yield TextArea(s["ssh"].get("authorized_keys") or host_public_keys(), id="ssh-keys")
            hp = host_port_for(self.c, 22)
            yield Static(Text(f"Connect: ssh -p {hp} root@<server>" if hp else
                              "⚠ No port mapping for container port 22: SSH is reachable only from other "
                              "containers. Add one with Recreate (e.g. 2222:22).", style="dim" if hp else "yellow"))
            yield Label("VS Code tunnel name")
            yield Input(s["tunnel"].get("name", ""), id="tunnel-name", placeholder="name shown in VS Code (optional)")
            yield Static(Text("First time only: open a shell (e) and run  code tunnel user login --provider github",
                              style="dim"))
            yield Label("Jupyter  (container port / token)")
            with Horizontal(classes="tz-row"):
                yield Input(str(s["jupyter"].get("port") or 8888), id="jup-port", type="integer")
                yield Input(s["jupyter"].get("token", ""), id="jup-token", placeholder="token")
            jp = host_port_for(self.c, int(s["jupyter"].get("port") or 8888))
            yield Static(Text(f"Open: http://<server>:{jp}/?token=<token>" if jp else
                              "⚠ No port mapping for the Jupyter port. Add one with Recreate.",
                              style="dim" if jp else "yellow"), id="jup-note")

            yield Label("Container settings  (applied immediately, no restart)")
            with Vertical(classes="res-box"):
                if not self.auto_remove:
                    yield Checkbox("Auto-restart the container (--restart unless-stopped)", value=self.cur_restart,
                                   id="live-restart")
                if caps["mem_limit"]:
                    limited = 0 < self.cur_mem < self.mem_total
                    with Horizontal(classes="res-row"):
                        yield Checkbox("Memory limit", value=limited, id="live-mem-on")
                        yield Input(human_size(self.cur_mem) if limited else "64g", id="live-mem")
                        yield Static(Text(f"max {caps['mem_gib']}GiB", style="dim"), classes="res-note")
                if caps["cpu_limit"]:
                    with Horizontal(classes="res-row"):
                        yield Checkbox("CPU limit", value=bool(self.cur_cpus), id="live-cpu-on")
                        yield Input(self.cur_cpus or "16", id="live-cpus")
                        yield Static(Text(f"max {caps['ncpu']} cores", style="dim"), classes="res-note")
            with Horizontal(classes="buttons"):
                yield Button("Recreate…  (GPUs, ports, mounts, image, options)", id="svc-recreate")
            yield Static(id="svc-errors")
            with Horizontal(classes="buttons"):
                yield Button("Apply (ctrl+s)", variant="primary", id="svc-apply")
                yield Button("Save & restart container", variant="warning", id="svc-restart")
                yield Button("Close (esc)", id="svc-close")
        yield Footer()

    def on_mount(self) -> None:
        for ta in self.query(TextArea):
            ta.show_line_numbers = False
        self.show_status(self.statuses)
        if self.running:
            self.set_interval(3, self.poll_status)

    # ---- 상태
    @work(thread=True, exclusive=True, group="svc-status")
    def poll_status(self) -> None:
        st = svc.status(self.cname)
        self.app.call_from_thread(self.show_status, st)

    def show_status(self, st: dict) -> None:
        self.statuses = st
        colors = {"running": "green", "restarting": "yellow", "stopped": "dim"}
        for row in self.query(ServiceRow):
            state = st.get(row.sid, "-") if self.running else "-"
            label = {"running": "● running", "restarting": "↻ restarting", "stopped": "○ stopped"}.get(state, "-")
            row.query_one(f"#st-{row.sid}", Static).update(Text(label, style=colors.get(state, "dim")))
            btn = row.query_one(f"#run-{row.sid}", Button)
            btn.label = "Stop" if state in ("running", "restarting") else "Start"

    # ---- 입력
    @on(Checkbox.Changed)
    def toggled(self, e: Checkbox.Changed) -> None:
        wid = e.checkbox.id or ""
        if wid.startswith("auto-"):
            keep = self.query_one(f"#keep-{wid[5:]}", Checkbox)
            keep.disabled = not e.value
            if not e.value:
                keep.value = False
        if wid == "live-mem-on":
            self.query_one("#live-mem", Input).disabled = not e.value

    def collect(self) -> dict:
        cfg = json.loads(json.dumps(self.cfg))
        for row in self.query(ServiceRow):
            entry = svc.service_entry(cfg, row.sid) if row.sid in [x for x, *_ in svc.BUILTIN] else None
            if entry is None:
                entry = next((x for x in cfg["custom"] if x["id"] == row.sid), None)
                if entry is None:
                    continue
                entry["command"] = self.query_one(f"#cmdtext-{row.sid}", Input).value.strip()
            entry["autostart"] = self.query_one(f"#auto-{row.sid}", Checkbox).value
            entry["keep"] = entry["autostart"] and self.query_one(f"#keep-{row.sid}", Checkbox).value
        live_ids = {r.sid for r in self.query(ServiceRow)}
        cfg["custom"] = [x for x in cfg["custom"] if x["id"] in live_ids]
        s = cfg["services"]
        s["ssh"]["authorized_keys"] = self.query_one("#ssh-keys", TextArea).text.strip()
        s["tunnel"]["name"] = self.query_one("#tunnel-name", Input).value.strip()
        s["jupyter"]["port"] = int(self.query_one("#jup-port", Input).value or 8888)
        s["jupyter"]["token"] = self.query_one("#jup-token", Input).value.strip()
        return cfg

    def validate(self, cfg: dict) -> list[str]:
        errs = []
        for cm in cfg["custom"]:
            if msg := svc.valid_custom_command(cm["command"]):
                errs.append(f"{cm['id']}: {msg}")
        if cfg["services"]["ssh"]["autostart"] and not cfg["services"]["ssh"]["authorized_keys"]:
            errs.append("SSH: add at least one public key, otherwise nobody can log in.")
        if self.has("#live-mem-on") and self.query_one("#live-mem-on", Checkbox).value:
            mem = size_bytes(self.query_one("#live-mem", Input).value)
            if not mem or mem < 6 << 20 or mem > self.mem_total:
                errs.append(f"Memory must look like 64g and be between 6m and {docker_caps()['mem_gib']}GiB.")
        return errs

    def has(self, selector: str) -> bool:
        return bool(self.query(selector))

    def live_updates(self) -> list[list[str]]:
        """바로 바꿀 수 있는 컨테이너 설정 (docker update)."""
        ups = []
        if self.has("#live-restart"):
            restart = self.query_one("#live-restart", Checkbox).value
            if restart != self.cur_restart:
                ups.append(["--restart", "unless-stopped" if restart else "no"])
        if self.has("#live-mem-on"):
            on = self.query_one("#live-mem-on", Checkbox).value
            want = size_bytes(self.query_one("#live-mem", Input).value) if on else self.mem_total
            limited = 0 < self.cur_mem < self.mem_total
            if (on and want != self.cur_mem) or (not on and limited):
                # swap 한도를 함께 넘겨야 올리고 내리기가 된다. 해제는 서버 전체 메모리로 올리는 것으로 대신
                ups.append(["--memory", str(want), "--memory-swap", str(want * 2)])
        if self.has("#live-cpu-on"):
            on = self.query_one("#live-cpu-on", Checkbox).value
            cpus = self.query_one("#live-cpus", Input).value.strip() if on else "0"
            if cpus != (self.cur_cpus or "0"):
                ups.append(["--cpus", cpus])
        return ups

    # ---- 버튼
    @on(Button.Pressed)
    def pressed(self, e: Button.Pressed) -> None:
        bid = e.button.id or ""
        if bid == "svc-apply":
            self.action_apply()
        elif bid == "svc-close":
            self.action_close()
        elif bid == "svc-restart":
            self.save_and_restart()
        elif bid in ("svc-recreate", "svc-recreate-hook"):
            self.app.pop_screen()
            self.app.open_recreate(self.cname)
        elif bid == "add-cmd":
            cfg = self.collect()
            sid = svc.new_custom_id(cfg)
            entry = {"id": sid, "name": sid, "command": "", "autostart": False, "keep": False}
            self.cfg["custom"].append(entry)
            row = ServiceRow(sid, sid, entry, None, custom=True)
            self.query_one("#custom-rows").mount(row)
            self.call_after_refresh(lambda: self.query_one(f"#cmdtext-{sid}", Input).focus())
        elif bid.startswith("del-"):
            sid = bid[4:]
            self.cfg["custom"] = [x for x in self.cfg["custom"] if x["id"] != sid]
            self.query_one(f"#row-{sid}").remove()
        elif bid.startswith("log-"):
            if not self.running:
                self.notify("The container is stopped.", severity="warning")
            else:
                self.app.push_screen(ServiceLog(self.cname, bid[4:]))
        elif bid.startswith("run-"):
            self.start_stop(bid[4:], str(e.button.label))

    def start_stop(self, sid: str, label: str) -> None:
        if not self.running:
            self.notify("Start the container first.", severity="warning")
            return
        if sid not in svc._ids(self.saved):
            self.notify("Apply first, then start it.", severity="warning")
            return
        self.run_script("stop" if label == "Stop" else "start", sid)

    @work(thread=True)
    def run_script(self, action: str, sid: str) -> None:
        try:
            svc.exec_script(self.cname, action, sid)
        except RuntimeError as e:
            self.app.call_from_thread(self.notify, f"{action} {sid} failed: {e}", severity="error")
        time.sleep(1)
        self.app.call_from_thread(self.poll_status)

    def action_close(self) -> None:
        if json.dumps(self.collect(), sort_keys=True) != json.dumps(self.saved, sort_keys=True) or self.live_updates():
            self.app.push_screen(Confirm("Discard unsaved changes?"), lambda ok: ok and self.app.pop_screen())
        else:
            self.app.pop_screen()

    def action_apply(self) -> None:
        cfg = self.collect()
        errs = self.validate(cfg)
        self.query_one("#svc-errors", Static).update(Text("\n".join(errs), style="red"))
        if not errs:
            self.do_apply(cfg, self.live_updates(), restart=False)

    def save_and_restart(self) -> None:
        cfg = self.collect()
        errs = self.validate(cfg)
        self.query_one("#svc-errors", Static).update(Text("\n".join(errs), style="red"))
        if errs:
            return
        top = docker("top", self.cname, check=False) if self.running else ""
        procs = "\n".join(top.splitlines()[1:21]) if top else "(container is stopped)"
        self.app.push_screen(
            Confirm(f"Restart {self.cname}? Every process inside will be stopped.", "Running processes:\n" + procs),
            lambda ok: ok and self.do_apply(cfg, self.live_updates(), restart=True))

    @work(thread=True, exclusive=True, group="svc-apply")
    def do_apply(self, cfg: dict, updates: list, restart: bool) -> None:
        call = self.app.call_from_thread
        done = []
        try:
            svc.write_config(self.cname, cfg)
            done.append("settings saved")
            for args in updates:
                docker("update", *args, self.cname)
                done.append("updated " + " ".join(args[::2]).replace("--", ""))
            if restart:
                docker("restart", self.cname)
                done.append("container restarted")
            elif self.running:
                st = svc.status(self.cname)
                stop, start, again = svc.plan(self.saved, cfg)
                for sid in stop:
                    if st.get(sid) in ("running", "restarting"):
                        svc.exec_script(self.cname, "stop", sid)
                for sid in again:
                    if st.get(sid) in ("running", "restarting"):
                        svc.exec_script(self.cname, "stop", sid)
                        svc.exec_script(self.cname, "start", sid)
                for sid in start:
                    svc.exec_script(self.cname, "start", sid)
                changed = len(stop) + len(start) + len(again)
                done.append(f"{changed} service(s) updated" if changed else "no running service changed")
        except (DockerError, RuntimeError) as e:
            call(self.notify, f"Apply failed: {e}", severity="error", timeout=10)
            return
        self.saved = json.loads(json.dumps(cfg))
        self.cfg = cfg
        try:
            c = inspect(self.cname)[0]
            hc = c.get("HostConfig") or {}
            self.c, self.cur_mem = c, hc.get("Memory") or 0
            self.cur_restart = (hc.get("RestartPolicy") or {}).get("Name") in ("unless-stopped", "always")
            self.cur_cpus = f"{hc['NanoCpus'] / 1e9:g}" if hc.get("NanoCpus") else ""
        except (DockerError, IndexError):
            pass
        call(self.notify, "Applied: " + ", ".join(done))
        time.sleep(1)
        self.app.call_from_thread(self.poll_status)
        call(self.app.action_refresh_all)


# ---------------------------------------------------------------- 탭 위젯

class ContainerTable(DataTable):
    BINDINGS = [
        Binding("n", "app.new_container", "New"),
        Binding("m", "app.recreate", "Recreate"),
        Binding("v", "app.services", "Services"),
        Binding("s", "app.start_stop", "Start/Stop"),
        Binding("t", "app.restart_container", "Restart"),
        Binding("e", "app.exec_shell", "Shell"),
        Binding("l", "app.show_logs", "Logs"),
        Binding("d", "app.delete_container", "Delete"),
    ]


class ImageTable(DataTable):
    BINDINGS = [
        Binding("n", "app.run_image", "Run this image"),
        Binding("p", "app.pull_image", "pull"),
        Binding("d", "app.delete_image", "Delete"),
    ]


class CleanupPane(VerticalScroll):
    pass


# ---------------------------------------------------------------- 앱

class DockerTUI(App):
    TITLE = "tdock"
    CSS = """
    #containers-body { height: 1fr; }
    ContainerTable { width: 3fr; }
    #detail { width: 2fr; border-left: solid $primary; padding: 0 1; }
    #form { padding: 1 2; }
    #form Label { margin-top: 1; color: $text-muted; }
    #form TextArea { height: 5; }
    .image-row { height: auto; }
    .image-row Input { width: 1fr; }
    #image-info { height: auto; }
    ImagePicker { align: center middle; }
    #picker { width: 95%; height: 90%; border: thick $primary; background: $surface; padding: 0 1; }
    #pk-tabs { height: 1fr; }
    #pk-tabs TabPane { height: 1fr; padding: 0; }
    #pk-local-table, #pk-repos, #pk-tags { height: 1fr; }
    #pk-hub-body { height: 1fr; }
    #pk-hub-body > Vertical { width: 1fr; }
    #pk-log { height: 1fr; border: round $panel; }
    #pk-status { height: auto; }
    Checkbox > .toggle--button { color: $panel; background: $panel; }
    Checkbox.-on > .toggle--button { color: $background; background: $success; text-style: bold; }
    Checkbox.-on > .toggle--label { text-style: bold; }
    #opts Checkbox.-on, .res-row Checkbox.-on { background: $success 12%; }
    .gpu-row { height: auto; }
    GpuToggle { border: round $panel-lighten-2; background: $surface; margin-right: 2; padding: 0 1; }
    GpuToggle.-on { border: round $success; background: $success 25%; }
    GpuToggle:focus { border: round $accent; }
    GpuToggle.-on:focus { border: round $accent; background: $success 25%; }
    #opts { grid-size: 2; grid-gutter: 0 2; height: auto; }
    #opts > * { height: auto; }
    .svc-row { height: auto; }
    .svc-name { width: 18; padding-top: 1; }
    .svc-cmd { width: 35; }
    #custom-rows { height: auto; }
    .svc-inst { width: 17; padding-top: 1; }
    .svc-status { width: 15; padding-top: 1; }
    .svc-row Checkbox { width: auto; }
    .svc-btn { min-width: 8; width: auto; }
    .svc-banner { height: auto; }
    .svc-banner Static { width: 1fr; }
    #ssh-keys { height: 4; }
    .tz-row { height: auto; }
    .tz-row Select { width: 40; }
    .tz-row #tz-other { width: 30; }
    .res-box { height: auto; border: round $panel; padding: 0 1; }
    .res-row { height: auto; }
    .res-row Checkbox { width: 26; }
    .res-row Input { width: 12; }
    .res-note { width: 1fr; padding: 1 0 0 2; }
    #opt-help { color: $text-muted; min-height: 2; }
    .editor-box { height: auto; border: round $panel; }
    .editor { height: 20; }
    .editor.ports { height: 14; }
    .port-left { width: 44; padding-right: 2; }
    .port-left Button, .ports .pane-right Button { margin-top: 0; }
    .pane-left, .pane-right { width: 1fr; }
    #form .pane-title { margin-top: 0; color: $accent; }
    .mid { width: 14; align: center middle; padding: 0 1; }
    .mid Button { width: 12; min-width: 5; margin-bottom: 1; }
    .pathbar { height: 3; }
    .pathbar Button { width: auto; min-width: 8; }
    .pathbar Input { width: 1fr; }
    #m-tree { height: 1fr; }
    #m-table, #p-table { height: 1fr; }
    .editrow { height: 3; }
    .editrow Input { width: 1fr; }
    .editrow Checkbox { width: auto; }
    #preview { border: round $success; padding: 0 1; margin-top: 0; }
    #errors { margin-top: 1; }
    .buttons { height: auto; margin-top: 1; }
    .buttons Button { margin-right: 2; }
    Confirm, TaskLog, LogsView { align: center middle; }
    #dialog { width: 80; height: auto; max-height: 80%; border: thick $primary; background: $surface; padding: 1 2; }
    #confirm-detail { height: auto; max-height: 20; margin-top: 1; }
    #tasklog { width: 90%; height: 85%; border: thick $primary; background: $surface; padding: 1 2; }
    #tasklog RichLog { height: 1fr; border: round $panel; }
    #gpu-info, #df { padding: 1 2; }
    #cleanup-buttons { height: auto; padding: 0 2; }
    #cleanup-buttons Button { margin: 1 1 0 0; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("f5,ctrl+l", "refresh_all", "Refresh"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.containers: dict[str, dict] = {}   # 이름 → inspect
        self.image_names: list[str] = []
        self.gpus = list_gpus()
        self.server_tz = load_state().get("server_tz", "")
        self.pathmap = PathMap()

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="tab-containers", id="main-tabs"):
            with TabPane("Containers", id="tab-containers"):
                with Horizontal(id="containers-body"):
                    yield ContainerTable(id="ctable", cursor_type="row", zebra_stripes=True)
                    with VerticalScroll(id="detail"):
                        yield Static(id="detail-text")
            with TabPane("Images", id="tab-images"):
                yield ImageTable(id="itable", cursor_type="row", zebra_stripes=True)
            with TabPane("GPU", id="tab-gpu"):
                yield Static(id="gpu-info")
            with TabPane("Cleanup", id="tab-cleanup"):
                with CleanupPane():
                    yield Static(id="df")
                    with Horizontal(id="cleanup-buttons"):
                        yield Button("Remove stopped containers", id="prune-containers")
                        yield Button("Remove untagged images", id="prune-dangling")
                        yield Button("Remove all unused images", id="prune-images", variant="warning")
                        yield Button("Clear build cache", id="prune-build", variant="warning")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#ctable", DataTable).add_columns("Name", "Status", "Image", "GPU", "Ports", "DooD")
        self.query_one("#itable", DataTable).add_columns("Image", "ID", "Size", "Created")
        self.action_refresh_all()
        self.set_interval(5, self.action_refresh_all)
        self.warm_up()
        self.query_one("#ctable").focus()

    @work(thread=True)
    def warm_up(self) -> None:
        """경로 변환 확인과 rootless 감지를 미리 해둔다 (폼을 열 때 기다리지 않도록)."""
        is_rootless()
        self.pathmap.ensure()
        if not self.server_tz:
            self.server_tz = detect_server_tz()
            if self.server_tz:
                save_state(server_tz=self.server_tz)

    @on(TabbedContent.TabActivated)
    def tab_switched(self, e: TabbedContent.TabActivated) -> None:
        # 아래 키 안내는 포커스된 위젯 기준이라, 탭을 바꾸면 그 탭의 목록으로 포커스를 옮긴다
        if e.tabbed_content.id != "main-tabs":
            return
        target = {"tab-containers": "#ctable", "tab-images": "#itable",
                  "tab-cleanup": "#prune-containers"}.get(e.pane.id or "")
        if target:
            self.query_one(target).focus()

    # ------------------------------------------------ 새로고침

    def action_refresh_all(self) -> None:
        self.load_data()

    @work(thread=True, exclusive=True, group="refresh")
    def load_data(self) -> None:
        try:
            names = [c["Names"] for c in docker_json_lines("ps", "-a", "--format", "{{json .}}")]
            conts = {c["Name"].lstrip("/"): c for c in inspect(*names)}
            images = docker_json_lines("images", "--format", "{{json .}}")
            df = docker("system", "df")
        except DockerError as e:
            self.call_from_thread(self.notify, str(e), severity="error")
            return
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout if shutil.which("nvidia-smi") else ""
        self.call_from_thread(self.apply_data, conts, images, df, gpus)

    def apply_data(self, conts: dict, images: list[dict], df: str, gpus: str) -> None:
        self.containers = conts
        self.image_names = [f"{i['Repository']}:{i['Tag']}" for i in images if i["Repository"] != "<none>"]
        self.fill_containers()
        self.fill_images(images)
        self.fill_gpu(gpus)
        self.query_one("#df", Static).update(Text("Disk usage (docker system df)\n\n" + df))

    @staticmethod
    def refill(table: DataTable, rows: list[tuple[str, list]]) -> None:
        """커서 위치를 유지하며 다시 채운다."""
        keep = None
        if table.row_count:
            try:
                keep = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
            except Exception:
                keep = None
        table.clear()
        for key, cells in rows:
            table.add_row(*[c if isinstance(c, Text) else Text(str(c)) for c in cells], key=key)
        keys = [k for k, _ in rows]
        if keep in keys:
            table.move_cursor(row=keys.index(keep))

    def fill_containers(self) -> None:
        rows = []
        order = sorted(self.containers.values(),
                       key=lambda c: (not c["State"].get("Running"), c["Name"]))
        for c in order:
            name = c["Name"].lstrip("/")
            st = c["State"]
            status = Text(st.get("Status", ""), style="green" if st.get("Running") else "dim")
            rows.append((name, [name, status, c["Config"].get("Image", ""), gpu_desc(c),
                                ", ".join(ports_of(c)), "O" if is_dood(c) else ""]))
        self.refill(self.query_one("#ctable", DataTable), rows)
        self.show_detail()

    def fill_images(self, images: list[dict]) -> None:
        rows, seen = [], set()
        for i in images:
            ref = f"{i['Repository']}:{i['Tag']}" if i["Repository"] != "<none>" else i["ID"]
            if ref in seen:
                continue
            seen.add(ref)
            rows.append((ref, [ref, i["ID"][:12], i.get("Size", ""), i.get("CreatedSince", "")]))
        self.refill(self.query_one("#itable", DataTable), rows)

    def fill_gpu(self, raw: str) -> None:
        if not raw.strip():
            self.query_one("#gpu-info", Static).update("nvidia-smi not found.")
            return
        t = Text()
        users: dict[str, list[str]] = {}
        for c in self.containers.values():
            if c["State"].get("Running") and (g := gpu_desc(c)):
                ids = [x.strip() for x in g.split(",")] if g != "all" else ["*"]
                for i in ids:
                    users.setdefault(i, []).append(c["Name"].lstrip("/"))
        for line in raw.strip().splitlines():
            idx, name, used, total, util = [x.strip() for x in line.split(",")]
            used_i, total_i = int(used), int(total)
            bar_n = int(30 * used_i / max(total_i, 1))
            t.append(f"GPU {idx}  ", style="bold")
            t.append(f"{name}\n")
            t.append("  Memory ")
            t.append("█" * bar_n, style="cyan")
            t.append("░" * (30 - bar_n), style="dim")
            t.append(f" {used_i / 1024:.1f}/{total_i / 1024:.1f} GiB   util {util}%\n")
            names = users.get(idx, []) + users.get("*", [])
            t.append("  Running containers with access: ")
            t.append((", ".join(names) or "none") + "\n\n", style="yellow" if names else "dim")
        t.append("(Containers started with all GPUs appear under every GPU. Check memory and utilization for actual use.)",
                 style="dim")
        self.query_one("#gpu-info", Static).update(t)

    # ------------------------------------------------ 선택/상세

    def selected(self, table_id: str) -> str | None:
        table = self.query_one(f"#{table_id}", DataTable)
        if not table.row_count:
            return None
        return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value

    def selected_container(self) -> dict | None:
        name = self.selected("ctable")
        c = self.containers.get(name) if name else None
        if c is None:
            self.notify("Select a container.", severity="warning")
        return c

    @on(DataTable.RowHighlighted, "#ctable")
    def show_detail(self) -> None:
        name = self.selected("ctable")
        c = self.containers.get(name) if name else None
        box = self.query_one("#detail-text", Static)
        if not c:
            box.update(Text("No containers. Press n to create one.", style="dim"))
            return
        t = Text()
        t.append(f"{name}\n", style="bold")
        t.append(f"Image    {c['Config'].get('Image', '')}\n")
        t.append(f"Status   {c['State'].get('Status', '')}  (created {c.get('Created', '')[:19].replace('T', ' ')})\n")
        t.append(f"GPU      {gpu_desc(c) or 'none'}\n")
        t.append(f"DooD     {'connected' if is_dood(c) else 'no'}\n")
        ports = ports_of(c)
        t.append(f"Ports    {', '.join(ports) or 'none'}\n")
        hc = c.get("HostConfig") or {}
        if hc.get("NanoCpus"):
            t.append(f"CPU      {hc['NanoCpus'] / 1e9:g} cores\n")
        t.append(f"Memory   {human_size(hc['Memory']) if hc.get('Memory') else 'no limit'}"
                 f"   shm {'host IPC' if hc.get('IpcMode') == 'host' else human_size(hc.get('ShmSize') or DEFAULT_SHM)}\n")
        t.append("\nMounts\n", style="bold")
        for m in c.get("Mounts") or []:
            src = m.get("Source") if m.get("Type") == "bind" else f"[{m.get('Type')}] {m.get('Name', '')}"
            t.append(f"  {src}\n", style="cyan")
            t.append(f"    → {m.get('Destination')}{'' if m.get('RW', True) else ' (ro)'}\n")
        if not c.get("Mounts"):
            t.append("  none\n", style="dim")
        cmd = container_argv(c)
        t.append("\nCommand\n", style="bold")
        t.append(f"  {shlex.join(cmd) if cmd else '-'}\n")
        t.append("\nServices  ", style="bold")
        t.append("autostart ready (press v)\n" if has_hook(c) else "no startup hook (press v for details)\n",
                 style="green" if has_hook(c) else "dim")
        restart = (c["HostConfig"].get("RestartPolicy") or {}).get("Name") or "no"
        t.append(f"\nRestart policy  {restart}\n")
        box.update(t)

    # ------------------------------------------------ 컨테이너 동작

    def action_new_container(self) -> None:
        st = load_state()
        self.push_screen(RunForm(RunSpec(image=st.get("last_image", ""), tz=st.get("last_tz", TZ_SERVER))))

    def action_recreate(self) -> None:
        c = self.selected_container()
        if not c:
            return
        spec, warns = spec_from_container(c)
        self.push_screen(RunForm(spec, recreate=c, warns=warns))

    def open_recreate(self, name: str) -> None:
        try:
            c = inspect(name)[0]
        except (DockerError, IndexError):
            return
        spec, warns = spec_from_container(c)
        self.push_screen(RunForm(spec, recreate=c, warns=warns))

    def action_services(self) -> None:
        c = self.selected_container()
        if c:
            self.notify("Loading services…", timeout=2)
            self.load_services(c["Name"].lstrip("/"))

    @work(thread=True, exclusive=True, group="services")
    def load_services(self, name: str) -> None:
        try:
            c = inspect(name)[0]
        except (DockerError, IndexError) as e:
            self.call_from_thread(self.notify, str(e), severity="error")
            return
        cfg = svc.read_config(name) or svc.default_config()
        running = bool(c["State"].get("Running"))
        installed = svc.installed(name) if running else {}
        statuses = svc.status(name) if running and has_hook_or_script(name) else {}
        # 화면 객체는 메인 스레드에서 만들어야 한다 (Python 3.9 의 Textual 은 이벤트 루프가 필요)
        self.call_from_thread(lambda: self.push_screen(ServicesScreen(c, cfg, installed, statuses)))

    def action_start_stop(self) -> None:
        c = self.selected_container()
        if not c:
            return
        name = c["Name"].lstrip("/")
        if c["State"].get("Running"):
            self.push_screen(Confirm(f"Stop {name}?"),
                             lambda ok: ok and self.bg_docker(f"Stop {name}", "stop", name))
        else:
            self.bg_docker(f"Start {name}", "start", name)

    def action_restart_container(self) -> None:
        c = self.selected_container()
        if c:
            name = c["Name"].lstrip("/")
            self.push_screen(Confirm(f"Restart {name}?"),
                             lambda ok: ok and self.bg_docker(f"Restart {name}", "restart", name))

    def action_delete_container(self) -> None:
        c = self.selected_container()
        if not c:
            return
        name = c["Name"].lstrip("/")
        detail = "This container is running. It will be force-stopped and removed.\n" if c["State"].get("Running") else ""
        detail += "Files in mounted host folders are not deleted.\nChanges made only inside the container will be lost."
        self.push_screen(Confirm(f"Delete {name}?", detail),
                         lambda ok: ok and self.bg_docker(f"Delete {name}", "rm", "-f", name))

    def action_show_logs(self) -> None:
        c = self.selected_container()
        if c:
            self.push_screen(LogsView(c["Name"].lstrip("/")))

    def action_exec_shell(self) -> None:
        c = self.selected_container()
        if not c:
            return
        name = c["Name"].lstrip("/")
        if not c["State"].get("Running"):
            self.notify("Only running containers can be opened.", severity="warning")
            return
        with self.suspend():
            print(f"\n[{name}] shell. Type exit to return.\n", flush=True)
            subprocess.run(["docker", "exec", "-it", name, "sh", "-c",
                            "if command -v bash >/dev/null; then exec bash -l; else exec sh; fi"])
        self.action_refresh_all()

    @work(thread=True)
    def bg_docker(self, label: str, *args: str) -> None:
        try:
            docker(*args)
            self.call_from_thread(self.notify, f"{label}: done")
        except DockerError as e:
            self.call_from_thread(self.notify, f"{label} failed: {e}", severity="error", timeout=10)
        self.call_from_thread(self.action_refresh_all)

    # ------------------------------------------------ 생성 / 다시 만들기

    def start_run(self, spec: RunSpec) -> None:
        log = TaskLog("Create container")
        self.pop_screen()
        self.push_screen(log)
        self.run_task(log, spec, None, False)

    def start_recreate(self, old: dict, spec: RunSpec, commit: bool) -> None:
        log = TaskLog(f"Recreate: {old['Name'].lstrip('/')}")
        self.pop_screen()
        self.push_screen(log)
        self.run_task(log, spec, old, commit)

    @work(thread=True)
    def run_task(self, log: TaskLog, spec: RunSpec, old: dict | None, commit: bool) -> None:
        def say(msg: str, style: str = "") -> None:
            self.call_from_thread(log.write, msg, style)

        def step(label: str, *args: str) -> str:
            say(f"$ docker {shlex.join(args)}", "dim")
            out = docker(*args)
            say(f"✓ {label}", "green")
            return out

        renamed = stopped = False
        cid = ""
        name_was_free = False
        old_name = old["Name"].lstrip("/") if old else ""
        backup = f"{old_name}-old-{time.strftime('%Y%m%d-%H%M%S')}"
        try:
            if old:
                if commit:
                    say("Committing... (may take a while depending on changes inside the container)")
                    step(f"Saved image {spec.image}", "commit", old_name, spec.image)
                if old["State"].get("Running"):
                    step(f"Stopped {old_name}", "stop", old_name)
                    stopped = True
                if spec.name == old_name:
                    step(f"Renamed the original container to {backup}", "rename", old_name, backup)
                    renamed = True
            if not image_exists(spec.image):
                say(f"$ docker pull {spec.image}", "dim")
                if stream_cmd(["docker", "pull", spec.image], lambda line: say(line, "dim")) != 0:
                    raise DockerError(f"Could not pull image: {spec.image}")
                say(f"✓ Pulled image {spec.image}", "green")
            argv = None
            if spec.hook:
                if has_sh(spec.image):
                    argv = resolve_argv(spec)
                else:
                    say("! This image has no /bin/sh, so Services (autostart) will not be available.", "yellow")
            run_args = spec.args(argv)
            name_was_free = bool(spec.name) and not container_exists(spec.name)
            cid = step("Created new container", "create", *run_args[3:]).strip()
            if old and not commit and svc.copy_config(old["Id"], cid):
                say("✓ Copied service settings (/etc/tdock) from the original container", "green")
            step("Started new container", "start", cid)
            time.sleep(2)
            state = inspect(cid)[0]["State"]
            if not state.get("Running"):
                logs = docker("logs", "--tail", "30", cid, check=False)
                raise DockerError(f"Container exited immediately (exit {state.get('ExitCode')})\n{logs}")
            say(f"\nDone: {spec.name or cid[:12]} is running", "bold green")
            if not old:
                save_state(last_image=spec.image, last_tz=spec.tz)
            if old:
                say(f"The original container is kept, stopped, as {backup if renamed else old_name}.")
                say("Once the new container works, select the old one in the list and press d to delete it.")
                if commit:
                    say(f"The committed image {spec.image} is in the Images tab (used by the new container).")
        except DockerError as e:
            say(f"\n✗ Failed: {e}", "bold red")
            # 이번에 만들어진 컨테이너만 정리 (run이 실패해도 created 상태로 이름이 남을 수 있음)
            if cid:
                docker("rm", "-f", cid, check=False)
            elif name_was_free:
                docker("rm", "-f", spec.name, check=False)
            if old:
                say("\nRolling back...", "yellow")
                try:
                    if renamed:
                        step(f"Restored name {old_name}", "rename", backup, old_name)
                    if stopped:
                        step(f"Restarted {old_name}", "start", old_name)
                    say("The original container was restored.", "yellow")
                except DockerError as e2:
                    say(f"Rollback failed: {e2}\nCheck the state with 'docker ps -a'.", "bold red")
        finally:
            self.call_from_thread(log.finish)
            self.call_from_thread(self.action_refresh_all)

    # ------------------------------------------------ 이미지 동작

    def action_run_image(self) -> None:
        ref = self.selected("itable")
        if ref:
            self.push_screen(RunForm(RunSpec(image=ref)))

    def action_pull_image(self) -> None:
        def chosen(ref: str | None) -> None:
            if ref and not image_exists(ref):
                self.pull(ref)
            elif ref:
                self.notify(f"Pulled {ref}")
                self.action_refresh_all()
        self.push_screen(ImagePicker(pull_mode=True), chosen)

    def pull(self, ref: str) -> None:
        log = TaskLog(f"pull: {ref}")
        self.push_screen(log)
        self.stream_task(log, ["docker", "pull", ref])

    @work(thread=True)
    def stream_task(self, log: TaskLog, cmd: list[str]) -> None:
        self.call_from_thread(log.write, f"$ {shlex.join(cmd)}", "dim")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            self.call_from_thread(log.write, line.rstrip())
        rc = p.wait()
        self.call_from_thread(log.write, "✓ Done" if rc == 0 else f"✗ Failed (exit {rc})",
                              "bold green" if rc == 0 else "bold red")
        self.call_from_thread(log.finish)
        self.call_from_thread(self.action_refresh_all)

    def action_delete_image(self) -> None:
        ref = self.selected("itable")
        if not ref:
            return
        users = [c["Name"].lstrip("/") for c in self.containers.values()
                 if c["Config"].get("Image") == ref or c["Image"].startswith("sha256:" + ref)]
        if users:
            self.notify(f"{ref} is used by containers: {', '.join(users)}\nDelete those containers first.",
                        severity="warning", timeout=8)
            return
        self.push_screen(Confirm(f"Delete image {ref}?", "If the image is on Docker Hub, you can pull it again."),
                         lambda ok: ok and self.bg_docker(f"Delete {ref}", "rmi", ref))

    # ------------------------------------------------ 정리

    @on(Button.Pressed)
    def cleanup(self, e: Button.Pressed) -> None:
        bid = e.button.id or ""
        if not bid.startswith("prune-"):
            return
        if bid == "prune-containers":
            stopped = [c["Name"].lstrip("/") for c in self.containers.values() if not c["State"].get("Running")]
            if not stopped:
                self.notify("No stopped containers.")
                return
            self.push_screen(
                Confirm(f"Delete {len(stopped)} stopped containers?",
                        "\n".join(stopped) + "\n\nThis includes *-old-* containers kept after recreating."),
                lambda ok: ok and self.prune("Remove stopped containers", "container", "prune", "-f"))
        elif bid == "prune-dangling":
            self.push_screen(Confirm("Delete untagged (<none>) images?"),
                             lambda ok: ok and self.prune("Remove untagged images", "image", "prune", "-f"))
        elif bid == "prune-images":
            used = {c["Image"] for c in self.containers.values()}
            unused = [i for i in self.image_names
                      if not any(c["Config"].get("Image") == i for c in self.containers.values())]
            self.push_screen(
                Confirm("Delete all images not used by any container?",
                        "To be deleted (tagged):\n" + ("\n".join(unused) or "none")
                        + f"\n\nImages used by stopped containers are kept ({len(used)} in use)."),
                lambda ok: ok and self.prune("Remove unused images", "image", "prune", "-a", "-f"))
        elif bid == "prune-build":
            self.push_screen(
                Confirm("Clear the entire build cache?", "The next docker build will start from scratch."),
                lambda ok: ok and self.prune("Clear build cache", "builder", "prune", "-a", "-f"))

    def prune(self, label: str, *args: str) -> None:
        log = TaskLog(label)
        self.push_screen(log)
        self.stream_task(log, ["docker", *args])


def main() -> None:
    import sys
    if not shutil.which("docker"):
        sys.exit("tdock: the docker CLI was not found in PATH. Install Docker (or its CLI) first.")
    p = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("tdock: cannot reach the Docker daemon.\n" + (p.stderr.strip() or p.stdout.strip()))
    DockerTUI().run()


if __name__ == "__main__":
    main()
