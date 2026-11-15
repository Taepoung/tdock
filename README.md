# tdock — Docker container manager TUI

A terminal UI for managing Docker containers on shared GPU servers.
Create containers with GPUs, mounts and ports from forms instead of long `docker run` commands,
and change the settings of an existing container without losing what you installed inside.

## Features

- **Containers**: list with status, GPUs, ports and mounts; start/stop, restart, logs, shell, delete
- **Create from a form**
  - Image: pick a local image, search Docker Hub (with tag sizes and dates), or pull by name
  - Resources: per-GPU toggles, memory limit, shared memory (`/dev/shm`), CPU limit (only if the daemon supports it)
  - Mounts: browse host folders and add them with `→`; read-only toggle; one-click "same path + /workspace"
  - Ports: host port / container port, add and remove from a list
  - Timezone: server timezone, any IANA zone, or the image default (works without tzdata in the image)
  - Options shown only when they apply: auto-restart, Docker inside the container (DooD), NVIDIA ulimits, host IPC (multi-GPU), debugger support
  - Live preview of the exact `docker run` command and validation before running
- **Recreate** (`m`): commit the container, then rerun it with changed settings; rolls back automatically on failure
- **Images**: pull, delete, run
- **GPU**: memory and utilization per GPU, and which running containers can access it
- **Cleanup**: disk usage, remove stopped containers, unused images, build cache

## Requirements

- Linux with the `docker` CLI and access to a Docker daemon (rootful or rootless)
- Python 3.9+
- `nvidia-smi` for GPU features (optional)

## Install

**Any Linux, no Python setup needed** (uses [uv](https://docs.astral.sh/uv/), which fetches a Python if needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once
uv tool install tdock
```

Or with the install script, which uses your Python 3.9+ if available and falls back to uv otherwise:

```bash
curl -fsSL https://raw.githubusercontent.com/Taepoung/tdock/main/install.sh | bash
```

If you already manage Python tools:

```bash
pipx install tdock
pip install tdock      # inside a virtualenv / conda env
```

On recent Debian/Ubuntu, a plain `pip install` into the system Python is blocked
(`externally-managed-environment`); use one of the methods above.

Update with `uv tool upgrade tdock`, `pipx upgrade tdock` or `pip install -U tdock`.

## Usage

```bash
tdock
```

| Tab | Keys |
|---|---|
| Containers | `n` new, `m` recreate, `s` start/stop, `t` restart, `e` shell, `l` logs, `d` delete |
| Images | `n` run this image, `p` pull, `d` delete |
| Mount folder tree | `→` add mount, `Backspace` parent, `o` open folder, `r` refresh |
| Everywhere | `F5` refresh, `q` quit, `Ctrl+R` run (in the form), `Esc` cancel |

## Notes

- **Rootless Docker**: detected automatically. Options the daemon cannot honor are hidden
  (for example the CPU limit when the cpu cgroup controller is not delegated, or `memlock=-1`).
- **Running tdock inside a container**: if your shell is itself a container that talks to the host daemon,
  host paths can differ from what you see. tdock detects this, translates mount paths to the paths the daemon sees,
  and blocks paths that only exist inside your container (such as `/tmp`).
- **Mounts** use `--mount` so a missing host path fails instead of silently creating an empty directory.
- **DooD** mounts the Docker socket, the `docker` binary and CLI plugins. The binary mount works in glibc-based images
  (Ubuntu, Debian), not in Alpine.

## Files

- `~/.config/tdock/state.json`: last used image and timezone, detected server timezone
- `~/.config/tdock/pathmap.json`: cached host path translation (only when running inside a container)

## License

MIT
