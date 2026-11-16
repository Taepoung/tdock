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
- **Services** (`v`): start SSH, a VS Code tunnel, Jupyter or your own commands inside a container
  - per service: *Start with container* and *Restart if it dies*
  - detects what is installed inside, shows live status and logs
  - changes apply immediately without restarting the container; running jobs are not touched
  - also change auto-restart and the memory limit live (`docker update`)
- **Images**: pull, delete, run
- **GPU**: memory and utilization per GPU, and which running containers can access it
- **Cleanup**: disk usage, remove stopped containers, unused images, build cache

## Requirements

- Linux with the `docker` CLI and access to a Docker daemon (rootful or rootless)
- Python 3.9+
- `nvidia-smi` for GPU features (optional)

## Install

Pick one. All of them give you a `tdock` command.

| Method | Command | Good for |
|---|---|---|
| **uv** (recommended) | `uv tool install tdock` | Any Linux. No Python setup needed: uv downloads a Python if yours is missing or too old |
| **pipx** | `pipx install tdock` | If you already use pipx |
| **pip** | `pip install tdock` | Inside a conda env or virtualenv you already use |
| **Install script** | see below | Uses your Python 3.9+ if usable, otherwise falls back to uv |

### uv (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv once (no root needed)
source $HOME/.local/bin/env                       # or open a new terminal
uv tool install tdock
tdock
```

The uv installer adds `~/.local/bin` to your PATH by editing your shell config (`~/.bashrc` and similar),
but that only takes effect in **new** terminals, hence the `source` line.
If you installed uv some other way (or the PATH line is missing), run `uv tool update-shell` once and open a new terminal.

uv and pipx install tdock into its own isolated environment and put only the `tdock` command on your PATH,
so it works whichever conda env or virtualenv is active, and it does not add packages to your environments.

### pip

```bash
conda activate myenv    # or: source .venv/bin/activate
pip install tdock
tdock
```

With pip, `tdock` is installed into **that** environment: the command is available only while it is active,
and its dependencies (such as `textual`) are added to it.

On recent Debian/Ubuntu, `pip install` into the system Python is blocked with
`externally-managed-environment` (also with `--user`). Use uv or pipx there.

### Install script

```bash
curl -fsSL https://raw.githubusercontent.com/Taepoung/tdock/main/install.sh | bash
# uninstall
curl -fsSL https://raw.githubusercontent.com/Taepoung/tdock/main/install.sh | bash -s -- --uninstall
```

Installs into `~/.local/share/tdock` and links `~/.local/bin/tdock`.

### `tdock: command not found`?

uv, pipx and the install script put the command in `~/.local/bin`. Open a new terminal first;
if it is still not found, `~/.local/bin` is not on your PATH yet:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

(`uv tool update-shell` or `pipx ensurepath` do the same.)

### Update / uninstall

| Method | Update | Uninstall |
|---|---|---|
| uv | `uv tool upgrade tdock` | `uv tool uninstall tdock` |
| pipx | `pipx upgrade tdock` | `pipx uninstall tdock` |
| pip | `pip install -U tdock` | `pip uninstall tdock` |

## Usage

```bash
tdock
```

| Tab | Keys |
|---|---|
| Containers | `n` new, `m` recreate, `v` services, `s` start/stop, `t` restart, `e` shell, `l` logs, `d` delete |
| Services | `Ctrl+S` apply, `Esc` close |
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

## Services: how it works

Containers created by tdock get a small startup hook: on every start they run `/etc/tdock/startup.sh`
(if present) in the background and then the image's original command.
The Services screen writes that script and its settings into `/etc/tdock` inside the container,
so they survive restarts and recreating with commit. Each service runs in its own process group
under a tiny supervisor; logs go to `/var/log/tdock/<service>.log`.

- Containers created without tdock have no hook; the Services screen offers a one-time recreate to add it.
- SSH and Jupyter need a port mapping (e.g. `2222:22`), which can only be added by recreating.
- A VS Code tunnel needs a one-time login inside the container: `code tunnel user login --provider github`.
- Images without `/bin/sh` cannot use services.

## Files

- `~/.config/tdock/state.json`: last used image and timezone, detected server timezone
- `~/.config/tdock/pathmap.json`: cached host path translation (only when running inside a container)

## License

MIT
