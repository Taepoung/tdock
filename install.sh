#!/usr/bin/env bash
# tdock installer: works with or without a usable Python, no root needed.
#
#   curl -fsSL https://raw.githubusercontent.com/Taepoung/tdock/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/Taepoung/tdock/main/install.sh | bash -s -- --uninstall
#
# Environment overrides:
#   TDOCK_SOURCE   pip-installable source (package name, URL or local path). Default: tdock from PyPI
#   TDOCK_HOME     install directory (default: ~/.local/share/tdock)
#   TDOCK_BIN_DIR  where the tdock command is linked (default: ~/.local/bin)
set -euo pipefail

SOURCE="${TDOCK_SOURCE:-tdock}"
PREFIX="${TDOCK_HOME:-$HOME/.local/share/tdock}"
BIN_DIR="${TDOCK_BIN_DIR:-$HOME/.local/bin}"
VENV="$PREFIX/venv"

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

fetch() {
    if command -v curl >/dev/null 2>&1; then curl -fsSL "$1"
    elif command -v wget >/dev/null 2>&1; then wget -qO- "$1"
    else die "curl or wget is required"; fi
}

# 잘못된 TDOCK_HOME(/, 홈 등)으로 엉뚱한 폴더를 지우지 않도록 tdock 전용 경로만 허용
case "$(basename "$PREFIX")" in
    *tdock*) ;;
    *) die "TDOCK_HOME must be a dedicated directory whose name contains 'tdock' (got: $PREFIX)" ;;
esac

if [ "${1:-}" = "--uninstall" ]; then
    rm -rf "$PREFIX"
    [ -L "$BIN_DIR/tdock" ] && rm -f "$BIN_DIR/tdock"
    say "tdock removed (settings in ~/.config/tdock were kept)"
    exit 0
fi

# Python 3.9+ that can create a venv with pip
find_python() {
    local p
    for p in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
        command -v "$p" >/dev/null 2>&1 || continue
        "$p" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null || continue
        "$p" -c 'import venv, ensurepip' 2>/dev/null || continue
        echo "$p"
        return 0
    done
    return 1
}

install_with_python() {
    say "Using $($1 --version 2>&1) at $(command -v "$1")"
    "$1" -m venv "$VENV"
    "$VENV/bin/python" -m pip install -q --upgrade pip
    "$VENV/bin/python" -m pip install -q "$SOURCE"
}

install_with_uv() {
    local uv
    if command -v uv >/dev/null 2>&1; then
        uv="$(command -v uv)"
    else
        say "No usable Python 3.9+ found; downloading uv to provide one"
        fetch https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PREFIX/uv" UV_NO_MODIFY_PATH=1 sh >/dev/null
        uv="$PREFIX/uv/uv"
    fi
    export UV_PYTHON_INSTALL_DIR="$PREFIX/python"   # keep the downloaded Python inside PREFIX
    "$uv" venv -q --python 3.12 "$VENV"
    "$uv" pip install -q --python "$VENV/bin/python" "$SOURCE"
}

case "$(uname -s)" in
    Linux) ;;
    *) warn "tdock is developed for Linux; $(uname -s) is untested" ;;
esac

say "Installing tdock into $PREFIX"
rm -rf "$VENV"
mkdir -p "$PREFIX" "$BIN_DIR"
if py="$(find_python)"; then
    install_with_python "$py" || { warn "install with $py failed; falling back to uv"; rm -rf "$VENV"; install_with_uv; }
else
    install_with_uv
fi

ln -sf "$VENV/bin/tdock" "$BIN_DIR/tdock"
"$VENV/bin/python" -c 'import tdock.app' || die "installation check failed"
say "Installed: $BIN_DIR/tdock ($("$VENV/bin/python" -c 'import tdock; print(tdock.__version__)'))"

command -v docker >/dev/null 2>&1 || warn "docker CLI not found; tdock needs it to work"
case ":$PATH:" in
    *":$BIN_DIR:"*) say "Run: tdock" ;;
    *) say "Add $BIN_DIR to PATH, e.g.:  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
       say "Then run: tdock" ;;
esac
