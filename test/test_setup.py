import argparse
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]


def _run(
    cmd,
    *,
    cwd,
    env=None,
    timeout=60 * 60,
):
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"cwd: {cwd}\n"
            f"cmd: {cmd}\n"
            f"exit_code: {proc.returncode}\n"
            f"output:\n{proc.stdout}"
        )
    return proc.stdout


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_pip(python: Path) -> None:
    try:
        _run([str(python), "-m", "pip", "-V"], cwd=_ROOT, timeout=60)
        return
    except Exception:
        pass
    _run([str(python), "-m", "ensurepip", "--upgrade"], cwd=_ROOT, timeout=5 * 60)
    _run([str(python), "-m", "pip", "install", "-U", "pip"], cwd=_ROOT, timeout=10 * 60)


@pytest.fixture
def isolated_venv(tmp_path_factory: pytest.TempPathFactory):
    # Dedicated venv per test, deleted at teardown.
    # We avoid system_site_packages so the test can't accidentally import a globally
    # installed tensordict instead of the one we just installed.
    venv_dir = tmp_path_factory.mktemp("tensordict-install-venv")
    venv.EnvBuilder(with_pip=True, system_site_packages=False, clear=True).create(
        str(venv_dir)
    )
    python = _venv_python(venv_dir)
    _ensure_pip(python)
    probe_dir = tmp_path_factory.mktemp("tensordict-install-probe")
    yield {"venv_dir": venv_dir, "python": python, "probe_dir": probe_dir}
    shutil.rmtree(venv_dir, ignore_errors=True)


def _install_cmd_prefix(installer: str, *, python: Path) -> list[str]:
    if installer == "uv":
        return ["uv", "pip", "install", "--python", str(python)]
    return [str(python), "-m", "pip", "install"]


def _uninstall(python: Path, pkg: str) -> None:
    # Ensure best-effort removal. We don't error if it's not installed.
    proc = subprocess.run(
        [str(python), "-m", "pip", "uninstall", "-y", pkg],
        cwd=str(_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    _ = proc.stdout


@pytest.mark.slow
@pytest.mark.parametrize("editable", [True, False], ids=["editable", "wheel"])
@pytest.mark.parametrize(
    "no_build_isolation", [True, False], ids=["no-build-isolation", "build-isolation"]
)
def test_install_strategies_version(
    isolated_venv, editable: bool, no_build_isolation: bool
):
    if shutil.which("cmake") is None:
        pytest.skip("cmake not available")

    base_version = (_ROOT / "version.txt").read_text().strip()
    if not base_version:
        raise RuntimeError("Empty version.txt")

    python: Path = isolated_venv["python"]
    venv_dir: Path = isolated_venv["venv_dir"]
    probe_dir: Path = isolated_venv["probe_dir"]

    installer = "uv" if shutil.which("uv") is not None else "pip"
    if installer == "uv" and shutil.which("uv") is None:
        raise RuntimeError("Internal error: uv requested but not found")

    # Ensure build requirements exist in the venv, because no-build-isolation won't
    # install them from pyproject.toml.
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-U",
            "setuptools",
            "wheel",
            "pybind11",
        ],
        cwd=_ROOT,
        timeout=20 * 60,
    )

    # For the no-build-isolation path, explicitly ensure setuptools_scm is absent so
    # we cover the historical failure mode where version becomes 0.0.0.
    if no_build_isolation:
        _uninstall(python, "setuptools_scm")

    cmd = _install_cmd_prefix(installer, python=python)
    if no_build_isolation:
        cmd.append("--no-build-isolation")
    if editable:
        cmd.extend(["-e", "."])
    else:
        cmd.append(".")
    # Avoid network / dependency resolution changes; rely on system_site_packages instead.
    cmd.append("--no-deps")

    _run(cmd, cwd=_ROOT, timeout=60 * 60)

    # Determine what version we expect based on the current branch.
    def _git(args):
        return _run(["git", *args], cwd=_ROOT, timeout=60).strip()

    branch = None
    for key in ("GITHUB_REF_NAME", "GIT_BRANCH", "BRANCH_NAME", "CI_COMMIT_REF_NAME"):
        val = os.environ.get(key)
        if val:
            branch = val
            break
    if branch is None:
        b = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        branch = None if b == "HEAD" else b

    expected_dist_version = base_version
    if not (
        branch is not None
        and (
            branch == f"release/v{base_version}"
            or branch.endswith(f"/release/v{base_version}")
        )
    ):
        expected_dist_version = (
            f"{base_version}+g{_git(['rev-parse', '--short', 'HEAD'])}"
        )

    code = r"""
import importlib.metadata as md
import json

out = {}
out["dist_version"] = md.version("tensordict")
try:
    import tensordict
    out["pkg_version"] = getattr(tensordict, "__version__", None)
    out["pkg_file"] = getattr(tensordict, "__file__", None)
except Exception as err:
    out["import_error"] = repr(err)

print(json.dumps(out))
"""
    # Run from a directory outside the repo, otherwise Python will import the local
    # "tensordict/" folder from the current working directory.
    out = _run([str(python), "-c", code], cwd=probe_dir, timeout=5 * 60)
    info = json.loads(out.strip())

    dist_version = str(info["dist_version"]).strip()
    assert dist_version != "0.0.0"
    assert dist_version == expected_dist_version

    pkg_version = info.get("pkg_version")
    pkg_file = info.get("pkg_file")
    if pkg_version is not None and pkg_file is not None:
        pkg_version = str(pkg_version).strip()
        assert pkg_version != "0.0.0"
        assert pkg_version == expected_dist_version

        pkg_path = Path(pkg_file).resolve()
        if editable:
            assert str(pkg_path).startswith(str(_ROOT.resolve()))
        else:
            assert str(pkg_path).startswith(str(venv_dir.resolve()))
    else:
        # If torch isn't available in the environment, importing tensordict can fail.
        # The packaging version should still be correct.
        assert "dist_version" in info


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
