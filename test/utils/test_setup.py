import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import venv
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement

_ROOT = Path(__file__).resolve().parents[2]


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


def _credential_workflow(name):
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((_ROOT / ".github/workflows" / name).read_text())


@pytest.mark.parametrize(
    "filename, overrides",
    [
        ("benchmarks.yml", {"benchmark-upload": {"contents": "write"}}),
        ("benchmarks_pr.yml", {}),
        ("build-wheel.yml", {"build": {"contents": "read", "id-token": "write"}}),
        ("nightly_build.yml", {"upload-wheel": {}}),
        (
            "release.yml",
            {
                "build": {"contents": "read", "id-token": "write"},
                "collect-wheels": {},
                "update-docs": {"contents": "write"},
                "create-release": {"contents": "write"},
                "publish-pypi": {"id-token": "write"},
                "release-summary": {},
            },
        ),
        (
            "nightly_orchestrator.yml",
            {
                "docs": {"contents": "write"},
                "benchmarks": {"contents": "write"},
                "update-status-dashboard": {"contents": "write"},
            },
        ),
        ("test-macos.yml", {}),
    ],
)
def test_ci_credential_permissions(filename, overrides):
    workflow = _credential_workflow(filename)
    assert workflow["permissions"] == {"contents": "read"}
    for name, job in workflow["jobs"].items():
        assert job.get("permissions", workflow["permissions"]) == overrides.get(
            name, {"contents": "read"}
        ), name


@pytest.mark.parametrize(
    "filename",
    [
        "benchmarks.yml",
        "benchmarks_pr.yml",
        "build-wheel.yml",
        "nightly_build.yml",
        "release.yml",
        "test-linux.yml",
        "test-rl-gpu.yml",
        "docs.yml",
    ],
)
def test_ci_credential_checkout(filename):
    for name, job in _credential_workflow(filename)["jobs"].items():
        for step in job.get("steps", []):
            if not step.get("uses", "").startswith("actions/checkout@"):
                continue
            settings = step.get("with", {})
            if (filename, name) == ("release.yml", "update-docs"):
                # This job still uses plain git pull/push, not an action token.
                assert settings.get("persist-credentials", True) is True
            else:
                assert settings.get("persist-credentials") is False, (name, step)


@pytest.mark.parametrize("filename", ["release.yml", "nightly_orchestrator.yml"])
def test_ci_credential_reusable_permission_ceiling(filename):
    workflow = _credential_workflow(filename)
    levels = {"none": 0, "read": 1, "write": 2}
    for name, job in workflow["jobs"].items():
        called = job.get("uses", "")
        if not called.startswith("./.github/workflows/"):
            continue
        ceiling = job.get("permissions", workflow.get("permissions", {}))
        callee = _credential_workflow(called.rsplit("/", 1)[1])
        for child_name, child in callee["jobs"].items():
            requested = child.get("permissions", callee.get("permissions", ceiling))
            for permission, level in requested.items():
                assert levels[level] <= levels[ceiling.get(permission, "none")], (
                    name,
                    child_name,
                    permission,
                )


def test_ci_credential_nightly_publisher():
    workflow = _credential_workflow("nightly_build.yml")
    assert "secrets." not in json.dumps(workflow.get("env", {}))
    for name, job in workflow["jobs"].items():
        assert "secrets." not in json.dumps(job.get("env", {}))
        for step in job.get("steps", []):
            if step.get("name") != "Push tensordict Binary to PYPI":
                assert "secrets." not in json.dumps(step), (name, step)
    upload = workflow["jobs"]["upload-wheel"]
    assert upload["needs"] == "test-wheel"
    assert "workflow_dispatch" in upload["if"] and "schedule" in upload["if"]
    assert "github.repository_owner == 'pytorch'" in upload["if"]
    steps = upload["steps"]
    install = next(s for s in steps if s.get("name") == "Install publishing tools")
    publish = next(
        s for s in steps if s.get("name") == "Push tensordict Binary to PYPI"
    )
    assert steps.index(install) < steps.index(publish)
    assert "--require-hashes" in install["run"]
    assert "--only-binary=:all:" in install["run"]
    assert "pip==26.2.2" in install["run"]
    assert "certifi==2026.10.23" in install["run"]
    assert "twine==" in install["run"]
    assert '"$RUNNER_TEMP/publisher/bin/python" -m twine upload' in publish["run"]
    for block in install["run"].split("<<'REQUIREMENTS'\n")[1:]:
        requirements = block.split("\nREQUIREMENTS")[0]
        for line in requirements.splitlines():
            assert re.fullmatch(r"[\w.-]+==[\w.]+ --hash=sha256:[0-9a-f]{64}", line), (
                line
            )
    assert publish["env"] == {"PYPI_TOKEN": "${{ secrets.PYPI_TOKEN }}"}
    assert "pip install" not in publish["run"]
    assert '--password "$PYPI_TOKEN"' in publish["run"]
    assert "--skip-existing" in publish["run"]
    assert "wheels/tensordict_nightly-*.whl" in publish["run"]
    assert workflow["jobs"]["test-wheel"]["continue-on-error"] is True
    assert workflow["jobs"]["test-subset"]["continue-on-error"] is True
    for name in ("build-wheel", "test-wheel", "test-subset"):
        runs = [s["run"] for s in workflow["jobs"][name]["steps"] if "run" in s]
        upgrade = next(run for run in runs if "pip install --upgrade" in run)
        assert '"pip>=26.2.2"' in upgrade
        assert '"certifi>=2026.10.23"' in upgrade
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "-artifact@" in step.get("uses", ""):
                assert step["with"]["name"] == "tensordict-nightly-wheel"


def test_ci_credential_stable_publisher():
    workflow = _credential_workflow("release.yml")
    publish = workflow["jobs"]["publish-pypi"]
    assert publish["permissions"] == {"id-token": "write"}
    assert publish["environment"] == {
        "name": "pypi",
        "url": "https://pypi.org/project/tensordict/",
    }
    assert publish["needs"] == ["sanity-checks", "collect-wheels", "create-release"]
    assert workflow["jobs"]["collect-wheels"]["needs"] == ["sanity-checks", "build"]
    assert "needs.build.result == 'success'" in workflow["jobs"]["collect-wheels"]["if"]
    step = next(s for s in publish["steps"] if "pypi-publish@" in s.get("uses", ""))
    assert step["uses"] == "pypa/gh-action-pypi-publish@release/v1"
    assert step["if"] == "${{ env.DRY_RUN != 'true' }}"
    assert step["with"] == {"verbose": True, "print-hash": True, "skip-existing": True}
    assert "secrets." not in json.dumps(publish)
    preview = _credential_workflow("upload-docs-preview.yml")
    assert preview["permissions"]["id-token"] == "write"


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
    if no_build_isolation and sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip(
            "no-build-isolation installs are unreliable on free-threaded Python"
        )

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


def test_ci_bootstrap_uses_verified_installers():
    paths = [
        ".github/unittest/linux/scripts/setup_env.sh",
        ".github/unittest/rl_linux_optdeps/scripts/run_all.sh",
        ".github/workflows/benchmarks.yml",
        ".github/workflows/benchmarks_pr.yml",
    ]
    for path in paths:
        source = (_ROOT / path).read_text()
        assert "install_ci_tool.sh" in source, path
        assert "ci-constraints.txt" in source, path
        assert "http://repo.continuum.io" not in source, path
        assert "https://pyenv.run" not in source, path
        assert "https://astral.sh/uv/install.sh" not in source, path


@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux CI installers use GNU coreutils"
)
@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
@pytest.mark.parametrize("tool", ["miniconda", "uv", "pyenv"])
@pytest.mark.parametrize("valid_checksum", [False, True])
def test_ci_installer_checks_before_execution(tmp_path, tool, arch, valid_checksum):
    """A local harmless download must pass SHA-256 before execution/extraction."""
    script = _ROOT / ".github/scripts/install_ci_tool.sh"
    source = script.read_text()
    package = tmp_path / "package"
    package.mkdir()
    payload = package / "payload"
    payload.write_text(
        '#!/bin/sh\ncase "$0" in *.sh) ;; *) exit 1 ;; esac\necho installer-executed\n'
    )
    if tool != "miniconda":
        archive = tmp_path / "archive.tar.gz"
        _run(["tar", "-czf", str(archive), "-C", str(tmp_path), "package"], cwd=_ROOT)
        payload = archive
    digest = _run(["sha256sum", str(payload)], cwd=_ROOT).split()[0]
    # Replace only this tool's architecture pin in a private copy, never the real file.
    pins = {
        "miniconda": "4ecd6f5ced565b02ef2fd5992fca4244b5e695ef5bff2beffbf03c7da5686e35",
        "uv": "f29c3e94ffabd7e6f3ba521dba169a9b7ec12100bf3257087f0f24f6123f3645",
        "pyenv": "0db903040bd06f4adc333bff60aefbdfc4af522f8e2b13f9cce3fbf721b3e468",
    }
    if arch == "aarch64":
        pins.update(
            miniconda="772e0bbaa3639b34b203e0c5322de403cc4cc478c3f6cd8696c58dbb4b902344",
            uv="229e194255c7958d86d56e8790d969a6f74736592133f0dfd0076f754c642c1d",
        )
    assert source.count(pins[tool]) == 1
    if valid_checksum:
        source = source.replace(pins[tool], digest)
    local_script = tmp_path / "install.sh"
    local_script.write_text(source)
    # Download stub accepts only HTTPS with HTTPS-only redirects.
    curl = tmp_path / "curl"
    curl.write_text(
        "#!/bin/bash\n"
        '[[ "$*" == *"--proto =https --proto-redir =https"* ]] || exit 2\n'
        'while [[ "$1" != "--output" ]]; do shift; done\n'
        f'cp "{payload}" "$2"\n'
    )
    curl.chmod(0o755)
    destination = tmp_path / "installed"
    result = subprocess.run(
        ["bash", str(local_script), tool, str(destination), arch],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    if valid_checksum:
        assert result.returncode == 0, result.stderr
        if tool == "miniconda":
            assert "installer-executed" in result.stdout
        else:
            assert (destination / "payload").read_text().startswith("#!/bin/sh")
    else:
        assert result.returncode != 0
        assert "installer-executed" not in result.stdout
        assert not destination.exists()


@pytest.mark.parametrize(
    "package,vulnerable,patched",
    [
        ("uv", "0.13.3", "0.13.6"),
        ("pip", "26.2.1", "26.2.2"),
        ("certifi", "2026.10.17", "2026.10.23"),
        ("Brotli", "1.2.0", "1.2.1"),
        ("brotlicffi", "1.2.0.0", "1.2.1.0"),
    ],
)
def test_ci_security_constraints(package, vulnerable, patched):
    constraints = {
        requirement.name: requirement.specifier
        for line in (_ROOT / ".github/ci-constraints.txt").read_text().splitlines()
        if line and not line.startswith("#")
        for requirement in [Requirement(line)]
    }
    assert vulnerable not in constraints[package]
    assert patched in constraints[package]


@pytest.mark.slow
def test_wheel_is_pure_python(isolated_venv, tmp_path):
    """tensordict compiles nothing, so one wheel serves every platform."""
    # The venv's python has pip, which sys.executable lacks in a uv venv made
    # without --seed.
    _run(
        [
            str(isolated_venv["python"]),
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(tmp_path),
            str(_ROOT),
        ],
        cwd=_ROOT,
        timeout=20 * 60,
    )
    (wheel,) = tmp_path.glob("tensordict-*.whl")
    assert wheel.name.endswith("-py3-none-any.whl"), wheel.name
    with zipfile.ZipFile(wheel) as archive:
        binaries = [
            name
            for name in archive.namelist()
            if name.endswith((".so", ".pyd", ".dll", ".dylib"))
        ]
        top_level = {name.split("/")[0] for name in archive.namelist()}
    assert not binaries, binaries
    # Only the tensordict package and its metadata: no scripts/, and no build/
    # left by an earlier build in the same tree.
    dist_info = wheel.name.removesuffix("-py3-none-any.whl") + ".dist-info"
    assert top_level == {"tensordict", dist_info}, top_level


@pytest.mark.slow
def test_pyarrow_extra_wheel_metadata(isolated_venv):
    """The Parquet security floor and its CI pin must agree in wheel metadata."""
    python = isolated_venv["python"]
    _run(
        [str(python), "-m", "pip", "install", "--no-deps", str(_ROOT)],
        cwd=_ROOT,
    )
    requirements = json.loads(
        _run(
            [
                str(python),
                "-c",
                "import importlib.metadata as md, json; "
                "print(json.dumps(md.requires('tensordict')))",
            ],
            cwd=isolated_venv["probe_dir"],
            env={**os.environ, "PYTHONPATH": ""},
        )
    )
    arrow_requirements = [
        req for text in requirements if (req := Requirement(text)).name == "pyarrow"
    ]
    ci_recipe = (
        _ROOT / ".github/unittest/linux/scripts/run_test_min_deps.sh"
    ).read_text()
    (ci_version,) = re.findall(r'"pyarrow==([^"\n]+)"', ci_recipe)
    for extra in ("parquet", "tabular"):
        (requirement,) = [
            req
            for req in arrow_requirements
            if req.marker is not None and req.marker.evaluate({"extra": extra})
        ]
        assert str(requirement.specifier) == ">=14.0.1"
        for old_version in ("10.0.0", "11.0.0", "12.0.1", "13.0.0", "14.0.0"):
            assert old_version not in requirement.specifier
        assert "14.0.1" in requirement.specifier
        assert ci_version == "14.0.1"
        assert ci_version in requirement.specifier


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
