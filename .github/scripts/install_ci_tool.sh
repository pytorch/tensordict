#!/usr/bin/env bash
# Reviewed installer pins. Refresh URLs and SHA-256 together when upgrading.
set -euo pipefail

tool="${1:?tool: miniconda, uv or pyenv}"
destination="${2:?installation directory}"
arch="${3:-$(uname -m)}"
case "${tool}:${arch}" in
    miniconda:x86_64) sha256=4ecd6f5ced565b02ef2fd5992fca4244b5e695ef5bff2beffbf03c7da5686e35 ;;
    miniconda:aarch64) sha256=772e0bbaa3639b34b203e0c5322de403cc4cc478c3f6cd8696c58dbb4b902344 ;;
    uv:x86_64) sha256=f29c3e94ffabd7e6f3ba521dba169a9b7ec12100bf3257087f0f24f6123f3645 ;;
    uv:aarch64) sha256=229e194255c7958d86d56e8790d969a6f74736592133f0dfd0076f754c642c1d ;;
    pyenv:*) sha256=0db903040bd06f4adc333bff60aefbdfc4af522f8e2b13f9cce3fbf721b3e468 ;;
    *) echo "Unsupported CI installer: ${tool}:${arch}" >&2; exit 1 ;;
esac
case "${tool}" in
    miniconda)
        # Digests published at https://repo.anaconda.com/miniconda/
        url="https://repo.anaconda.com/miniconda/Miniconda3-py314_26.9.4-0-Linux-${arch}.sh"
        ;;
    uv)
        # Digests from the release's matching .tar.gz.sha256 assets.
        url="https://github.com/astral-sh/uv/releases/download/0.13.6/uv-${arch}-unknown-linux-gnu.tar.gz"
        ;;
    pyenv)
        # v2.8.14 = 3fb25668b463ad38ac2db6092dbf476ef744c0ca; hash of the tag archive.
        url="https://github.com/pyenv/pyenv/archive/refs/tags/v2.8.14.tar.gz"
        ;;
esac

# Miniconda's constructor requires the executed filename to end in .sh.
download="$(mktemp --suffix=.sh)"
trap 'rm -f "${download}"' EXIT
curl --fail --location --silent --show-error --retry 3 \
    --proto '=https' --proto-redir '=https' --output "${download}" "${url}"
printf '%s  %s\n' "${sha256}" "${download}" | sha256sum --check --status
# Do not execute or extract anything until verification succeeds.
if [[ "${tool}" == miniconda ]]; then
    bash "${download}" -b -f -p "${destination}"
else
    mkdir -p "${destination}"
    tar -xzf "${download}" --strip-components=1 -C "${destination}"
fi
