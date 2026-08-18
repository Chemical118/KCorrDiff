#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
namespace="${KCORRDIFF_NAMESPACE:-ws-md93se6gk3270}"
pod="kcorrdiff-stager"
container="stager"
manifest="${repo_root}/k8s/stager.yaml"
destination="/workspace/KCorrDiff"
remote_stage="/workspace/.staging/kcorrdiff-source-$(date -u +%Y%m%dT%H%M%SZ)-$$"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  kubectl -n "${namespace}" exec "${pod}" -c "${container}" -- \
    rm -rf -- "${remote_stage}" >/dev/null 2>&1 || true
  kubectl -n "${namespace}" delete pod "${pod}" \
    --ignore-not-found --wait=true >/dev/null || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

command -v kubectl >/dev/null
command -v tar >/dev/null
test -f "${repo_root}/pyproject.toml"

kubectl -n "${namespace}" delete pod "${pod}" \
  --ignore-not-found --wait=true >/dev/null
kubectl apply -f "${manifest}"
kubectl -n "${namespace}" wait \
  --for=condition=Ready "pod/${pod}" --timeout=5m
kubectl -n "${namespace}" exec "${pod}" -c "${container}" -- \
  mkdir -p "${remote_stage}"

tar -C "${repo_root}" \
  --exclude='.git' \
  --exclude='.pytest_cache' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env' \
  -cf - . | kubectl -n "${namespace}" exec -i "${pod}" -c "${container}" -- \
    tar -C "${remote_stage}" -xf -

kubectl -n "${namespace}" exec "${pod}" -c "${container}" -- \
  bash -ceu '
    stage="$1"
    destination="$2"

    test -f "${stage}/pyproject.toml"
    test -d "${stage}/kcorrdiff"
    mkdir -p "${destination}"
    find -H "${destination}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
    cp -a "${stage}/." "${destination}/"
    rm -rf -- "${stage}"
    test -f "${destination}/pyproject.toml"
    test -d "${destination}/kcorrdiff"
  ' -- "${remote_stage}" "${destination}"

printf 'Updated KCorrDiff source at %s on PVC saycorn-volume.\n' "${destination}"
