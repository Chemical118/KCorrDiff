#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
namespace=ws-md93se6gk3270
pod=kcorrdiff-stage2-fold-source-publisher
snapshot=stage2-folds-porsche-v1
temporary="/workspace/code/.${snapshot}.incomplete"
destination="/workspace/code/${snapshot}"
run_root="/workspace/runs/${snapshot}"

cleanup() {
  kubectl delete pod "${pod}" -n "${namespace}" --ignore-not-found --wait=true >/dev/null || true
}
trap cleanup EXIT

kubectl apply -f "${repo_root}/k8s/publish-stage2-fold-source.yaml"
kubectl wait -n "${namespace}" --for=condition=Ready "pod/${pod}" --timeout=2m
kubectl exec -n "${namespace}" "${pod}" -- /bin/sh -ceu -- \
  'test ! -e "$1"; test ! -e "$2"; test ! -e "$3"; mkdir -m 0770 "$1"' \
  sh "${temporary}" "${destination}" "${run_root}"

tar \
  --create \
  --file=- \
  --directory="${repo_root}" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env' \
  pyproject.toml kcorrdiff configs \
  scripts/collect_stage2_folds.py \
  scripts/run_stage2_fold.sh \
  scripts/telegram_notify.py \
  | kubectl exec -i -n "${namespace}" "${pod}" -- \
      tar -xf - -C "${temporary}"

kubectl exec -n "${namespace}" "${pod}" -- /bin/sh -ceu -- \
  'test -f "$1/pyproject.toml"; test -f "$1/kcorrdiff/training/train_stage2.py"; test -f "$1/scripts/run_stage2_fold.sh"; chmod -R a-w "$1"; mv "$1" "$2"; sync -f "$(dirname "$2")"' \
  sh "${temporary}" "${destination}"

echo "Published immutable porsche source snapshot: ${destination}"
