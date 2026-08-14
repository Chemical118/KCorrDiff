#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
namespace=ws-md93se6gk3270
training_job=kcorrdiff-stage2-folds-porsche-v2
collector_job=kcorrdiff-stage2-fold-collector-porsche-v2
manifest="${repo_root}/k8s/train-stage2-folds-porsche.yaml"

for job in "${training_job}" "${collector_job}"; do
  if kubectl get job "${job}" -n "${namespace}" >/dev/null 2>&1; then
    echo "refusing to reuse existing immutable Job: ${namespace}/${job}" >&2
    exit 1
  fi
done

pv_name="$(kubectl get pvc saycorn-volume -n "${namespace}" -o jsonpath='{.spec.volumeName}')"
pv_node="$(kubectl get pv "${pv_name}" -o jsonpath='{.spec.nodeAffinity.required.nodeSelectorTerms[0].matchExpressions[0].values[0]}')"
test "${pv_node}" = porsche

python3 "${repo_root}/scripts/apply_telegram_secret.py" --env-file "${repo_root}/.env"
"${repo_root}/scripts/publish_stage2_fold_source.sh"

secret_keys="$(kubectl get secret kcorrdiff-telegram -n "${namespace}" -o json | jq -r '.data | keys | sort | join(",")')"
test "${secret_keys}" = 'bot-token,chat-id'
kubectl apply --dry-run=server -f "${manifest}" >/dev/null
kubectl apply -f "${manifest}"
kubectl patch job "${collector_job}" -n "${namespace}" \
  --type=merge -p '{"spec":{"suspend":false}}'
kubectl patch job "${training_job}" -n "${namespace}" \
  --type=merge -p '{"spec":{"suspend":false}}'

echo "Launched three one-GPU folds on porsche (parallelism 2, batch 12, accumulation 1)."
echo "Monitor with: python3 ${repo_root}/scripts/monitor_stage2_folds.py --watch"
