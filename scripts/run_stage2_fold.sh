#!/usr/bin/env bash
set -euo pipefail

fold_id="${JOB_COMPLETION_INDEX:?missing indexed Job completion index}"
node_name="${NODE_NAME:?missing scheduled node name}"
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
run_root=/workspace/runs/stage2-folds-porsche-v3
output_dir="${run_root}/workers/fold-${fold_id}"
runtime_report=/workspace/envs/stage2-python-v1/pip-report.json
data_contract="${source_root}/configs/data-contract-v1.1.3b.json"
container_image_digest=sha256:6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4

[[ "${fold_id}" =~ ^[0-9]+$ ]]

notify() {
  python3 "${source_root}/scripts/telegram_notify.py" --optional "$1" || true
}
failure_notification() {
  rc=$?
  notify "[KCorrDiff] Stage 2 fold ${fold_id} failed on porsche (exit=${rc})."
  exit "${rc}"
}
trap failure_notification EXIT

install -d -m 0770 \
  "${output_dir}/logs" "${output_dir}/wandb" \
  "${output_dir}/oof" "${output_dir}/provenance" "${run_root}/completion"
exec > >(tee -a "${output_dir}/train.log") 2>&1

test -f "${source_root}/pyproject.toml"
test -f "${source_root}/kcorrdiff/training/train_stage2.py"
test -f "${source_root}/configs/stage2-full-width.yaml"
test -f /workspace/data/cache/radar-v1/manifest.json
test -f /workspace/data/cache/era5-v1/manifest.json
test -f /workspace/data/manifests/event-pretrain-production-v1/training-draw-manifest.jsonl
test -f /workspace/data/manifests/event-pretrain-production-v1/candidate-manifest.json
test -f /workspace/data/manifests/event-pretrain-production-v1/bundle-metadata.json
test -f /workspace/data/static/target_static.npz
test -r /workspace/.ssh/config

actual_policy_sha256="$(python3 -c 'from kcorrdiff.training.train_stage2 import SINGLE_NODE_FOLD_POLICY_SHA256; print(SINGLE_NODE_FOLD_POLICY_SHA256)')"
nvidia-smi -L
gpu_count="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
(( gpu_count > 0 ))
python3 -c 'import numpy, scipy, torch, yaml, wandb'
notify "[KCorrDiff] Stage 2 fold ${fold_id} started on ${node_name} with ${gpu_count} GPU(s)."

export WANDB_DIR="${output_dir}/wandb"
export WANDB_NAME="${RUN_ID}-fold-${fold_id}"
export WANDB_RUN_ID="${RUN_ID}-fold-${fold_id}"
export WANDB_RESUME=allow

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${gpu_count}" \
  -m kcorrdiff.training.train_stage2 \
  --source-root "${source_root}" \
  --config "${source_root}/configs/stage2-full-width.yaml" \
  --container-image-digest "${container_image_digest}" \
  --runtime-report "${runtime_report}" \
  --radar-cache-root /workspace/data/cache/radar-v1 \
  --era5-cache-root /workspace/data/cache/era5-v1 \
  --draw-manifest /workspace/data/manifests/event-pretrain-production-v1/training-draw-manifest.jsonl \
  --candidate-manifest /workspace/data/manifests/event-pretrain-production-v1/candidate-manifest.json \
  --bundle-metadata /workspace/data/manifests/event-pretrain-production-v1/bundle-metadata.json \
  --data-contract "${data_contract}" \
  --static-path /workspace/data/static/target_static.npz \
  --output-dir "${output_dir}" \
  --oof-output-dir "${output_dir}/oof" \
  --oof-remote-ssh-config /workspace/.ssh/config \
  --precision float32 \
  --disable-tf32 \
  --target-widths 64,128,256,384,512 \
  --context-widths 32,64,128,256,384 \
  --era-latent-channels 128 \
  --era-grid-size 33 \
  --per-rank-microbatch-size 12 \
  --gradient-accumulation-steps 1 \
  --num-workers 12 \
  --prefetch-factor 2 \
  --fail-on-fallback \
  --single-node-fold-worker \
  --node-name porsche \
  --phases crossfit \
  --roles "fold:${fold_id}" \
  --wandb-mode online \
  --run-id "${RUN_ID}-fold-${fold_id}" \
  --storage-reserve-gib 10

python3 "${source_root}/scripts/collect_stage2_folds.py" mark-complete \
  --run-root "${run_root}" \
  --worker-root "${output_dir}" \
  --fold-id "${fold_id}" \
  --policy-sha256 "${actual_policy_sha256}" \
  --config "${source_root}/configs/stage2-full-width.yaml" \
  --draw-manifest /workspace/data/manifests/event-pretrain-production-v1/training-draw-manifest.jsonl

notify "[KCorrDiff] Stage 2 fold ${fold_id} completed and was verified on the porsche PVC."
trap - EXIT
