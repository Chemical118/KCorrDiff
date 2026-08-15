# SNU Kubernetes 실행 매니페스트

`saycorn-volume`은 250 GiB OpenEBS local RWO PVC다. 2026-08-13 사용자의 명시적
허가로 비어 있던 기존 `ferrari` claim을 삭제하고 `porsche`의 첫 소비 Pod를 통해
다시 provision했다. local PV이므로 이후 GPU 학습과 PVC 작업 Pod는 `porsche`에
고정한다.

1. `pvc-porsche.yaml`로 claim을 만들고 `porsche-gpu-shell.yaml` 또는 학습 Job을
   첫 consumer로 적용해 PV를 `porsche`에 provision한다.
2. 원자료, mmap cache, checkpoint, 로그를 `/workspace`에 보존한다.
3. CPU-only PVC Pod도 local PV 제약 때문에 `porsche`에 배치하되 GPU는 요청하지 않는다.
4. 장기 학습은 `restartPolicy: Never` Job으로 실행하고 완료 후 GPU Pod를 지운다.

로그인 확인은 저장소 상위 디렉터리에서 다음 명령을 사용한다.

```bash
npm run configure:kubectl-login
kubectl auth whoami
```

비밀값은 YAML에 넣지 않는다. `WANDB_API_KEY`는 Kubernetes Secret으로 주입한다.
Telegram의 `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`도 `.env`에만 두고
`scripts/apply_telegram_secret.py`가 두 값만 `kcorrdiff-telegram` Secret의
`bot-token`/`chat-id` key로 전달한다. 값은 명령 인자나 출력에 노출하지 않는다.
OOF overflow용 1047 HTTPS 계정과 self-signed CA는 `kcorrdiff-oof-remote` Secret의
`credentials.env`/`server.crt`로 주입하며, Pod에는 fsGroup-read-only(0440)로 mount한다.
모든 Job은 기존 `saycorn-volume`만 참조하며 PVC를 생성·삭제하지
않는다. 특히 PVC 삭제는 이 workflow에 포함하지 않는다.

현재 파일:

- `rbac.yaml`: 토큰 자동 mount가 꺼진 전용 ServiceAccount
- `pvc-porsche.yaml`: porsche-local 250 GiB RWO claim
- `stager.yaml`: porsche-local PVC를 직접 점검하는 CPU-only Pod
- `porsche-gpu-shell.yaml`: goal lifetime 동안 유지하는 독립 1-GPU Pod 두 개
  (`gpu-0`은 Running, `gpu-1`은 현재 Pending)
- `train-stage2.yaml`: 검증된 B12 fold-set을 model-only로 가져와 world2/B8 OOF와
  deployment/direct arm을 이어가는 full-width Stage 2 Job 초안
- `train-stage3.yaml`: full-width residual EDM의 `screen` / `finalists` /
  `bind-decision` 경계를 분리한 suspended Job 3개와 immutable runner ConfigMap
- `benchmark-loader.yaml`: batch/worker 수 탐색용 1-GPU W&B benchmark Job 초안
- `benchmark-production-loader.yaml`: 실제 dataset/regrid/normalization/nested
  batch/pinning/H2D 경로를 측정하는 1-GPU W&B benchmark Job 초안
- `porsche-gpu-probe.yaml`: 다른 사용자 작업으로 2 GPU 동시 확보가 어려울 때 쓰는
  짧은 단일-GPU CUDA correctness/memory probe
- `build-stage2-env.yaml`: 고정 image에 없는 SciPy/W&B를 버전 고정 PVC layer로
  원자적으로 게시하는 CPU-only 종료형 Job
- `publish-stage2-fold-source.yaml`: 로컬 source snapshot을 새 immutable PVC
  디렉터리로 받는 임시 CPU Pod
- `train-stage2-folds-porsche.yaml`: porsche에 고정한 3-completion/parallelism-2
  Indexed GPU Job과 같은 PVC를 검증하는 CPU collector

## training CLI 계약

각 Job은 표의 production module entrypoint를 직접 호출한다. 파일이나 immutable 입력
artifact가 없거나 config/CLI/runtime 계약이 다르면 GPU 학습 전에 fail-closed로 종료한다.

| 매니페스트 | 예상 module entrypoint | 주요 PVC 입력 | PVC 출력 |
|---|---|---|---|
| `benchmark-loader.yaml` | `kcorrdiff.training.benchmark_loader` | cache, outer-train draw manifest, static | `/workspace/benchmarks/loader/<pod>/` |
| `benchmark-production-loader.yaml` | `kcorrdiff.training.production_benchmark` | cache, candidate/draw/bundle manifest, normalization, coordinates, static | `/workspace/benchmarks/production-loader/<pod>/` |
| `train-stage2.yaml` | `kcorrdiff.training.train_stage2` | production cache/bundle/normalization, static, ERA5, verified B12 fold-set, OOF HTTPS Secret | `/workspace/runs/stage2/stage2-fullwidth-production-v3/` (OOF와 deployment/direct checkpoint) |
| `train-stage3.yaml` | `kcorrdiff.training.train_stage3` | cache, promoted Stage 2 release, 외부 평가 artifact | `/workspace/runs/stage3/stage3-fullwidth-v1-1-3b/` (EDM checkpoint와 선택 결정 결합 manifest) |

production config는 각각 `configs/stage2-full-width.yaml`과
`configs/stage3-full-width.yaml`이다. 공통 CLI는 `--precision`,
`--target-widths`, `--context-widths`, `--era-latent-channels`,
`--era-grid-size`, `--fail-on-fallback`을 받아야 한다. 학습 CLI는 추가로
`--require-world-size 2`를 검증한다. YAML은 suspended 상태로 먼저 생성하고,
phase별 입력 및 실제 runtime을 확인한 뒤 명시적으로 시작한다.

`train_stage2`는 docs의 `train_regression`, `crossfit_regression`,
`build_oof_residuals`, `residual_scales`를 순서와 hash가 기록되는 하나의 fail-closed
stage로 구현한 production orchestration entrypoint다. 성공한 run을 검토한 뒤에만 checkpoint,
OOF와 stage manifest를 `/workspace/releases/stage2/selected/`에 immutable release로
승격한다. `train_stage3`는 promoted Stage 2 deployment encoder를 동결한 채 residual
EDM만 학습한다. 학습 loss나 train label로 모델을 고르지 않으며, 독립 평가기가 만든
screening/final artifact가 없으면 다음 단계로 넘어가지 않는다. 최종 `bind-decision`도
선택 결과와 deployment checkpoint를 결합할 뿐 calibration을 실행하지 않는다. 독립
calibration은 생성된 `stage3-training-manifest.json`을 입력으로 받는 별도 명령에서
수행해야 한다.

## Stage 2 3-fold porsche 선행 학습

`saycorn-volume`의 local PV가 porsche에 묶여 있으므로 세 fold 모두 porsche에서만
실행한다. 각 Pod는 PVC를 직접 mount하고 A100 40 GB 한 장, microbatch 12,
accumulation 1을 사용한다. Indexed Job의 `parallelism: 2`와 namespace GPU quota로
두 fold까지 동시에 실행되고 세 번째는 같은 porsche GPU가 날 때까지 Pending이다.

각 fold는 서로 다른 worker 디렉터리에 기록한다. 학습 image에서 실행되는
`mark-complete`는 실제 draw manifest로 B12 plan을 재구성하고 checkpoint 내부
provenance/cursor/plan/training-block/model tensor와 partial manifest를 교차검증한다.
stdlib-only CPU collector는 그 strict receipt, 파일 크기/SHA-256과 세 fold 공통 lineage를
다시 확인한 뒤 hard link로 다음 immutable fold set을 게시한다. NFS, cross-node
stage-in, 서버 간 복사는 사용하지 않는다.

```text
/workspace/runs/stage2-folds-porsche-v3/assembled/fold-set-v1/fold-{0,1,2}/final.pt
/workspace/runs/stage2-folds-porsche-v3/assembled/fold-set-v1/fold-set-manifest.json
```

이 fold set은 Stage 2 전체 release가 아니다. 세 fold가 끝난 뒤
`train-stage2.yaml`은 `fold-set-manifest.json`과 sidecar hash를 명시적으로 전달한다.
importer는 collector receipt를 신뢰하지 않고 세 checkpoint를 다시 deserialize하여
world1/B12 provenance와 draw plan, 현재 model state-dict 호환성을 모두 검증한다. 또한
producer와 consumer의 config 및 source-tree identity가 정확히 같아야 한다. 이후
OOF model-only load에만 producer provenance를 사용하고, training resume의 exact
topology 검증은 그대로 유지한다. OOF inference/residual scale 및
deployment/direct mean/direct q50 arm이 성공해야 complete Stage 2가 된다.

중요: 과거 `stage2-folds-porsche-v2` checkpoint는 수정 전 spatial
geometry/advection/ERA 의미로 학습됐으므로 v3 importer가 거부하며 release 입력으로 쓰지
않는다. fixed run은 launcher가 먼저 `/workspace/code/stage2-folds-porsche-v3`를 새 immutable
snapshot으로 게시하고, fold 학습과 `train-stage2.yaml` continuation 모두 그 동일 snapshot을
사용한다. 어느 실행 중에도 snapshot이나 기존 run directory를 덮어쓰지 않는다.

v3 fold-set publication 후 `fold-set-manifest.json`의 실제 hash와 sidecar를 비교하고 내용을
검토한 뒤,
`train-stage2.yaml`의 `FOLD_SET_MANIFEST_SHA256` placeholder를 검토한 lowercase SHA-256로
교체한다. Job은 이 외부 pin, sidecar, 실제 manifest 세 값이 모두 같지 않으면 GPU model
load 전에 중단한다. 같은 writable PVC의 sidecar 값을 기대 hash로 자동 채우지 않는다.

다음 launcher가 `.env`의 Telegram 값만 Secret으로 적용하고, immutable source snapshot을
PVC에 게시하고, server-side dry-run 후 training/collector Job을 시작한다.

```bash
cd KCorrDiff
scripts/launch_stage2_folds.sh
python3 scripts/monitor_stage2_folds.py --watch
```

fold 시작/성공/실패와 세 fold 최종 검증 시 Telegram을 보낸다. 토큰은 로그나
manifest에 기록하지 않는다.

## Stage 3 실행 경계

`train-stage3.yaml`은 하나의 `List` 안에 immutable runner ConfigMap과 다음 suspended
Job 세 개를 정의한다. 세 Job을 동시에 unsuspend하지 않는다.

| 순서 | Job | 실행 전 필수 artifact | 성공 출력 |
|---|---|---|---|
| 1 | `kcorrdiff-stage3-screen-fullwidth` | complete Stage 2 release | `screening-training-manifest.json`과 EDM-A/B seed 11103 checkpoint |
| 외부 경계 | 별도 model-selection evaluator | 위 screening manifest/checkpoint | `/workspace/releases/stage3/evaluations/screening-evaluation.json` |
| 2 | `kcorrdiff-stage3-finalists-fullwidth` | 검증된 external screening evaluation | `finalist-training-manifest.json`과 A/B seed 11103/11105/11106 checkpoint |
| 외부 경계 | 별도 final evaluator | 위 finalist manifest/checkpoint | `/workspace/releases/stage3/model-selection-decision.json` |
| 3 | `kcorrdiff-stage3-bind-decision` | 검증된 external final decision | `stage3-training-manifest.json` (`calibration_required_next: true`) |

세 phase는 의도적으로 같은 `STAGE3_RUN_ID=stage3-fullwidth-v1-1-3b`와 같은 PVC
run root를 사용한다. 그래야 screening의 seed 11103 checkpoint와 rank-0 W&B run/audit를
finalists 단계가 동일 provenance로 재사용할 수 있다. phase별 Job 이름만 다르며 run ID나
output root를 한 단계에서만 바꾸면 continuation은 fail-closed로 거부된다.

promoted Stage 2 release는 self-contained `kcorrdiff.stage2-release-index.v2` 트리여야
한다. 인덱스는 Stage 3 입력 전체, 정확한 regression checkpoint 6개(fold 3개와
deployment/direct-mean/direct-q50), B12 fold import를 사용했다면 원본 fold-set
manifest와 partial manifest 3개를 모두 release root 상대 경로와 SHA-256으로 고정한다.
Stage 3는 Stage 2 manifest 안의 원래 run 절대 경로를 다시 열지 않는다.

```text
/workspace/releases/stage2/selected/release-index.json
/workspace/releases/stage2/selected/release-index.sha256
# release-index.json이 참조하는 모든 파일/디렉터리도 selected/ 아래에 존재
```

`release-index.sha256`는 `release-index.json`의 lowercase SHA-256 64자리와 newline만
담는다. runner는 sidecar와 실제 파일 hash를 먼저 비교한 뒤 인덱스의 containment,
symlink 금지, 파일 크기/hash, artifact lineage와 checkpoint role을 전부 재검증한다.
release tree를 다른 mount root로 복사해도 인덱스 hash와 checkpoint-set hash는 변하지
않는다. 또한 첫 phase에서 Stage 3
source-tree identity와 CUDA/PyTorch runtime report를 원자적으로 기록하고, 후속 phase는
두 identity가 조금이라도 달라지면 실행을 중단한다. 컨테이너 image는 immutable digest와
그 digest를 전달하는 `KCORRDIFF_CONTAINER_IMAGE_SHA256`가 일치해야 한다.

처음에는 immutable ConfigMap 하나와 suspended Job 세 개를 생성한다.

```bash
kubectl apply -f k8s/train-stage3.yaml
kubectl get jobs -l app.kubernetes.io/component=stage3-diffusion-training
```

각 외부 경계의 artifact와 hash를 검토한 뒤 해당 Job 하나만 시작한다.

namespace quota는 memory 128 GiB, GPU 2개다. `kcorrdiff-stager`의 24 GiB limit과
Stage 3 Job 하나의 104 GiB limit이 합계 128 GiB로 정확히 quota를 채우므로 memory
headroom은 없다. 또한 Pending Pod도 quota를 점유한다. 따라서 어느 phase든 시작하기
전에 `porsche-gpu-0`과 `porsche-gpu-1`을 **둘 다** 삭제하고, 다른 GPU Job이 없는지
확인한다. stager는 PVC를 유지한 채 함께 둘 수 있다.

```bash
kubectl delete pod porsche-gpu-0 porsche-gpu-1
kubectl get resourcequota ws-quotas
```

```bash
kubectl patch job kcorrdiff-stage3-screen-fullwidth \
  --type=merge -p '{"spec":{"suspend":false}}'

# external screening-evaluation.json 검토 후
kubectl patch job kcorrdiff-stage3-finalists-fullwidth \
  --type=merge -p '{"spec":{"suspend":false}}'

# external model-selection-decision.json 검토 후
kubectl patch job kcorrdiff-stage3-bind-decision \
  --type=merge -p '{"spec":{"suspend":false}}'
```

각 Job은 `backoffLimit: 0`, `restartPolicy: Never`, 정확히 2 GPU/NCCL/FP32/no-TF32,
full-width/no-fallback 계약을 사용한다. `bind-decision`도 현재 production CLI의 exact
two-rank runtime 및 Stage 2 lineage를 다시 검증하므로 2 GPU Job이다. 완료된 phase를
다시 실행하거나 기존 immutable ConfigMap/Job을 덮어쓰지 말고, 재실행이 필요하면 원인을
기록한 뒤 새 versioned run ID와 resource 이름을 사용한다.

학습 Job은 실수로 apply해도 GPU를 잡지 않도록 기본 `spec.suspend: true`다.
모든 입력과 CLI 계약을 확인하고 batch/worker 값을 확정한 뒤 manifest에서 이를
`false`로 바꾸어 새 Job 이름으로 적용한다. 이미 생성한 Job을 즉석에서 재사용해
연구 run provenance를 흐리지 않는다.

초안의 full-width 계약은 다음과 같이 명시되어 있다.

```text
target widths  = 64,128,256,384,512
context widths = 32,64,128,256,384
ERA latent     = 128 channels on the native 33 x 33 grid
compute        = float32, TF32 disabled
fallback       = CPU/model-width/precision/ERA-grid automatic fallback forbidden
```

환경변수는 이 계약을 한 번 더 전달하지만, CLI 구현은 config 및 실제 runtime 상태와
대조한 뒤 불일치 시 non-zero로 종료해야 한다. 환경변수만 보고 full-width 실행이라고
간주하면 안 된다. Stage 2는 power-of-two-only 실측으로 per-rank microbatch 8 x
accumulation 1을 선택해 2-GPU DDP global effective batch 16으로 고정했다. Stage 3의
초기값은 per-rank microbatch 1 x accumulation 4, global effective batch 8이다. 먼저 loader benchmark 결과로 worker/batch
값을 확정하고, OOM을 성공으로 처리하거나 자동으로 작은 모델/정밀도/grid로 바꾸지 않는다.
Benchmark에서 특정 batch의 OOM을 관측값으로 기록하는 것은 허용하지만, 그 시도 안에서
모델 폭·정밀도·ERA grid를 바꾸어 성공으로 기록해서는 안 된다.

Radar/ERA payload의 전체 SHA-256 검증은 cache atomic publication 직후 CPU stager에서
한 번 수행한다. 학습 Job에서 `--verify-cache-hashes`를 켜면 worker별 lazy cache open마다
약 72 GiB payload를 다시 hash해 I/O를 중복하므로 사용하지 않는다. 학습 시작 시에는
고정된 cache manifest, timestamp index, 좌표, static, normalization 및 selection artifact
hash를 계속 검증한다.

## OOF PVC hot tier와 1047 overflow

Stage 2 OOF는 float32 두 field를 lossless byte-shuffle+DEFLATE shard로 기록한다. PVC는
primary hot tier이며, 다음 shard의 최악 크기와 후속 checkpoint 용량까지 고려해
최종 10 GiB free-space가 보존되도록 OOF 중에는 14 GiB를 예약한다. 임계점에 닿으면
가장 오래된 sealed shard 하나만 `https://168.188.119.187:1047/kcorrdiff/oof`로 PUT한다.
서버에서 전체 파일을 다시 GET해 SHA-256/byte 수가 일치하고 durable receipt가 게시된
뒤에만 PVC 사본을 지운다. 전송 실패·read-back 불일치 때는 로컬 shard를 유지하고
학습을 fail-closed로 중단한다.

최종 OOF manifest는 각 shard를 `local` 또는 `remote_https`로 명시한다. Stage 3는
원격 shard가 필요할 때 worker별 `/tmp/oof-remote-cache`에 하나만 받아 SHA-256을
검증하고, FP32 bitwise 복원 후 압축 파일을 즉시 지운다. 따라서 전체 OOF를 PVC에
재복제하지 않는다. 실제 porsche→1047 smoke에서 v3 shard upload, full HTTPS read-back,
로컬 삭제, 재다운로드 및 uint32 bit-pattern roundtrip을 모두 확인했다.

`benchmark-loader.yaml`은 host-to-device와 GPU utilization을 함께 측정하는 데 필요한
최소 1 GPU만 요청한다. Stage 2 학습과 Stage 3의 세 phase Job은 정확히 2 GPU를 요청하며
`/dev/shm` 32 GiB를 mount한다. 일반 Job은 Pod 이름을 run ID로 사용하지만 Stage 3 세
phase만 exact continuation을 위해 같은 고정 run ID와 PVC 경로를 공유한다. Kubernetes
Secret `wandb-api`의 `WANDB_API_KEY`만 참조하고 `.env` 내용을 manifest에 복사하지 않는다.

현재 manifest의 image digest는 저장소의 기존 PyTorch runtime과 맞춘 초안 값이다.
실행 전 같은 CUDA/PyTorch 계열의 immutable project training image digest로 교체하고,
그 image가 `numpy`, `scipy`, `torch`, `PyYAML`, `wandb` 및 필요한 data extra를 포함하는지
검증한다. Job 시작 중 인터넷에서 가변 dependency를 설치하지 않는다. read-only root에서
JIT/cache가 실패하지 않도록 비영속 cache만 `/tmp` emptyDir에 둔다.

적용 전에 `kubectl diff -f <file>`로 현재 클러스터와 비교한다. 목표가 끝나면
`porsche`, `kcorrdiff-stager`를 삭제해 GPU와 PVC writer를 해제한다.

초안의 schema만 검증할 때는 실제 apply 대신 다음을 사용한다.

```bash
python -c 'import pathlib, yaml; [list(yaml.safe_load_all(p.read_text())) for p in pathlib.Path("k8s").glob("*.yaml")]'
kubectl apply --dry-run=client -f k8s/benchmark-loader.yaml
kubectl apply --dry-run=client -f k8s/train-stage2.yaml
kubectl apply --dry-run=client -f k8s/train-stage3.yaml
kubectl apply --dry-run=server -f k8s/train-stage3.yaml
```
