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
OOF overflow는 PVC의 `/workspace/.ssh/config`와 개인키를 사용해
`hyunwoo-home:/hyunwoo/kcorrdiff/oof`로 rsync한다. SSH 파일은 UID 1035 소유,
config와 개인키는 0600으로 유지한다.
모든 Job은 기존 `saycorn-volume`만 참조하며 PVC를 생성·삭제하지
않는다. 특히 PVC 삭제는 이 workflow에 포함하지 않는다.

로컬 KCorrDiff 소스만 기존 `/workspace/KCorrDiff`에 반영할 때는 데이터 전체를
다시 스테이징하지 않고 다음 명령을 사용한다. `.git`, 캐시, `.env`는 복사하지 않으며
data, runs, logs, checkpoint 경로는 변경하지 않는다.

```bash
scripts/update_pvc_source.sh
```

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

각 Job은 표의 module entrypoint를 직접 호출한다. 필요한 파일과 과학적
shape/dtype/split 계약은 확인하되, config·artifact hash와 GPU topology는 실행 게이트로
쓰지 않는다.

| 매니페스트 | 예상 module entrypoint | 주요 PVC 입력 | PVC 출력 |
|---|---|---|---|
| `benchmark-loader.yaml` | `kcorrdiff.training.benchmark_loader` | cache, outer-train draw manifest, static | `/workspace/benchmarks/loader/<pod>/` |
| `benchmark-production-loader.yaml` | `kcorrdiff.training.production_benchmark` | cache, candidate/draw/bundle manifest, normalization, coordinates, static | `/workspace/benchmarks/production-loader/<pod>/` |
| `train-stage2.yaml` | `kcorrdiff.training.train_stage2` | production cache/bundle/normalization, static, ERA5, B12 fold-set, OOF HTTPS Secret | `/workspace/runs/stage2/stage2-fullwidth-research-v4/` (OOF와 deployment/direct checkpoint) |
| `train-stage3.yaml` | `kcorrdiff.training.train_stage3` | cache, promoted Stage 2 release, 외부 평가 artifact | `/workspace/runs/stage3/stage3-fullwidth-v1-1-3b/` (EDM checkpoint와 선택 결정 결합 manifest) |

production config는 각각 `configs/stage2-full-width.yaml`과
`configs/stage3-full-width.yaml`이다. 공통 CLI는 `--precision`,
`--target-widths`, `--context-widths`, `--era-latent-channels`,
`--era-grid-size`, `--fail-on-fallback`을 받는다. world size는 Pod에 보이는 GPU 수에서
계산하며 어떤 양의 값도 허용한다. YAML은 suspended 상태로 먼저 생성하고, phase별
입력 및 실제 runtime을 확인한 뒤 명시적으로 시작한다.

`train_stage2`는 docs의 `train_regression`, `crossfit_regression`,
`build_oof_residuals`, `residual_scales`를 하나의 orchestration entrypoint로 구현한다.
hash와 크기는 재현성 메타데이터로 기록할 뿐 실행을 막지 않는다. 성공한 run을 검토한
뒤 checkpoint, OOF와 stage manifest를 `/workspace/releases/stage2/selected/`에 release로
승격한다. `train_stage3`는 promoted Stage 2 deployment encoder를 동결한 채 residual
EDM만 학습한다. 학습 loss나 train label로 모델을 고르지 않으며, 독립 평가기가 만든
screening/final artifact가 없으면 다음 단계로 넘어가지 않는다. 최종 `bind-decision`도
선택 결과와 deployment checkpoint를 결합할 뿐 calibration을 실행하지 않는다. 독립
calibration은 생성된 `stage3-training-manifest.json`을 입력으로 받는 별도 명령에서
수행해야 한다.

## Stage 2 3-fold porsche 선행 학습

`saycorn-volume`의 local PV가 porsche에 묶여 있으므로 세 fold 모두 porsche에서만
실행한다. 각 Pod는 PVC를 직접 mount하고 현재 manifest에서는 GPU 한 장, microbatch 12,
accumulation 1을 사용한다. Indexed Job의 `parallelism: 2`와 namespace GPU quota로
두 fold까지 동시에 실행되고 세 번째는 같은 porsche GPU가 날 때까지 Pending이다.

각 fold는 서로 다른 worker 디렉터리에 기록한다. 학습 image에서 실행되는
`mark-complete`는 실제 draw manifest로 B12 plan을 재구성하고 checkpoint 내부
cursor/plan/training-block/model tensor와 partial manifest를 검증한다.
stdlib-only CPU collector는 fold 역할과 tensor 호환성을 확인하고 일반 복사로 fold set을
게시한다. hash, 파일 크기와 producer topology는 정보로만 남긴다. NFS, cross-node
stage-in, 서버 간 복사는 사용하지 않는다.

```text
/workspace/runs/stage2-folds-porsche-v3/assembled/fold-set-v1/fold-{0,1,2}/final.pt
/workspace/runs/stage2-folds-porsche-v3/assembled/fold-set-v1/fold-set-manifest.json
```

이 fold set은 Stage 2 전체 release가 아니다. 세 fold가 끝난 뒤
`train-stage2.yaml`은 `fold-set-manifest.json`을 전달한다. importer는 checkpoint를 다시
deserialize하여 fold 역할, 완료 cursor, training-block 할당과 현재 model state-dict
호환성을 검증한다. producer/consumer hash와 topology가 달라도 허용한다. OOF inference,
residual scale 및
deployment/direct mean/direct q50 arm이 성공해야 complete Stage 2가 된다.

중요: 과거 `stage2-folds-porsche-v2` checkpoint는 수정 전 spatial
geometry/advection/ERA 의미로 학습됐으므로 v3 importer가 거부하며 release 입력으로 쓰지
않는다. 활성 `/workspace/code/stage2-folds-porsche-v3` snapshot은 그대로 보존하고,
새 코드는 `/workspace/code/stage2-research-flex-v4` 같은 별도 versioned snapshot에
게시한다. 어느 실행 중에도 기존 snapshot이나 run directory를 덮어쓰지 않는다.

v3 fold-set publication 후 `fold-set-manifest.json`의 fold coverage와 역할을 검토한다.
sidecar hash가 있으면 provenance 참고 자료로 보존하지만 launcher 입력이나 실행 허가
조건으로 사용하지 않는다.

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
finalists 단계가 재사용할 수 있다. phase별 Job 이름만 다르며 같은 run ID와 output
root를 쓰면 기존 checkpoint를 이어서 사용한다.

promoted Stage 2 release는 self-contained `kcorrdiff.stage2-release-index.v2` 트리여야
한다. 인덱스는 Stage 3 입력 전체, 정확한 regression checkpoint 6개(fold 3개와
deployment/direct-mean/direct-q50), B12 fold import를 사용했다면 원본 fold-set
manifest와 partial manifest를 함께 기록한다. 경로는 이동 가능하며 SHA-256은 정보다.
Stage 3는 Stage 2 manifest 안의 원래 run 절대 경로를 다시 열지 않는다.

```text
/workspace/releases/stage2/selected/release-index.json
/workspace/releases/stage2/selected/release-index.sha256
# release-index.json이 참조하는 모든 파일/디렉터리도 selected/ 아래에 존재
```

`release-index.sha256`가 있으면 provenance 참고 자료로 보존한다. runner는 참조 파일의
존재와 checkpoint 역할 coverage를 확인하지만 hash, containment, symlink 여부를 실행
게이트로 쓰지 않는다. CUDA/PyTorch runtime report도 관찰 메타데이터로 기록한다.

처음에는 immutable ConfigMap 하나와 suspended Job 세 개를 생성한다.

```bash
kubectl apply -f k8s/train-stage3.yaml
kubectl get jobs -l app.kubernetes.io/component=stage3-diffusion-training
```

각 외부 경계의 artifact 내용을 검토한 뒤 해당 Job 하나만 시작한다.

namespace quota와 현재 여유 GPU를 확인한 뒤 phase를 시작한다. manifest는 GPU 한 장을
요청하며, 더 많은 GPU를 요청하는 변형에서도 entrypoint가 보이는 GPU 수를 자동으로
사용한다. Pending Pod도 quota를 점유한다.

```bash
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

각 Job은 `backoffLimit: 0`, `restartPolicy: Never`, FP32/no-TF32 설정을 사용한다.
GPU 수는 고정하지 않으며 global batch metadata와 실제 topology가 다르면 경고만 남긴다. 완료된 phase를
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

이 값들은 현재 실험의 시작점이며 다른 양의 batch/worker/grid 설정으로 바꿀 수 있다.
Stage 2의 예시는 per-rank microbatch 8 x accumulation 1이다. Stage 3의
초기값은 per-rank microbatch 1 x accumulation 4, global effective batch 8이다. 먼저 loader benchmark 결과로 worker/batch
값을 확정하고, OOM을 성공으로 처리하거나 자동으로 작은 모델/정밀도/grid로 바꾸지 않는다.
Benchmark에서 특정 batch의 OOM을 관측값으로 기록하는 것은 허용하지만, 그 시도 안에서
모델 폭·정밀도·ERA grid를 바꾸어 성공으로 기록해서는 안 된다.

Radar/ERA payload의 SHA-256은 cache key와 provenance로만 쓴다. 학습 Job에서 live
payload를 재해시하거나 expected hash와 비교하지 않는다.

## OOF PVC hot tier와 rsync overflow

Stage 2 OOF는 float32 두 field를 lossless byte-shuffle+DEFLATE shard로 기록한다.
각 sealed shard는 생성 즉시 `hyunwoo-home:/hyunwoo/kcorrdiff/oof`로 rsync하므로
전송 시간이 OOF inference 사이에 분산된다. 중단된 전송은 partial 파일에서 이어가며,
원격 파일 크기와 durable receipt를 확인한다. PVC 사본은 EDM-A의 hot-read tier로
유지하고, 14 GiB headroom을 지키기 위해 필요할 때만 이미 mirror된 오래된 shard부터
지운다. 전송 실패 때는 로컬 shard를 유지하고 오류를 보고한다.

최종 OOF manifest는 각 shard를 `local` 또는 `remote_rsync`로 명시한다. Stage 3는
원격 shard가 필요할 때 worker별 `/tmp/oof-remote-cache`에 하나만 받아 크기와 FP32
shape/dtype을 확인한 뒤 압축 파일을 즉시 지운다. 따라서 전체 OOF를 PVC에
재복제하지 않는다. 기존 `remote_https` manifest는 읽기 호환성을 유지한다.

`benchmark-loader.yaml`은 host-to-device와 GPU utilization을 함께 측정하는 데 필요한
최소 1 GPU만 요청한다. Stage 2 학습과 Stage 3의 세 phase Job도 기본 1 GPU를 요청하며
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
