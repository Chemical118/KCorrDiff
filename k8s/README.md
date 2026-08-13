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
세 Job은 기존 `saycorn-volume`만 참조하며 PVC를 생성·삭제하지
않는다. 특히 PVC 삭제는 이 workflow에 포함하지 않는다.

현재 파일:

- `rbac.yaml`: 토큰 자동 mount가 꺼진 전용 ServiceAccount
- `pvc-porsche.yaml`: porsche-local 250 GiB RWO claim
- `stager.yaml`: CPU-only PVC 통신/점검 Pod
- `porsche-gpu-shell.yaml`: 짧은 2-GPU 디버깅 전용 shell
- `train-stage2.yaml`: full-width deterministic/full-data·cross-fit OOF regression Job 초안
- `train-stage3.yaml`: full-width residual EDM 학습·독립 calibration Job 초안
- `benchmark-loader.yaml`: batch/worker 수 탐색용 1-GPU W&B benchmark Job 초안

## 향후 training CLI 계약

세 Job은 현재 아직 구현되지 않은 training CLI를 임의의 기존 명령으로 가장하지 않는다.
대신 다음 module entrypoint와 option을 구현할 것을 명시적으로 전제하며, 파일이나 입력
artifact가 없으면 GPU 작업 시작 직후 preflight에서 실패한다.

| 매니페스트 | 예상 module entrypoint | 주요 PVC 입력 | PVC 출력 |
|---|---|---|---|
| `benchmark-loader.yaml` | `kcorrdiff.training.benchmark_loader` | cache, outer-train draw manifest, static | `/workspace/benchmarks/loader/<pod>/` |
| `train-stage2.yaml` | `kcorrdiff.training.train_stage2` | cache, draw manifest, static, ERA5 | `/workspace/runs/stage2/<pod>/` (full-data/cross-fit checkpoint와 OOF) |
| `train-stage3.yaml` | `kcorrdiff.training.train_stage3` | cache, train/calibration manifest, promoted Stage 2 release | `/workspace/runs/stage3/<pod>/` (EDM checkpoint와 calibration) |

필요한 향후 config는 각각 `configs/stage2-full-width.yaml`과
`configs/stage3-full-width.yaml`이다. 공통 CLI는 `--precision`,
`--target-widths`, `--context-widths`, `--era-latent-channels`,
`--era-grid-size`, `--fail-on-fallback`을 받아야 한다. 학습 CLI는 추가로
`--require-world-size 2`를 검증해야 한다. 현재 repository에 이 module/config가
모두 생기기 전에는 세 YAML을 실제 apply하지 않는다.

`train_stage2`는 docs의 `train_regression`, `crossfit_regression`,
`build_oof_residuals`, `residual_scales`를 순서와 hash가 기록되는 하나의 fail-closed
stage로 조정하는 예상 orchestration entrypoint다. 성공한 run을 검토한 뒤에만 checkpoint,
OOF와 stage manifest를 `/workspace/releases/stage2/selected/`에 immutable release로
승격한다. `train_stage3`는 그 release를 입력으로 residual EDM을 학습하고 독립
calibration split에서 calibration artifact를 생성하는 예상 orchestration entrypoint다.
DDP orchestration 중 calibration은 rank 0 전용 실행과 rank barrier를 CLI가 책임져야 한다.

세 Job 초안은 실수로 apply해도 GPU를 잡지 않도록 기본 `spec.suspend: true`다.
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
간주하면 안 된다. Stage 2와 Stage 3 모두 초기 per-rank microbatch 1 x accumulation 4로,
2-GPU DDP에서 global effective batch 8이다. 먼저 loader benchmark 결과로 worker/batch
값을 확정하고, OOM을 성공으로 처리하거나 자동으로 작은 모델/정밀도/grid로 바꾸지 않는다.
Benchmark에서 특정 batch의 OOM을 관측값으로 기록하는 것은 허용하지만, 그 시도 안에서
모델 폭·정밀도·ERA grid를 바꾸어 성공으로 기록해서는 안 된다.

`benchmark-loader.yaml`은 host-to-device와 GPU utilization을 함께 측정하는 데 필요한
최소 1 GPU만 요청한다. 두 학습 Job은 정확히 2 GPU를 요청하며 `/dev/shm` 32 GiB를
mount한다. 각 Pod 이름을 run ID로 사용하므로 log, W&B staging, checkpoint가 서로 다른
PVC 디렉터리에 남는다. Kubernetes Secret `wandb-api`의 `WANDB_API_KEY`만 참조하고
`.env` 내용을 manifest에 복사하지 않는다.

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
```
