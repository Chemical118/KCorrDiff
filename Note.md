# K-CorrDiff 구현 기록

이 저장소는 `docs/k_corrdiff_architecture_v1_1_3b.md`의 전체 구조를
`CPrecNet event-conditioned pretraining + ERA5 full-trajectory oracle` 연구 트랙으로
구현한다. CPrecNet archive에는 연속 dry timeline과 동적 pixel validity가 없으므로,
여기서 얻은 occurrence 확률을 운영 강수 base rate나 continuous KMA HSR 성능으로
해석하지 않는다.

## Stage 1 — 데이터 계약, manifest, 고속 입력 기반

### 고정한 계약

- radar archive의 KST timestamp를 aware UTC issue time으로 바꾸고, 모든 샘플을
  `(t0, tau, condition_signature)`로 식별한다. 리드는 0.5–6시간의 30분 간격 12개다.
- 30분 target은 정확히 7개의 5분 순간 강수강도를 trapezoid 적분한다. 전체 scan
  부재와 pixel invalid를 구분하고, 미래 `M_target_tau`는 label/loss에만 보관한다.
- CPrecNet normalized reflectivity를 먼저 선형 `mm h-1`로 복원한다. `0.1 mm` censor,
  `z=log1p(A_model/1 mm)`, wet label의 일관성을 한 helper에서 만든다.
- 로컬 KMA HSR 포맷 문서와 샘플을 대조해 little-endian int16, 1024-byte header,
  `-30000/-25000` null sentinel을 고정했다. `-20000`은 유효한 표시 하한이므로
  결측으로 처리하지 않는다.
- ERA5는 23개 instantaneous field와 1-hour-end `tp`를 분리한다. `tp`는
  `(valid_time-1h, valid_time]`, `stepType=accum`, metre 단위를 검증한 뒤 float32
  millimetre로 변환한다. 8개 native-hour token에는 data validity, trajectory 범위,
  intentional temporal access mask를 각각 둔다.
- 실제 2020–2025 ERA5 216개 GRIB을 ecCodes로 전수 검사했다. 총 1,262,592개
  message에서 각 24개 field가 52,608 UTC hour씩 정확히 한 번 존재했고, 33×33
  grid, 북→남 원 위도축, `tp`의 `accum`/metre/정확한 1시간 interval이 모두
  계약과 일치했다. Message order나 filename을 시간축으로 사용하지 않는다.
- 좌표는 target 중심 LCC와 100 km scale을 공통으로 사용한다. Context는 선형
  rain-rate 공간에서 sparse separable area-integrated operator로 regrid하고 mean,
  local maximum/nearest detail, 유효도와 geometry confidence를 별도 채널로 만든다.

### Manifest와 leakage 방지

Target/condition 8쌍의 Dataverse MD5를 다시 계산해 manifest와 모두 일치함을
확인했고, 138,236개 timestamp key가 쌍마다 정확히 일치했다. 전체 key는 939개
연속 run, 77,444개 strict 6-hour issue window, 1,190개 비중첩 84-frame block lower
bound를 만든다.

CPrecNet은 애초 event-conditioned selection이므로 존재하지 않는 dry frame을
합성하지 않았다. 12-hour merge와 ±7-hour dependency guard를 적용한 conservative
event block으로 chronological split을 먼저 고정했다. 생성한
compact event index는 77,393 issue time, 928,716 `(t0,tau)` item, 278 event
block을 포함하며 split boundary로 인한 embargo 비율은 약 0.066%다. Sampling은
counter-based global draw index로 고정해 worker/GPU/batch 변경이 draw sequence를
바꾸지 않으며, 각 row에 `P_target`, `P_draw`, `omega=P_target/P_draw`를 기록한다.
OOF artifact는 전체 eligible timeline을 dense하게 쓰지 않고 실제 bounded draw에서
고유한 row만 shard하며, 쓰기 전에 byte budget을 강제한다.

Compact Parquet index를 입력으로 12개 lead를 전부 논리 확장하는 training bundle CLI도
구현했다. Split별 `P_target`은 eligible item-uniform이고, `P_draw`는 uniform 또는
event-block-balanced를 명시적으로 선택한다. Outer-train event block은 deterministic
3-fold로 배정하며 candidate와 draw row에 `fold_id`를 보존한다. Bundle은 독립적으로
재계산한 eligible sample-ID universe의 누락률이 0인지 확인한 뒤 config, source index,
candidate universe, fold map과 draw manifest의 SHA-256 및 OOF byte preflight를 한 atomic
artifact에 결박한다. 실제 928,716-item Parquet 재사용 smoke에서는 outer-train
431,400 item, fold별 43/48/49 block, unassigned fraction 0을 확인했다. PVC에 생성한
초기 training bundle은 8,192 draw 중 7,789개 고유 OOF item을 포함하며 float32
2-field dense 상한은 4,083,679,232 byte다. Metadata SHA-256의 확인된 prefix는
`243ee0…`이며 source index, candidate, fold map과 draw artifact hash를 모두 다시
읽어 검증했다.

CLI 책임은 다음처럼 분리했다.

- `kcorrdiff-build-event-index`: raw target/condition에서 compact Parquet index만 생성
- `kcorrdiff-build-manifest`: Parquet index 재사용 또는 raw 입력으로 candidate/draw
  training bundle 생성
- `kcorrdiff-audit-manifest`: 이미 생성된 candidate manifest의 leakage/probability 감사
- `kcorrdiff-build-cache`, `kcorrdiff-era5-cache`: radar 및 ERA5 mmap cache 작업

이 단계의 모집단은 CPrecNet event-conditioned archive support로 한정된다. Continuous
KMA timeline의 context-active/significant-component seed, UTC-day strict-dry 및
marginal/background 전수 partition, `rho_invalid_max` QC와 season/regime/intensity·ESS
보고는 존재하지 않는 CPrecNet dry frame으로 대체하지 않았다. 또한 모델이 필요한
label-free inference smoke benchmark는 Stage 2/3 모델 구현 전이므로 아직 완료 증거로
주장하지 않는다.

### 250 GiB PVC의 입력 레이아웃

기존 `saycorn-volume`은 `ferrari` local PV였고 점검 Pod에서 파일이 없음을 확인했다.
사용자의 명시적 허가 후 이를 삭제하고, `porsche`에 고정된 새 250 GiB RWO PVC를
만들었다. 원자료는 한 번만 `/workspace/data/raw`에 두고 release마다 복제하지 않는다.

압축 NPZ는 timestamp 하나마다 DEFLATE member를 여는 구조라 학습 중 random access가
병목이다. 따라서 source archive 단위의 target/condition float32 `.npy` 16개와 불변
timestamp index로 변환했다. 결과는 약 68 GiB이고 persistent DataLoader worker가
lazy mmap으로 열어 kernel page cache를 공유한다. 16개 shard에서 첫/중간/끝/난수
112 frame을 원본 NPZ와 비교해 dtype과 값이 bitwise equal임을 확인했다. Batch sampler는
동일한 `(t0, condition_signature)`의 여러 lead를 묶어 12-frame history page를 재사용한다.

ERA5도 raw GRIB을 학습 중 해석하지 않고, 2020–2025 연도별로 instantaneous
`[T,23,33,33]`와 `tp [T,1,33,33]` float32 mmap을 분리한다. 이렇게 해야 미래 `tp`를
잘못 읽거나 GRIB message order를 시간축으로 오인하지 않으면서 월/연도 경계 window를
absolute UTC valid time으로 읽을 수 있다.

Production ERA5 cache도 incomplete staging에서 전수 decode한 뒤 atomic rename으로
게시했다. 2020–2025 연도별 12개 float32 field shard와 validity/좌표/manifest를 합친
27개 파일은 5,499,984,038 logical byte이며, manifest SHA-256은
`77f1387c29f1783ab9cf739651c3fef46e7e28b1e7e5cdad2a9f722a8e335bf3`이다. 모든
52,608 UTC hour에서 instantaneous/tp validity가 참이고 모든 shard SHA-256을 다시
검증했다. 연말·월말·윤년 경계 24시각과 seed 11103 난수 12시각의 raw GRIB을 ecCodes와
provider adapter로 다시 읽어 864 field, 940,896 float32 값을 mmap과 uint32 bit pattern으로
대조했으며 mismatch 0, 최대 절대오차 0이었다. Raw/cache sample SHA-256은 둘 다
`44940062bfa4022bf8ea247201b90c407d6ec938a176fb5850c3e1ed3a826980`였다.

Dataset condition 경계도 end-to-end로 고정했다. Context 12장은 한 번에 선형 강수율로
복원한 뒤 동일 regrid operator를 거치고, ERA5 window는 24채널 값과 네 종류 mask,
presence flag, provenance를 함께 전달한다. Static 입력은 명명된 channel schema로 만들며
`target_validity`, `M_target`, 미래 wet/z 같은 이름은 condition에서 거부한다. 완성된
issue-time condition 뒤에만 미래 7장을 읽어 label을 만들도록 호출 순서도 테스트한다.

### Kubernetes 실행 파일

`k8s/`에는 porsche-local PVC, CPU stager, 짧은 2-GPU shell, RBAC/secret 경계,
loader benchmark와 Stage 2/3 종료형 Job을 둔다. 중요한 log/checkpoint/cache는 모두
`/workspace`에 기록하고 `/dev/shm`은 memory-backed `emptyDir`로 mount한다. 비밀값은
manifest에 저장하지 않고 Kubernetes Secret에서 주입한다.

### Stage 1 검증

- 전체 unit test: `pytest -q` → `141 passed`
- Python import/bytecode: `python -m compileall -q kcorrdiff tests scripts`
- config/Kubernetes 11개 JSON·YAML·TOML parse, 5개 console alias help 계약,
  source/test/config/k8s trackability와 root-anchored runtime ignore 확인
- `git diff --check`
- Dataverse archive 16개 MD5 재계산 및 target/condition key equality
- radar mmap 16개 shard/112개 원본 frame bitwise parity
- ERA5 216개 파일/1,262,592개 message의 ecCodes semantic completeness audit
- ERA5 mmap 6년 shard SHA-256, all-hour mask 및 raw GRIB 36시각 bitwise parity
- 실제 NON_UNI 좌표에서 sparse regrid affine reproduction 최대 오차
  `7.11e-14`

## Stage 2 — full-width regression/cross-fit/OOF 구현 및 학습 준비

### 모델과 학습 경계

- Target/Context temporal encoder, ERA encoder, physical cross-attention,
  causal advection, hurdle occurrence/positive-amount regression과 direct
  physical mean/q50 arm을 원 설계 폭 그대로 구현했다. Production hurdle
  system은 62,008,276 trainable FP32 parameter이며 TF32와 모든 width/grid/
  precision fallback을 금지한다.
- 3-fold grouped cross-fit, deployment/direct checkpoint, exact global weighted
  SUM gradient, rank padding, atomic resume/RNG, OOF inference, residual scale과
  complete manifest를 하나의 two-rank orchestration으로 묶었다. Future target
  validity와 label은 loss/target 경계 밖으로 나오지 않는다.
- A100-40GB에서 2의 제곱 batch만 검사했다. B=8/rank, workers=12,
  prefetch=2를 선택했고 B=16은 fit하지만 global batch와 memory headroom 때문에
  채택하지 않았으며 B=32는 OOM으로 거부됐다. B=8 full-width 32-step endurance는
  5.72 sample/s, peak allocated/reserved 18.68/21.21 GB로 fallback 없이 끝났고
  W&B run은 `kcorrdiff-stage2-b8-endurance-20260813-1`이다.

### Production draw와 OOF 저장

- 초기 8,192 draw를 최종 학습으로 오인하지 않도록 outer-train 431,400 item을
  seed 11103 hash shuffle로 정확히 한 번 방문하는 no-replacement production
  bundle을 별도 게시했다. Condition signature는 label과 독립적으로 frozen
  50/25/25 정책(ERA+tp / ERA tp-off / whole-ERA null)에서 하나를 선택한다.
- Production candidate/draw/bundle SHA-256은 각각 `4f3210…0566`,
  `839071…03c`, `c09ed5…b2de`다. Train-only normalization은 전체 cache payload
  hash를 다시 검증한 artifact만 production factory가 수용한다.
- 431,400개의 두 FP32 OOF field 논리 크기는 226,177,843,200 byte다. Shard는
  bitwise-lossless byte-shuffle+DEFLATE로 기록하고 총 compressed cap/ratio를
  계속 강제한다. PVC는 hot tier로 쓰되 OOF 뒤 checkpoint까지 고려해 14 GiB를
  남긴다. 임계점에서는 oldest sealed shard 하나를 인증된 1047 HTTPS 서버로
  전송하고, 전체 GET SHA-256이 일치한 durable receipt 뒤에만 로컬 파일을 지운다.
  Stage 3는 원격 shard 하나만 bounded cache에 받아 검증·복원 후 즉시 지운다.
- 실제 porsche→1047 smoke에서 OOF v3 upload/read-back/local-delete/re-download와
  uint32 bit-pattern equality를 확인했다. 비밀번호는 git/YAML에 넣지 않고
  `.env`와 CA를 `kcorrdiff-oof-remote` Kubernetes Secret으로 0440 mount한다.

### 현재 검증과 실행 상태

- 전체 repository suite: `474 passed, 7 skipped`(CUDA/pin allocator host-only skip)
- OOF/remote/Stage 3 lazy reader 집중 회귀, compile, `git diff --check`, 모든
  Stage 2/3 YAML parse와 embedded bash `bash -n`, Kubernetes server dry-run을 통과했다.
- porsche에서 A100 두 장을 동시에 바로 확보할 수 없어 Stage 2의 3-fold cross-fit을
  하나의 2-GPU 작업으로 기다리지 않고, fold 0/1/2를 각각 독립적인 1-GPU 학습으로
  나눴다. `Indexed Job(completions=3, parallelism=2)`이 각 fold 번호를 관리하며,
  GPU가 비는 순서대로 같은 porsche에서 실행한다. 당시 실제 상태는 fold 0 Running,
  fold 1 GPU Pending이고, fold 2는 앞 index가 끝난 뒤 Job controller가 생성하는
  대기 상태였다.
- 각 fold는 porsche A100-40GB 한 장에서 microbatch 12, accumulation 1로 학습한다.
  이는 기존 2-GPU B8/global batch 16과 동일한 effective batch를 강제한 구성이 아니라,
  사용자가 선택한 single-node fold 실행 정책(global effective batch 12)이다. 정책 hash와
  실제 topology를 checkpoint, partial manifest와 W&B config에 기록해 다른 topology의
  checkpoint를 잘못 resume하지 못하게 한다.
- `saycorn-volume`이 porsche-local RWO PVC이므로 세 worker와 CPU collector가 같은
  porsche에서 PVC를 직접 mount한다. NFS나 서버 간 입력 복사는 사용하지 않는다.
  fold별 출력 디렉터리를 분리하고, collector가 세 checkpoint와 partial manifest의
  SHA-256을 검증한 뒤 hard link 기반 immutable fold set을 조립하고 Telegram으로
  완료를 알린다. 3-fold 이후 OOF inference/residual scale 및 deployment/direct arm은
  별도 후속 실행으로 남는다.

## Stage 3 — residual EDM, 독립 평가 및 calibration 경계

### 모델·데이터 계약

- Stage 2의 complete manifest, 세 fold checkpoint, deployment checkpoint, 최종 OOF,
  residual scale 및 다섯 launch-identity hash를 모두 검증한 뒤에만 Stage 3 data
  factory를 연다. Partial OOF나 partial residual state는 입력으로 허용하지 않는다.
  OOF v3 remote shard는 worker-local bounded cache에 하나씩 받아 SHA-256과 lossless
  float32 복원을 확인하고 사용 직후 삭제한다.
- Stage 3 config도 초기 8,192-row artifact가 아니라 production 431,400-row bundle과
  frozen condition augmentation semantic SHA-256 `813d73…9c9f8`에 결박했다. 각 draw
  row의 ERA+tp, ERA tp-off, whole-ERA null signature를 그대로 재생하며 Stage 2와
  signature/hash가 하나라도 다르면 실행을 중단한다.
- EDM-A는 Stage 2 deployment pyramid를 제외하고, EDM-B는 detach한 deployment
  pyramid를 condition으로 사용한다. 두 모델 모두 residual state와 `(mu_z,p)`만
  입력받고 diffusion-owned adapter/QKV/gate, L3/L4 physical attention, sigma-independent
  source KV cache와 exact source-absence 경로를 사용한다. FP32 parameter 수는
  EDM-A 52,111,457개(약 198.8 MiB), EDM-B 52,605,025개(약 200.7 MiB)이며 정밀도나
  원래 폭을 줄이지 않는다.

### 학습·평가·게시 경계

- Exact two-rank NCCL, FP32/no-TF32, global masked-EDM numerator/denominator, manual
  gradient SUM, accumulation padding, stateless Philox noise, atomic optimizer/scheduler/
  rank별 RNG checkpoint와 W&B resume를 구현했다. Per-rank batch는 양의 2의 제곱만
  허용하며 현재 production 시작값은 B=1, accumulation=4, global batch=8이다.
- 실행은 `screen`, `finalists`, `bind-decision` 세 개의 suspended Job으로 분리했다.
  Seed 11103 EDM-A/B screening 뒤 외부 model-selection evidence 없이는 finalist를
  학습할 수 없고, 외부 frozen decision 없이는 final 선택을 게시할 수 없다.
  Finalist는 seed 11103/11105/11106을 사용한다. Calibration은 선택이 동결된 뒤
  별도 artifact layer에서만 수행하며 training loss로 후보를 고르는 경로는 없다.
- Sampling은 fixed Karras rho-7 schedule과 deterministic Heun을 사용하며 4×6,
  16×8, 32×12 및 distilled 8×4 profile을 고정했다. Calibration artifact는 b/c,
  optional d, spread gamma, monotone probability map과 pooling provenance를 immutable
  canonical JSON/SHA-256으로 게시한다.

### Stage 3 검증 및 현재 상태

- 전체 repository suite: `474 passed, 7 skipped`; host에서 skip된 CUDA 전용 7개
  경로를 porsche A100에서 재실행해 모두 통과했다. Triton/Torch cache는 read-only
  root가 아닌 `/tmp`로 고정한다.
- A100-40GB에서 EDM-B 단독 full-width forward/backward/AdamW를 B=1,2,4,8,16의
  2의 제곱 순서로 검사했다. Peak allocated는 각각 2.99, 5.73, 11.22, 22.22 GiB였고
  B=16은 OOM으로 거부됐다. 실제 Stage 3에는 frozen Stage 2 frontend와 production
  batch가 더해지므로 이 단독 결과만으로 B=8을 채택하지 않고, 현재 B=1/accumulation=4를
  보수적 시작값으로 유지해 complete Stage 2 artifact에서 end-to-end 재검증한다.
- Stage 2 fold Job은 immutable `/workspace/code/stage2-folds-porsche-v2`에서만
  import하며 porsche-local PVC에 fold별 checkpoint를 직접 기록한다.
  Stage 3 Job도 향후 immutable `/workspace/code/stage3-production-v1` snapshot만 본다.
- Stage 3 본학습은 complete Stage 2 OOF/checkpoint가 선행되어야 하므로 아직 시작하지
  않았다. 먼저 현재의 1-GPU-per-fold Indexed Job으로 세 cross-fit checkpoint를 모두
  확보한 뒤 OOF와 나머지 Stage 2 arm을 이어서 실행한다.
