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

- 전체 repository suite: `464 passed, 7 skipped`(CUDA/pin allocator host-only skip)
- OOF/remote/Stage 3 lazy reader 집중 회귀, compile, `git diff --check`, 모든
  Stage 2/3 YAML parse와 embedded bash `bash -n`, Kubernetes client/server dry-run 통과
- porsche A100 1개는 `porsche-gpu-0`이 계속 확보하고 있고 두 번째 shell은 Pending이다.
  현재 node의 8 GPU가 모두 할당되어 정식 exact 2-GPU Job은 아직 스케줄할 수 없다.
  실측 기반 Stage 2 순수 학습 예상은 2 GPU 약 52시간, 1 GPU 환산 약 105시간이며
  OOF inference/compression/overflow 시간이 추가된다.
