# K-CorrDiff Phase 1 Evaluation Protocol v1.1.3b

> 상태: v1.1.3b pre-freeze 평가·model-selection·calibration 규약 — release candidate, architecture §29 freeze ledger 참조  
> 대상 아키텍처: [K-CorrDiff Phase 1 Architecture v1.1.3b](./k_corrdiff_architecture.md)  
> 주 평가자료: continuous KMA HSR의 완전한 timeline과 validity mask

## 1. 목적

이 문서는 K-CorrDiff의 모델 선택, ablation, 확률 calibration과 최종 보고 규약을 고정한다. CPrecNet event archive에서 얻은 성능과 continuous operational timeline에서 얻은 성능을 섞지 않는다.

평가 단위는 다음 12개 lead의 30분 누적강수다.

```text
tau in {0.5, 1.0, 1.5, ..., 6.0 hours}
verification interval = (t0 + tau - 30 min, t0 + tau]
unit = mm/30min
grid = 500 m, 256 x 256
```

## 2. 검증 관측·estimand 계약

평가 target은 학습 target과 동일한 공통 helper로 만든다.

```text
A_raw_tau = 7-scan trapezoidal accumulation in linear R space
w_tau = 1[A_raw_tau >= 0.1 mm/30min]
A_obs_tau = A_raw_tau * w_tau
z_tau = log1p(A_obs_tau / A0)
```

검증에서만 별도의 thresholding, scan 포함규칙 또는 결측 보간을 사용하지 않는다.

```text
one of seven timestamp scans is wholly missing -> drop sample
scan exists but a pixel fails QC -> M_target_tau=1 only if all seven pixels are valid
missing scan -> zero replacement forbidden
observation below A_wet -> zero, exactly as in training
```

`A_raw_tau`는 sensitivity 분석용으로 별도 보존하지만 v1.1.3b 공식 metric은 censored `A_obs_tau`를 사용한다. `A0=1 mm`와 `A_wet=0.1 mm/30min`은 checkpoint·target·평가가 공유하는 고정 계약이다. Timestamp 전체 부재와 pixel QC invalid를 동일한 sample-drop 또는 dry 값으로 처리하지 않는다. 미래 관측에서 계산한 `M_target_tau`는 target-dependent training/calibration objective와 metric denominator에만 사용하며 어떤 forecast condition이나 inference API에도 전달하지 않는다. Pixel QC invalid의 판정 계약 — continuous KMA는 not-sentinel AND static coverage, CPrecNet은 static coverage로 강등 — 은 architecture §2.2를 단일 기준으로 따르며, sentinel 집합 확정 전의 manifest는 공식본이 아니다.

### 2.1 Radar-derived verification truth의 의미

공식 관측 진리는 gauge-equivalent 지상강수량이 아니라 KMA HSR 합성반사도에서 `Z=200 R^1.6` 변환, 10 dBZ 전처리와 30분 누적·`A_wet` censoring을 거쳐 만든 **radar-derived precipitation product**다. 따라서 bright band, 고체·혼합강수, attenuation, beam blockage, 지형 차폐와 고정 Z-R 관계의 체계오차가 포함될 수 있다. 본 protocol에서 “관측”, “truth” 또는 `A_obs`는 이 파생 product를 뜻하며 AWS 강수량과 동일하다고 주장하지 않는다.

Model selection, calibration과 final score는 이 고정 HSR product를 기준으로 수행한다. AWS와의 비교는 Appendix A의 사전등록 annex로만 수행하며 primary model selection, guardrail, calibration parameter 또는 threshold 결정에 사용하지 않는다.

### 2.2 Holdout validity estimand

`rho_invalid_max`는 outer-train의 regression, OOF와 diffusion 학습 item 선정 전용이다. Model-selection validation, calibration과 final test에서는 이 threshold를 이유로 sample 전체를 제거하지 않고 `M_target_tau=0` pixel만 objective·metric denominator에서 제외한다. 각 split·lead에서 `rho_invalid_max` 초과 sample 비율, valid-area fraction과 spatial missingness를 보고한다.

공식 primary estimand는 사전 선언한 chronological split-boundary embargo와 관측 validity를 통과한 eligible timeline에서 §3.2의 event, strict-dry와 marginal/background stratum을 모두 포함한 operational-time-weighted valid pixel-time 평균이다. 특정 stratum만 사용한 score는 반드시 조건부 estimand로 이름에 표시한다.

## 3. Split, model selection과 통계 단위

### 3.1 Split governance

먼저 연도 또는 사전 선언한 장기 연속기간으로 chronological split interval을 고정하고 각 경계에 최소 7시간 dependency embargo를 둔다. Eligible item은 `[t0-55 min,t0+6 h]` 전체가 하나의 split interior 안에 놓여야 한다. 그 뒤 각 split 내부에서 storm/event group, strict-dry block과 marginal/background block을 만들고 model sample을 materialize한다. 정보 흐름은 다음 한 방향으로 고정한다.

```text
outer train:
    manifest-block-grouped K-fold OOF regression and diffusion training
    train-only normalization, climatology and residual scale
    provisional calibration parameters for model comparison, fit from OOF artifacts only

model-selection validation:
    architecture and every discrete/continuous choice
    EDM-A/B, 3/5-fold, sparse L2, temporal stem, e_time ablation
    loss/training-sigma schedule and solver
    d-enabled arm, calibration family and pooling strategy
    all candidates use the fixed 16-member x 8-step selection signature

calibration split:
    frozen model and exact sampler signature only
    fit b, c, optional preselected d, gamma
    and predeclared probability-mapping parameters
    for the fixed 32-member x 12-step final-primary signature

final test:
    one-time report of the already frozen primary configuration
    no selection, refit, threshold change or recalibration
```

Model-selection candidate에 calibration이 필요하면 그 임시 parameter는 outer-train OOF artifact만으로 fit하고 model-selection validation label에는 fit하지 않는다. Validation은 candidate를 점수화하고 calibration family·pooling rule을 선택하는 데만 사용한다. Primary model과 sampler가 동결된 뒤 독립 calibration split에서 같은 사전 선택 family를 새로 fit하며, 이 최종 parameter만 final test에 적용한다.

EDM-A/B와 다른 사전 등록 ablation을 final test 보조표에 함께 보고할 수는 있지만 test 결과로 primary model을 바꾸지 않는다. Calibration 결과로 구조, K, solver, e_time, step 또는 member 수를 바꾸면 기존 calibration은 폐기하고 untouched calibration 자료로 처음부터 다시 해야 한다.

CPrecNet 결과는 `event-conditioned pretraining/evaluation`이라고 별도 표기한다. Dry base rate, initiation, decay와 false alarm을 포함하는 공식 0–6시간 성능은 continuous KMA timeline에서만 계산한다.

Chronological split 경계, embargo 길이, eligible item 수와 `boundary_embargo_fraction`을 manifest version에 저장한다. Event group이 예정 경계를 가로지르면 group 전체를 `test > calibration > model-selection validation > train` 우선순위의 높은 holdout에 배정하고 dependency embargo를 다시 계산한다. Daily background block을 무작위로 서로 다른 split에 배정하지 않는다.

### 3.2 Event, strict-dry와 marginal/background manifest

공식 continuous manifest는 307 km uniform context domain과 500 m target verification product를 함께 사용한다. Label-derived grouping은 split leakage를 막기 위한 offline data-contract 작업일 뿐 model condition으로 전달하지 않는다. Chronological split interior의 candidate verification item을 스캔해 group과 stratum을 먼저 고정한 뒤 model sample을 materialize한다.

#### 3.2.1 Context-active seed

Raw HSR의 10 dBZ cutoff에 해당하는 약 `0.153765 mm/h` 이상인 local-max context pixel이 64개 이상이면 해당 5분 시각을 `context_active`로 둔다.

```text
context_active(t):
    count(context_localmax_R(t) >= 0.153765 mm/h) >= 64

base context runs:
    consecutive context_active 5-minute times
```

`context_localmax_R`와 모든 seed 계산은 architecture §2.2의 sentinel/invalid 제외를 먼저 적용한 값으로 수행한다.

#### 3.2.2 Significant target wet component

각 valid 30분 verification item `(t0,tau)`에서 `M_target_tau=1`이고 `A_obs_tau>=A_wet`인 target pixel의 8-connected component를 계산한다. 다음 중 하나를 만족하는 component를 significant로 정의한다.

```text
component pixel count >= 4 target pixels     # 500 m grid에서 1 km^2
OR
component maximum A_obs >= 1 mm/30min
```

두 상수는 `rho_invalid_max`와 같은 governance를 따른다. Outer-train manifest QA 전에 고정하고 config hash에 포함하며 calibration/test를 본 뒤 바꾸지 않는다.

Significant component가 있는 verification interval `[t_start_tau,t_end_tau]`은 context activity와 무관하게 event seed interval을 만든다. Context-active run과 significant-wet seed interval을 합친 뒤 다음 규칙으로 group을 만든다.

```text
temporal merge:
    seed interval 사이 inactive gap <= 12 h이면 같은 event group

dependency guard:
    merged event interval을 앞뒤 7 h 확장
    확장 interval이 겹치면 재귀적으로 group merge

spatial rule:
    같은 307 km influence domain에서 시간적으로 겹치는 여러 rain object는
    보수적으로 같은 group; object별 spatial split 금지

sample assignment:
    [t0-55 min, t0+6 h] 전체 dependency interval이
    한 group 또는 한 background block과 한 split에만 속해야 함
```

공식 typhoon ID처럼 개별 event ID가 있는 metadata가 여러 자동 group과 겹치면 이를 하나로 merge한다. 장마·전선·대류형 같은 광범위 regime metadata는 평가 strata로만 사용하고 한 계절 전체를 하나의 group으로 합치지 않는다.

12 h merge gap과 ±7 h guard의 연쇄로 실효 병합 간격은 최대 14 h이며 장마철에는 수 주 규모 group이 생길 수 있다. 이는 의존성 관점에서 정직한 단위이므로 group을 자르지 않는다. 대신 manifest QA에서 group duration 분포와 JJA event-block 수를 보고한다. Pooling은 calibration cell support를 완화할 뿐 bootstrap의 독립 event 수를 늘리지 못하므로, 독립 event block 수, effective block count, 최대 block의 weight 비율과 JJA event 시간 중 최대 block 점유율을 함께 보고한다. 판정 집계의 독립 event block 수가 20개(§29 BLOCKING-3, §13.3 확률 cell 규칙과 동일 상수) 미만이면 해당 superiority CI에 low-support를 표기하고 확정적 우월 주장을 하지 않는다.

#### 3.2.3 Strict-dry block

Event guard 밖에서 radar validity가 연속으로 확인되고, dependency interval의 모든 5분 context 시각에서 threshold pixel count가 정확히 0이며, candidate item의 모든 valid target pixel이 `A_obs_tau=0`인 구간만 strict-dry 후보가 된다. Eligible item을 `t0`의 UTC calendar day 기준 비중첩 24시간 block에 배정한다. Daily block 경계에서 item을 제거하지 않으며 split-boundary dependency는 §3.1의 embargo가 처리한다.

```text
strict dry means:
    no target wet pixel at A_wet
    AND context threshold-pixel count == 0 at every dependency time
```

#### 3.2.4 Marginal/background block과 speckle rule

Event group에도 strict-dry에도 해당하지 않는 모든 valid timeline은 marginal/background stratum으로 유지한다. 대표적으로 다음이 포함된다.

```text
context threshold pixel count in 1..63
non-significant target wet component
isolated A_wet-level speckle failing both significant criteria
other valid weak-precipitation/background transition periods outside event guard
```

Significant 기준 미만의 고립 wet pixel은 event를 새로 만들지 않지만 strict dry로도 분류하지 않는다. 이를 제거하거나 dry로 재명명하지 않고 `t0`의 UTC calendar day 기준 비중첩 24시간 marginal/background block에 넣는다. Speckle item 수, wet pixel 수, component 면적, maximum, season과 split 분포를 보고한다.

계산량 때문에 marginal/background block을 subsample할 경우 inclusion probability를 manifest에 저장하고 operational-time aggregate에서 `P_target/P_draw` weight를 적용한다.

#### 3.2.5 Manifest assertions와 estimand 경계

어떤 모델도 학습하기 전에 다음 세 audit를 모든 split에 적용한다.

```text
strict-dry contamination:
    strict-dry item의 모든 valid target pixel에서 A_obs == 0

significant-wet coverage:
    significant target component가 있는 모든 valid item은 event group에 포함
    # 구성적 귀결이며 위반은 구현 오류

full-timeline partition:
    split-boundary embargo를 통과한 모든 eligible valid item이 정확히 하나의
    event / strict-dry / marginal-background block에 포함
    unassigned_eligible_item_fraction == 0
```

Split·계절별 event, strict-dry, marginal/background 비율, `boundary_embargo_fraction`과 unassigned fraction을 보고한다. `unassigned_eligible_item_fraction>0`이면 sample을 사후 제거하거나 `not context_active`를 자동으로 dry로 재분류하지 않는다. Grouping 구현을 수정하고 전체 manifest와 split을 다시 만든다.

공식 CRPS, Brier, FAR와 bias는 사전 선언 split-boundary embargo와 관측 validity를 통과한 eligible timeline에서 세 stratum 전체에 대한 operational-time estimand다. Event+strict-dry만 사용하거나 marginal/background를 제외한 score는 `conditional_on_selected_strata`로 표시하며 공식 전체-timeline score를 대체할 수 없다.

한 event group이 예정 split 경계를 넘으면 §3.1의 holdout 우선순위와 embargo 규칙을 적용한다. Strict-dry와 marginal daily block은 이미 고정된 split interior 안에서만 생성하며 경계 양쪽에 같은 dependency item을 나누지 않는다.

### 3.3 Paired block bootstrap

신뢰구간과 모델 비교는 다음 규약을 사용한다.

```text
resampling unit:
    complete storm/event group,
    complete 24 h strict-dry block,
    or complete 24 h marginal/background block

resampling:
    event, strict-dry and marginal/background strata 안에서 각각 replacement sampling

stratum counts:
    원 test set의 각 stratum block 수 유지

pairing:
    모든 비교 모델에 동일한 resampled block ID와 multiplicity 사용

replicates: 2,000
interval: percentile 95%
seed: bootstrap_seed=11101
```

Pixel, 개별 5분 window 또는 겹치는 84-frame window를 bootstrap 단위로 사용하지 않는다. Strict-dry와 marginal/background block도 resampling population에서 제외하지 않는다.

### 3.4 Primary model-selection 및 seed 규칙

Provider track마다 별도로 모델을 선택한다. 모든 architecture candidate는 model-selection validation에서 고정된 `selection_signature=16 members x 8 EDM steps`로 비교한다. Member/step 수 자체는 v1.1.3b의 model-selection 변수가 아니다.

```text
Primary endpoint to minimize:
    Score_CRPS = (1/12) * sum_tau mean_valid_pixel_time(
        physical-space all-valid fair_CRPS_tau
    )

Reference configuration per provider track:
    e_time enabled
    causal advection, no sparse L2
    EDM-B deployment pyramid
    3-fold OOF, d disabled
    otherwise frozen v1.1.3b defaults

Mandatory non-inferiority guardrails versus the reference:
    upper 95% CI of relative event-conditioned fair-CRPS degradation <= 2%
    lower 95% CI of event-conditioned q_cal(A_wet) BSS difference >= -0.01
    lower 95% CI of 5 mm/30min, 8 km FSS difference
        computed from the primary deterministic A_ensmedian threshold field >= -0.01
```

Event-conditioned occurrence BSS는 event block 안의 모든 valid wet·dry pixel-time을 포함하며 관측 wet pixel만 사후 선택한 score가 아니다. Guardrail을 모두 통과한 candidate 중 `Score_CRPS`가 가장 작은 것을 선택한다. Observation-intensity strata와 heavy-rain-only score는 필수 진단과 guardrail 보조자료지만 단독 primary endpoint로 사용하지 않는다.

#### 3.4.0 Stage-0 regression funnel

Diffusion 학습 전에 regression 단계만으로 candidate를 선별하는 2단 funnel을 사전등록한다. 3-fold OOF 기준 configuration 하나의 full pipeline은 seed당 5개 학습(fold 3 + deployment 1 + diffusion 1)이므로, regression을 바꾸는 candidate는 Stage-0을 먼저 통과해야 한다.

```text
Stage-0 endpoints (model-selection validation, regression only):
    E1 = lead-averaged operational-time-weighted all-valid
         weighted MSE(mu_z_tau, z_tau)               # lower is better
    E2 = event-conditioned raw p_tau BSS
         (§5.8 climatology reference)                # higher is better
    두 endpoint는 합산하지 않고 각각 판정한다

Stage-0 cull rule (paired block bootstrap, §29 BLOCKING-3 상수):
    reference 대비 E1 상대 열화 upper 95% CI > 5%
    또는 E2 하락 lower 95% CI < -0.02
    -> diffusion 단계로 승격하지 않음

적용:  regression 입력·손실·인코더를 바꾸는 candidate
       (ERA 변수, advection, e_time, sparse L2, p-detach L_mean 등)
비적용: diffusion-side sweep(solver, training sigma, EDM-A/B, adapter)은
       동일 regression과 OOF artifact를 공유하므로 Stage-0을 거치지
       않지만, candidate configuration당 diffusion 학습 1회는 필요하다
```

Stage-0 margin과 endpoint는 결과 확인 후 바꾸지 않으며, cull된 candidate도 Stage-0 수치와 함께 기록한다. Stage-0 진단으로 ERA whole-source·tp occlusion에 더해 wind/thermodynamic/moisture/mass 변수 family occlusion 민감도를 lead별로 기록한다. 이는 candidate 선별 기준이 아니라 24채널의 실질 기여를 설명하는 진단이다. Provider track은 radar-only → ERA5 oracle → operational 순으로 순차 진행한다.

#### 3.4.1 Training 및 ensemble seed

Broad candidate screening은 모든 candidate에 동일한 training seed와 common random numbers를 사용한다.

```text
screen_training_seed = 11103
common_ensemble_seed = 11104
repeat_training_seeds = [11103, 11105, 11106]
```

Ensemble noise는 counter-based PRNG로 `(common_ensemble_seed, sample_id, tau, member_index, stochastic_step_index)`에 keying하고 model/checkpoint ID는 key에 넣지 않는다. 따라서 model 간 동일 initial noise와, stochastic solver인 경우 동일 step-noise stream을 사용한다. 32-member 평가의 앞 16개 member는 16-member selection과 동일한 member key를 사용한다.

Screening 후 reference와 guardrail을 통과한 `Score_CRPS` 상위 두 candidate를 총 3개 training seed로 재학습한다. Reference가 상위 두 candidate에 포함되더라도 reference는 항상 3 seed를 갖는다. EDM-A는 screening `Score_CRPS` 순위와 무관하게 reference와 함께 항상 3-seed finalist 지위를 가진다. EDM-A/B pair에는 다음 dispersion non-inferiority를 사전등록한다.

```text
E_SSR = abs(log(max(S_adj, eps) / max(RMSE_ensmean, eps)))
        S_adj는 gamma 적용 후 (1+1/N) 보정 spread
        S_adj와 RMSE_ensmean은 event-conditioned pooled second
        moment를 먼저 집계한 뒤 비율을 계산한다
        (pixel별 비율을 만들어 평균하는 방식 금지)
RI    = sum_j abs(f_j - 1/(N+1))
        randomized-tie rank histogram의 reliability index

주판정: event-conditioned, 12-lead 집계, 3-seed statistic 평균
보조:   lead별과 strict-dry/marginal stratum별 기술 보고
판정:   (E_SSR_B - E_SSR_A) 와 (RI_B - RI_A) 각각의
        paired block bootstrap upper 95% CI <= +0.05
        두 metric 모두 통과해야 EDM-B가 non-inferior
```

절대 margin `+0.05`는 Stage-0 margin과 같은 사전 선언 기본값이며 model-selection 시작 전에 비준한다. EDM-B가 CRPS에서 우세하더라도 이 dispersion non-inferiority를 통과하지 못하면 §11.5의 encoder in-sample 의심 절차를 우선 적용한다. EDM-A/B의 종결 경로는 다음으로 완결하며 합성점수는 사용하지 않는다.

```text
EDM-B primary eligibility:
    E_SSR와 RI non-inferiority를 모두 통과해야 함

EDM-B가 하나라도 실패하면:
    EDM-B는 primary 부적격
    EDM-A가 §3.4 guardrail과 §3.5 gate를 모두 통과하면 EDM-A 선택
    아니면 해당 provider track에 primary CorrDiff 없음
        -> §3.5 diffusion 포함 gate 실패와 동일하게
           regression + probability calibration만 배포
    재도전은 새 protocol version

CRPS practical tie이고 둘 다 dispersion-eligible이면:
    한 arm이 E_SSR와 RI 모두 우위 -> 그 arm 선택
    metric이 하나씩 갈리면        -> dispersion tie
    dispersion tie               -> p95 latency
``` Finalist 선택은 seed별 score의 단일 최선 run이 아니라 3-seed 평균으로 수행한다.

Finalist 재학습 후 mandatory guardrail도 동일한 세 training seed로 다시 평가한다. Screening seed에서의 통과만으로는 충분하지 않다. 각 seed에서 event-conditioned fair CRPS, event-conditioned `q_cal(A_wet)` BSS와 primary `A_ensmedian` 기반 5 mm/8 km FSS를 완전한 statistic으로 먼저 계산하고, 세 seed statistic의 평균으로 candidate-reference difference를 만든다. 이 3-seed 평균 guardrail을 모두 통과한 finalist만 최종 선택 대상이 된다.

```text
s_ref  = sample_std of reference Score_CRPS across 3 seeds
s_cand = sample_std of candidate Score_CRPS across 3 seeds
sigma_seed_pair = sqrt((s_ref^2 + s_cand^2) / 2)

delta_seed_s = Score_CRPS_candidate(seed_s) - Score_CRPS_reference(seed_s)
paired_seed_delta_std = sample_std_s(delta_seed_s)
```

Finalist는 동일한 세 training-seed ID를 사용한다. 각 manifest block에서 먼저 seed별 score contribution을 계산하고, 같은 configuration의 세 seed contribution을 평균한 뒤 그 seed-averaged block contribution으로 paired block bootstrap을 수행한다. 별도로 seed별 전체 `Score_CRPS`에서 `s_ref`, `s_cand`를 계산한다. `paired_seed_delta_std`는 common-seed pairing에 따른 차이의 변동성을 보여주는 진단값으로 함께 기록한다. 공분산에 따라 `sigma_seed_pair`보다 항상 크거나 작다고 가정하지 않으며, v1.1.3b의 practical-tie 선택규칙은 사전 고정된 `sigma_seed_pair`를 계속 사용한다.

Candidate와 reference의 paired-block bootstrap fair-CRPS difference 95% interval이 0을 포함하거나, interval이 0을 제외하더라도 3-seed 평균 차이의 절대값이 `sigma_seed_pair`보다 작으면 practical tie로 처리한다. Mandatory guardrail의 bootstrap도 같은 resampled block과 3-seed statistic 평균을 사용한다. `sigma_seed_pair`는 정식 신뢰구간이 아니라 사전 고정된 seed-stability threshold이며 `paired_seed_delta_std`는 진단 전용이다. 동률이면 p95 end-to-end latency가 낮은 쪽을 선택한다. Seed 규칙이나 margin을 결과 확인 후 바꾸면 protocol version을 올려야 한다.

Configuration 선택과 배포 checkpoint는 분리된 규칙이다. 3-seed 평균은 configuration을 고르는 인증치이며, calibration·final test·배포에 쓰는 checkpoint는 다음으로 고정한다.

```text
deployment_checkpoint_rule:
    configuration selection = 3-seed mean over [11103, 11105, 11106]
    deployment_training_seed = 11103    # §29 BLOCKING-3, 사전 고정
    validation 결과로 seed를 고르는 best-seed selection = 금지

final calibration and final test:
    deployment_training_seed의 fold-regression, deployment-regression,
    diffusion checkpoint hash를 명시적으로 binding
    calibration table, condition-swap, latency, final test 결과는
    모두 이 checkpoint 계보에서 산출
```

이 규칙은 reference, EDM-A와 모든 arm에 일률 적용한다. 인증치(3-seed 평균)와 배포치(단일 seed)가 다르므로 배포 checkpoint의 seed 단독 `Score_CRPS`를 3-seed 평균 옆에 병기 보고한다. 새 seed로 최종 재학습하는 경로를 쓰려면 `final_training_seed`를 지금 선언하고 fold·deployment regression과 diffusion 전체를 그 seed로 재학습·재calibration해야 하며, v1.1.3b 기본은 재학습 없는 seed 11103 재사용이다.

선택된 model은 처음부터 고정된 `final_primary_signature=32 members x 12 steps`로 독립 calibration을 fit한 뒤 final test에 한 번 사용한다. `operational_signature=8 members x 4 distilled steps`는 별도 calibration과 표를 갖는 운영 전이 track이다. Member/step 수를 실제로 튜닝하려면 finalist를 validation에서 여러 signature로 다시 비교하는 새 protocol을 먼저 선언해야 한다.

### 3.5 절대 baseline gate와 diffusion 포함 gate

§3.4의 guardrail은 reference 대비 non-inferiority이므로 reference 자체가 약하면 baseline보다 나쁜 모델이 통과할 수 있다. 선택된 primary candidate는 model-selection validation에서 다음 절대 gate를 추가로 통과해야 한다. Deterministic baseline의 CRPS는 퇴화 예측분포의 닫힌형 `CRPS_det(x, y) = |x - y|`로 정의하며 절대오차와 일치한다. `N(N-1)` 분모를 쓰는 fair U-statistic 구현은 `N>=2` 전용이므로 deterministic baseline에 사용하지 않는다. STEPS ensemble이 가용하면 ensemble 대 ensemble로 비교한다.

```text
lead 0.5–1.0 h  (baseline 집합 {Lagrangian persistence,
                deterministic pySTEPS} 각각에 대해 모두 통과):
    event-conditioned fair CRPS:
        (CRPS_model - CRPS_base) / CRPS_base 의 upper 95% CI <= 0.02
    1 mm/8 km FSS:
        (FSS_model - FSS_base) 의 lower 95% CI >= -0.01   # 절대 margin

lead 1.5–2.0 h  (같은 baseline 집합 각각에 대해 모두):
    (CRPS_base - CRPS_model) 의 lower 95% CI > 0
    (FSS_model - FSS_base)   의 lower 95% CI > 0

lead 2.5–4.0 h  (deterministic baseline = radar-only A_q50_direct,
                occurrence baseline = radar-only regression의 p_cal):
    (CRPS_base - CRPS_model) 의 lower 95% CI > 0
    candidate q_cal(A_wet) BSS - baseline p_cal(A_wet) BSS 의
        lower 95% CI > 0

lead 4.5–6.0 h  (baseline 집합 {track별 coarse forecast tp
                (§5.7의 30분 변환 규약), radar-only A_q50_direct}
                각각에 대해 모두):
    (CRPS_base - CRPS_model) 의 lower 95% CI > 0

all leads:
    lead-averaged candidate q_cal(A_wet) BSS 의 lower 95% CI > 0
    (outer-train lead x season climatology reference)
    per-lead BSS는 기술 보고 (12개 개별 CI gate는 다중비교로 과엄)
```

Band statistic은 대역 내 각 lead-level statistic을 완결 계산한 뒤 등가중 평균한 값이며 — primary score의 lead 등가중과 일관 — bootstrap은 이 band 평균에 적용한다. Pixel-time을 대역 전체에서 pooling하는 구현이나 sample-수 비례 lead 가중은 사용하지 않는다.

절대 gate의 deterministic CRPS baseline은 `A_q50_direct`다. Deterministic CRPS는 절대오차와 같고 그 최적해는 조건부 median이므로, 물리단위 조건부 median을 직접 학습한 field가 공정한 상대다. `A_reg_zinv = A0*expm1(mu_z)`는 z-공간 평균의 역변환이라 물리 mean(Jensen 하향)도 median도 아니므로 시스템 절대 gate의 baseline으로 쓰지 않는다. `A_reg_zinv`는 diffusion 포함 gate의 parent baseline으로만 사용해 diffusion의 증분 기여를 분리한다. 더 보수적인 변형으로 절대 gate를 `A_q50_direct`와 `A_reg_zinv` 양쪽 AND로 판정할 수 있으나 이는 선택 사항이다.

Baseline 합성은 min/max 선택이 아니라 나열된 각 baseline에 대한 AND-합성이다. 이는 score 방향(CRPS는 낮을수록, FSS/BSS는 높을수록 좋음)에 따른 `max()` 정의 모호성을 제거하고 validation 결과로 baseline을 고르는 여지도 없앤다. FSS margin이 상대가 아니라 절대인 이유는 baseline FSS가 0에 가까울 때 상대 열화율이 불안정하기 때문이다. `p_cal`은 §13.2의 동일 mapping family를 regression `p_tau`에 적용한 calibrated 확률이며, regression-only baseline에는 `q_cal`이 존재하지 않으므로 이름을 분리한다.

Baseline 짝은 provider track별로 고정한다: radar-only track은 radar baseline만, oracle track은 radar baseline과 ERA `tp` oracle baseline, operational track은 radar baseline과 operational coarse `tp` baseline을 사용한다. 판정은 모두 §3.3 paired block bootstrap을 사용한다. 0.5–1.0 h 대역에 우월이 아닌 margin non-inferiority를 두는 이유는 순수 이류 체제에서 Lagrangian persistence가 근사최적일 수 있기 때문이다.

Gate 실패는 조용한 완화 대상이 아니다. 실패 대역별 사전등록 escalation은 다음과 같다: 단기 실패는 §11.1의 advection/sparse-L2 escalation, 장기 실패는 architecture §17의 장리드 mean-anchor 교체 후보, occurrence 실패는 §13.4 ladder. Gate 정의나 margin 변경은 protocol version 변경이다.

Diffusion 포함 gate: diffusion은 `A_ensmean` MAE가 regression보다 나빠질 수 있으나, regression-only 대비 event-conditioned fair CRPS를 개선하고(paired block bootstrap 95% CI가 0 제외) 아래 보조 statistic 중 최소 1개에서 CI 기준 개선을 보여야 한다.

```text
FSS:      5 mm/8 km FSS (primary field 규약)
          (FSS_diff - FSS_reg) 의 lower 95% CI > 0

spectrum: E_PSD = mean_{lambda in [2,16] km} abs(log(ratio(lambda)))
          ratio = max(PSD_pred, PSD_floor) / max(PSD_obs, PSD_floor)
          PSD_floor는 train split에서 고정한 하한(또는 수치 epsilon)
          개별 member spectrum의 평균으로 계산
          집계는 bootstrap block 안에서 PSD를 먼저 pooled 평균한 뒤
          log-ratio를 취한다 (sample별 ratio 선계산 후 평균은
          다른 statistic이므로 금지)
          (E_PSD_diff - E_PSD_reg) 의 upper 95% CI < 0

tail:     E_freq5 = abs(f_pred(A >= 5 mm) - f_obs(A >= 5 mm))
          ensemble의 f_pred는 member별 exceedance 빈도의 평균
          (E_freq5_diff - E_freq5_reg) 의 upper 95% CI < 0
```

Ensemble-mean field spectrum은 member 평균화로 다시 평활되므로 고주파 복원 판정에 쓰지 않는다. Regression-only 쪽의 spectrum·tail은 `A_reg_zinv` 단일장으로 계산해 짝을 맞춘다. 보조 metric도 point improvement가 아니라 CI 판정을 요구하는 이유는 sampling noise만으로 gate가 열리는 것을 막기 위해서다. 미충족 시 operational 배포는 regression과 probability calibration만 사용하고 diffusion은 연구 track으로 유지한다. 이 gate는 §11.3 진단의 판정 규칙이다.

Model-selection validation의 모든 percentile interval은 selection interval로 명명한다. 같은 validation이 candidate 선택과 gate 판정을 함께 수행하므로 이 interval은 선택 과정을 조건부로 한 명목 95% coverage를 보장하지 않는다. Confirmatory interval은 untouched final test의 것뿐이다. 필요하면 validation 내부를 broad screening과 gate-confirmation 기간으로 시간 분할하는 변형을 사전 선언할 수 있다.

## 4. 입력 provider와 시간 접근별 실험 트랙

서로 다른 미래 condition을 같은 표의 단일 숫자로 합치지 않는다.

| Track | 24채널 ERA/NWP 입력과 access rule | 목적 |
|---|---|---|
| Radar-only degraded | `era_present=0`; `e_time`은 유지 | NWP feed 지연·장애 시 성능 |
| Full-trajectory ERA5 oracle | 최대 6시간 bracket 안의 미래 ERA5 reanalysis trajectory 허용 | 연구용 retrospective 상한 |
| Target-end-causal ERA5 oracle | 순간장은 `valid_time<=t_end_tau`, `tp`는 `interval_end<=t_end_tau` | target 이후 reanalysis 의존성 진단 |
| Operational provider | §4.2의 declared cycle/latency를 만족하는 hindcast | 실제 배포 조건의 성능 |

모든 ERA5 future-valid-time 결과에는 `oracle`을 표와 그림 제목에 명시한다. Target-end-causal도 `t0`에 이용 가능한 forecast가 아니라 미래 reanalysis이므로 oracle이다. `ceil_hour(t_end_tau)>t_end_tau`인 token까지 사용한 결과는 target-end-causal이라 부르지 않고 별도 `lead-local bracket oracle`로 표시한다.

```text
full-trajectory ERA5 oracle
    expected >= target-end-causal ERA5 oracle
    expected >= operational-provider hindcast
```

위 순서는 정보량에 따른 기대 가설이며 metric별 성공조건이나 강제 단조관계가 아니다. `full - target-end-causal`은 target 이후 reanalysis 상태 전체에 대한 의존을 뜻하며 특정 동화정보 하나로 해석하지 않는다. `target-end-causal - operational`도 analysis/forecast 품질과 provider·해상도·regrid·전처리 차이를 함께 포함한다. 공식 arm은 해당 access/provider 계약으로 학습하고 같은 계약으로 평가한다.

### 4.1 공통 ERA/NWP 시간 audit

모든 track은 UTC native-hour 8-frame reader와 동일한 24채널 schema를 사용한다.

```text
data_valid_inst       # token의 순간형 23변수가 모두 valid
tp_valid              # token별 1 h tp interval validity
trajectory_window_mask
temporal_access_mask  # 의도적인 track별 시간 제한
era_present
tp_present
provider and issue/valid-time provenance
```

`tp` channel dropout/occlusion은 필수 ablation이지만 23채널을 별도 주 모델로 정의하지 않는다. 평가 manifest는 다음 시간 정합 audit를 통과해야 한다.

```text
t0 minute-of-hour 분포를 00,05,...,55별로 보고
radar timezone -> UTC 변환 전후 timestamp 표본 대조
h0=floor_hour(t0_UTC), h0..h0+7h frame index 확인
모든 t_end_tau가 full track의 허용 token 범위 안에서 bracket됨을 확인
target-end-causal track에서 valid_time/interval_end > t_end_tau token attention이 0임을 확인
tp duration=1h, interval_end=valid_time, physical unit=mm assertion
ERA5 oracle required window에서 instantaneous 23변수 중 하나라도 빠지면 sample drop
tp-only 결측은 tp_valid=0이고 instantaneous token은 유지
t_c, mean-solar hour, annual phase가 t0/tau/calendar contract와 일치
```

정시 sample만 남겨 전체 5분 후보의 약 11/12를 버리는 전처리는 공식 track에서 허용하지 않는다.

### 4.2 Operational-provider cycle과 latency 계약

Operational arm은 provider adapter가 다음 계약을 한 번 동결하기 전에는 학습·calibration·평가하지 않는다.

```text
C_provider       = declared cycle-time set
latency_declared = cycle output이 실제 사용 가능해지는 고정 latency

c_star(t0) = max { c in C_provider : c + latency_declared <= t0 }
required_rollout_lead(t0,tau) = t0 + tau - c_star(t0)
forecast_cycle_age_at_t0 = t0 - c_star(t0)
availability_age_at_t0 = t0 - (c_star(t0) + latency_declared)
```

Adapter 계약은 다음을 포함한다.

```text
provider/model version and cycle set
latency_declared and late/missing-cycle handling
required variables의 latest common availability time
forecast step and tp accumulation interval alignment
control member / named member / ensemble mean 중 모델 입력 선택
normalization, conservative regrid and provenance version
maximum supported rollout lead
```

Provider가 ensemble을 제공하면 control, 특정 member 또는 ensemble mean 중 무엇을 24채널 condition으로 사용하는지 사전 선언한다. Cycle/latency/ensemble-input 선택을 바꾸면 별도 provider track과 checkpoint·calibration signature가 필요하다. 최종 보고에는 lead별 `required_rollout_lead`, forecast-cycle age와 availability age 분포를 포함한다.

### 4.3 OOD temporal-access diagnostics

Matched-training 공식 arm과 별도로 다음 inference-only diagnostic을 허용한다.

```text
issue-causal occlusion:
    instantaneous valid_time <= t0
    tp interval_end <= t0
    no separate training
```

Full-trajectory 또는 target-end-causal model에 이 mask를 적용한 결과는 OOD source-occlusion이다. `target-end-causal - issue-causal` 차이는 인과효과가 아니라 **future-valid coarse-environment token에 대한 OOD sensitivity**로만 해석한다. 이 진단 결과로 공식 access arm이나 model을 바꾸지 않는다.

## 5. Baseline

모든 baseline은 K-CorrDiff와 동일한 split, target helper, validity mask와 censoring으로 평가한다.

### 5.1 Eulerian persistence

`R(t0)`를 모든 미래 5분 scan에 그대로 유지하고 공통 trapezoidal helper로 30분 누적을 만든다.

### 5.2 Lagrangian persistence

`t<=t0` radar만으로 optical flow를 추정하고 semi-Lagrangian 외삽으로 미래 5분 rate를 만든 뒤 공통 helper로 누적한다. K-CorrDiff advection input과 같은 flow artifact를 쓰되 neural correction은 사용하지 않는다.

### 5.3 pySTEPS

다음을 구분한다.

```text
deterministic pySTEPS extrapolation
STEPS ensemble, when computationally available
```

### 5.4 Regression-only

Regression의 transformed-space mean을 다음처럼 물리단위로 표시한다.

```text
A_reg_zinv_tau = A0 * expm1(mu_z_tau)
```

`A_reg_zinv_tau`는 `E[z|C,tau]`의 역변환이며 `E[A|C,tau]`, K-CorrDiff ensemble의 물리단위 평균 또는 일반적인 의미의 조건부 중앙값이 아니다. 볼록한 `expm1` 때문에 일반적으로 물리단위 조건부 평균에 대해 하향편향된다. Wet-only `z`가 대칭이라는 추가 가정은 `A0*expm1(m_tau)`에만 wet-conditional 중앙값 근사를 줄 수 있으며 `A_reg_zinv_tau=A0*expm1(p_tau*m_tau)`에는 그대로 적용되지 않는다. 이 출력은 regression 단계의 deterministic baseline으로만 별도 보고한다.

### 5.5 Direct physical regression

CorrDiff ensemble mean/median과 공정하게 비교하기 위해 동일 condition, encoder 용량, split과 validity를 쓰는 별도 regression-only checkpoint를 둔다.

```text
A_mean_direct:
    nonnegative physical-space head trained with MSE
    target interpretation: E[A_model | C, tau]

A_q50_direct:
    nonnegative physical-space head that can output exact zero,
    trained with pinball loss q=0.5
    target interpretation: conditional physical median
```

Huber-trained field를 정확한 physical conditional mean이라고 부르지 않는다. `A_reg_zinv`는 기존 stage-1 transformed-space 진단값으로 계속 보고한다.

### 5.6 CPrecNet

원 CPrecNet의 4 input frame과 18개 5분 output은 최대 90분 예보다. 직접 비교는 미래 output을 동일한 30분 규약으로 집계한 `tau<=1.5 h`에 한정한다. 6시간 recursive rollout은 공정한 기본 baseline으로 사용하지 않으며, 재학습하거나 OOD 실험이라고 명시한다.

### 5.7 Coarse forecast precipitation

운영 NWP 또는 Aurora가 precipitation forecast를 제공하면 coarse forecast 자체를 장리드 baseline으로 둔다. ERA5 `tp` oracle baseline과 운영 forecast baseline을 구분한다.

Coarse `tp`는 1시간(또는 provider native interval) 누적이므로 30분 verification interval로의 시간 변환 규약을 다음으로 고정한다. 비정시 `t0`에서도 이 규약이 gate를 정의한다.

```text
coarse-tp 30-min baseline contract:
    cumulative-from-cycle field는 먼저 interval 누적으로 변환
    I_tau = (t0+tau-30min, t0+tau],  J_j = (v_j - 1h, v_j]
    A_tp30(I_tau) = sum_j tp_j * |I_tau ∩ J_j| / |J_j|
        # provider interval 내 uniform rate 가정, 가정 라벨 명기
    I_tau의 완전 커버를 요구하고 결측 구간이 있으면 해당 baseline NA
    시간 변환을 먼저 수행한 뒤 보수적 공간 regrid
    provider가 native 30분 누적을 제공하면 그 field를 우선 사용하고
    1시간 변환 baseline과 이름을 분리
```

이 변환은 architecture §8.2의 tp interval-end 의미론을 공유 target/tp helper에 구현해 사용하며, baseline과 모델 입력의 시간 처리가 갈라지지 않게 한다.

### 5.8 Outer-train climatology

Skill-score reference는 outer-train만으로 만든 pixel-wise `lead x meteorological season` climatology다. 계절은 DJF/MAM/JJA/SON으로 고정한다.

```text
Brier reference:
    pixel-wise event frequency by lead and season

CRPS reference:
    empirical outer-train A_obs distribution at the same pixel/lead/season

Neighborhood CRPS reference:
    empirical outer-train distribution of the observation after applying
    the same 8 or 32 km areal-mean operator at the same anchor/lead/season
```

Pixel cell의 valid timestamp가 1,000개 미만이면 같은 8 km neighborhood를 pool하고, 그래도 부족하면 domain-wide lead×season climatology로 fallback한다. Neighborhood reference는 pixel-wise marginal climatology를 평균해서 만들지 않고 outer-train 관측을 먼저 동일 areal operator로 집계한 뒤 경험분포를 만든다. Reference 생성도 operational time weighting을 사용하며 calibration/test 관측을 포함하지 않는다. CRPSS는 동일 scale의 reference CRPS로 계산해 secondary interpretation score로 보고한다.

## 6. 예측값의 물리단위 매핑

각 K-CorrDiff member는 다음 순서로 변환한다.

```text
r0_tau_n = s_oof[tau,condition_signature] * r_tilde_hat_tau_n
r1_tau_n = b[tau,condition_signature]
           + d[tau,condition_signature,sampler_core_signature]
           + c[tau,condition_signature] * r0_tau_n

rbar1_tau = mean_n(r1_tau_n)
r2_tau_n = rbar1_tau
           + gamma[tau,condition_signature,ensemble_signature]
             * (r1_tau_n - rbar1_tau)

z_hat_tau_n = mu_z_full_tau + r2_tau_n
A_pre_tau_n = A0 * expm1(max(z_hat_tau_n, 0))
A_hat_tau_n = A_pre_tau_n * 1[A_pre_tau_n >= A_wet]
```

`d_enabled=false`인 비활성 arm에서는 `d=0`이다. `d` 사용 여부는 model-selection validation에서 미리 동결하며 calibration/test에서 새로 활성화하지 않는다.

물리단위 ensemble mean과 median은 member별 역변환·censoring 뒤 계산한다.

```text
A_ensmean_tau = mean_n(A_hat_tau_n)
A_ensmedian_tau = empirical_lower_median_n(A_hat_tau_n)
```

Median은 interpolation 없는 empirical inverse-CDF quantile로 고정한다. NumPy 표현은 `quantile(method="inverted_cdf")`이며, `N=32`에서는 정렬된 16번째 member인 lower median이다. 모든 member가 `0` 또는 `>=A_wet`이므로 median도 같은 support에 놓이며 median 뒤 재-censoring하지 않는다.

다음 값들을 혼동하지 않는다.

```text
mu_z_tau                         transformed-space conditional mean
A_reg_zinv_tau                  inverse mapping of mu_z_tau; regression baseline
A_ensmean_tau                   physical-unit ensemble mean
A_ensmedian_tau                 physical-unit ensemble median
p_tau                           regression wet probability
q_tau = mean_n(1[A_hat>=A_wet]) diffusion ensemble wet fraction
p_cal_tau                       calibrated regression wet probability
q_cal_T                         calibrated ensemble exceedance probability at threshold T;
                                not a member fraction
```

공식 verification event threshold는 학습과 같은 `A_wet`으로 고정한다. 내부 생성 또는 calibration parameter를 실험하더라도 `A_hat>=A_wet`이라는 관측·예측 사건 정의를 바꾸지 않는다.

## 7. Deterministic metric

Metric은 lead별로 계산하고 전체 평균만 단독으로 보고하지 않는다.

### 7.1 Continuous error

Metric별 대표 forecast field를 미리 고정한다.

```text
MAE primary point forecast:
    A_ensmedian_tau

RMSE, bias, domain total and water-balance diagnostics:
    A_ensmean_tau

regression-only diagnostic baseline:
    A_reg_zinv_tau

probabilistic scores:
    all physical-unit ensemble members
```

조건부 median은 MAE의 population optimum이고 조건부 mean은 RMSE의 population optimum이므로 주 점예측을 위처럼 구분한다. 비교의 투명성을 위해 `A_ensmean`의 MAE와 `A_ensmedian`의 RMSE도 보조표에 낼 수 있지만 서로 다른 이름으로 표시한다. 기본 metric은 물리단위 `mm/30min`의 MAE, RMSE와 bias다. Heavy-rain subset과 전체 timeline을 따로 보고한다.

### 7.2 Categorical verification

기본 threshold는 다음과 같다.

```text
0.1, 1, 5 mm/30min
```

Primary deterministic categorical field는 interpolation 없는 `A_ensmedian`, secondary는 `A_ensmean`으로 고정한다. Probabilistic threshold verification은 각 threshold `T`의 `q_T=mean_n(1[A_hat_n>=T])`에 Brier/BSS/reliability를 적용한다.

10 mm/30min은 해당 provider track의 final test에 최소 20개 독립 관측 event block이 존재할 때만 사전 규칙에 따라 extreme 보조 threshold로 보고한다. 이 count rule은 모델과 무관하며 primary 선택에는 사용하지 않는다. 각 threshold와 lead에서 다음을 계산한다.

```text
CSI
POD
FAR
frequency bias
ETS or another explicitly named chance-corrected score
```

### 7.3 Fractions Skill Score

FSS scale은 반경이 아니라 square neighborhood의 한 변 길이로 정의한다.

```text
2 km  = 4 target pixels
8 km  = 16 target pixels
32 km = 64 target pixels
```

Threshold binary field의 neighborhood fraction을 `F`, observation fraction을 `O`라 할 때 다음 정의를 사용한다.

```text
FSS = 1 - sum((F-O)^2) / (sum(F^2) + sum(O^2))
```

분모가 0인 완전 무사건 sample을 임의로 FSS 1로 두지 않는다. Lead·threshold·storm 집계에서 numerator와 denominator를 먼저 합산한 뒤 ratio를 계산하고, 집계 분모도 0이면 `NA`로 보고한다.

다음을 같은 이름으로 섞지 않는다.

```text
primary deterministic FSS: A_ensmedian threshold field
secondary deterministic FSS: A_ensmean threshold field
member-mean FSS: compute FSS per member, then average
probability FSS: ensemble exceedance probability against observed fraction
```

`A_reg_zinv` FSS는 regression-only baseline으로 별도 표시하고 CorrDiff의 대표 deterministic FSS와 혼합하지 않는다.

FSS neighborhood 구현은 다음으로 고정한다.

```text
window: sliding square
even L anchor at output pixel (i,j):
    rows i-(L/2-1) ... i+L/2
    cols j-(L/2-1) ... j+L/2
    # L=4이면 offsets -1,0,1,2

boundary: truncate to in-domain cells; no zero, reflect or periodic padding
fraction denominator: common forecast/observation valid pixels only
minimum valid fraction: 0.8 of the nominal L x L window
below minimum: neighborhood marked invalid
```

Forecast와 observation에 동일한 anchor와 validity denominator를 사용한다. Even-window anchor 또는 padding을 library default에 맡기지 않는다.

## 8. Probabilistic metric

### 8.1 Finite-ensemble fair CRPS

CRPS는 transformed `z`가 아니라 censored physical `mm/30min` member에서 계산한다. `N>=2`에서 finite-ensemble bias를 제거한 U-statistic 형태를 공식 정의로 사용한다.

```text
fair_CRPS(x_1,...,x_N; y)
    = (1/N) * sum_n |x_n-y|
      - (1 / (2*N*(N-1))) * sum_{n != m} |x_n-x_m|
```

두 번째 합은 ordered off-diagonal pair를 뜻한다. Unordered pair 구현에서는 동등하게 `-(1/(N*(N-1))) * sum_{n<m}|x_n-x_m|`를 사용한다. 대각항을 포함한 `-0.5*mean(N^2 pairs)` V-statistic은 공식 score로 사용하지 않는다.

이 정의는 다음 모든 CRPS에 동일하게 적용한다.

```text
§3.4 primary Score_CRPS
all-valid pixel CRPS
event/strict-dry/marginal-conditioned CRPS
observation-intensity strata CRPS
8 km and 32 km neighborhood-aggregated CRPS
optional predeclared threshold-weighted CRPS
```

Domain-wide score는 dry pixel에 지배될 수 있으므로 all-valid와 각 manifest stratum, observation-intensity strata를 함께 보고한다. Primary model selection은 세 stratum 전체의 operational-time-weighted all-valid fair CRPS다.

공간적으로 집계된 확률 skill은 `8 km`와 `32 km` neighborhood의 areal-mean accumulation에 대한 fair CRPS로 추가한다. 각 member와 관측에 §7.3과 동일한 sliding-square anchor, boundary truncation, common-valid denominator와 `0.8` minimum-valid 규칙을 적용해 국지 평균장을 만든 뒤 위 fair CRPS 식을 적용한다. 이는 multivariate spatial CRPS가 아니라 `neighborhood-aggregated fair CRPS`라고 명시한다.

Pixel score에는 outer-train pixel-wise lead×season 경험분포를, 8/32 km score에는 §5.8의 동일 scale areal-mean 경험분포를 reference로 사용해 CRPSS를 보조 보고한다.

```text
CRPSS = 1 - fair_CRPS_model / CRPS_climatology
```

Reference denominator가 0인 cell은 `NA`이며 임의 epsilon으로 skill을 만들지 않는다. Selection 16-member와 final 32-member 점수는 fair estimator 수준에서 비교 가능하지만, 8-step과 12-step solver가 만드는 생성분포 차이까지 제거하는 것은 아니다.

### 8.2 Wet/exceedance probability

다음을 각각 관측 `w_tau`와 비교한다.

```text
raw:        p_tau, q_tau = ensemble fraction above A_wet
calibrated: p_cal_tau, q_cal_Awet
```

각 확률에 대해 Brier score, §5.8의 pixel-wise lead×season climatology를 reference로 한 Brier Skill Score와 reliability diagram을 계산한다. Frozen final system의 primary occurrence score는 `q_cal_Awet`, regression-head 보조 score는 `p_cal_tau`다. Raw `p_tau/q_tau`도 calibration 전 진단으로 반드시 보고한다. Reference에는 test나 calibration 관측을 포함하지 않는다.

Raw empirical member fraction `q_T`의 Brier score는 finite ensemble에서 대략 `p(1-p)/N`의 Monte-Carlo 항을 포함하므로 서로 다른 member 수의 raw `q_T` Brier를 직접 비교하지 않는다. Selection 16과 final 32의 raw Brier 차이를 model improvement로 해석하지 않으며, cross-signature 비교는 calibrated probability와 exact-signature 보고 또는 별도 사전등록 fair-Brier 분석에서만 수행한다.

Reliability bin은 모든 실험에서 고정된 20개 equal-width interval을 사용한다.

```text
bin j=0..18: [j/20, (j+1)/20)
bin j=19:    [19/20, 1]
```

각 bin의 forecast 평균, 관측빈도, valid pixel-time 수와 독립 event/dry/marginal block 수를 저장한다. 독립 block이 20개 미만인 bin은 그림에 sparse로 표시하고 정량적 calibration 결론을 내리지 않는다. Bin 경계는 결과를 본 뒤 바꾸거나 equal-count bin으로 대체하지 않는다.

다음 gap을 lead·season·regime·solar-hour bin과 precipitation-phase proxy별로 기록한다.

```text
signed occurrence gap   = mean(q_tau - p_tau)
absolute occurrence gap = mean(abs(q_tau - p_tau))
reliability-bin gap between p_tau and q_tau
calibrated gap          = mean(q_cal_Awet - p_cal_tau)
```

Cold-surface precipitation proxy는 모든 provider track에 동일한 retrospective evaluation metadata로 적용한다.

```text
cold_surface_proxy =
    full-trajectory ERA5 target-center t2m at t_c <= 1 degree C
```

Target-center `t2m`은 full-trajectory metadata의 native-hour 값을 `t_c`에 선형 시간보간해 만든다. 이는 모델 condition이나 실제 snow-phase truth가 아니며 “강설” 대신 `cold-surface precipitation proxy`로 명명한다. Full-track ERA5 metadata가 없는 sample은 proxy stratum에서 `NA`로 둔다.

Censoring과 finite-step diffusion이 `q_tau<p_tau`를 만들 수 있지만 부호를 불변식으로 가정하지 않는다. `p_tau`와 `q_tau`는 서로에게 맞추는 것이 아니라 관측 `w_tau`에 각각 검증한다. 1 및 5 mm/30min exceedance에도 raw `q_T`와 calibrated `q_cal_T`의 Brier score와 reliability를 추가한다.

### 8.3 Spread–skill과 rank histogram

Lead와 intensity bin별로 ensemble spread와 ensemble-mean RMSE를 비교한다. 유한 ensemble 보정은 고정된 정의를 사용한다.

```text
sample_std = standard_deviation_n(A_hat_n, ddof=1)
adjusted spread = sqrt(1 + 1/N) * sample_std
```

Censoring으로 대규모 zero tie가 생기므로 rank histogram은 tie 안에서 observation rank를 무작위화한다. `rank_tie_seed=11102`인 counter-based PRNG를 사용하고 counter key를 `(sample_id, tau, flat_pixel_index)`로 고정해, 동률 member가 만드는 admissible rank 중 하나를 균등하게 한 번 선택한다. 모델 비교와 bootstrap 반복에서도 같은 observation의 배정 rank를 재사용하며 반복 randomization 평균은 사용하지 않는다.

## 9. 공간구조와 극값

### 9.1 Radial power spectrum

Target과 prediction에 같은 detrending, validity 처리와 2D window를 적용한 뒤 radial PSD를 계산한다. 다음을 분리해서 보고한다.

```text
observation spectrum
mean spectrum across individual members
spectrum of ensemble-mean field
prediction/observation spectrum ratio
```

Radial spectrum은 위치 정확도를 측정하지 않으므로 FSS와 함께 해석한다.

### 9.2 Distribution과 tail

Wet-area rain-rate/accumulation PDF, threshold exceedance frequency와 upper quantile을 lead별로 비교한다. Event-conditioned archive와 continuous timeline의 PDF를 합치지 않는다.

## 10. 고정 case-study 분류

임의로 잘 나온 사례만 선택하지 않고 outer split 이전에 metadata rule로 case category를 정의한다.

```text
target 경계로 유입되는 대류선
target 내부의 initiation
rain-to-dry decay
정체전선 또는 장마형 지속강수
태풍/열대저기압 관련 강수
지형성 강수
겨울 강설·혼합강수 후보
strict-dry control period
marginal/background weak-echo period
```

겨울 강설·혼합강수 후보는 관측 phase truth가 아니라 §8.2의 cold-surface proxy와 계절 metadata를 이용한 case-study category다. 경계 유입 case는 optical flow 방향과 target 경계 교차를 이용해 정의한다. Advection 이후 sparse L2 도입 여부는 `tau<=2h` 경계거리별 FSS/CSI와 위상오차로 판단한다.

## 11. 필수 ablation

### 11.1 Advection과 context L2

먼저 다음 세 구성을 순서대로 평가한다.

```text
no advection, no L2
advection, no L2       # v1.1.3b default
advection, sparse L2   # only if residual boundary error remains
```

필요할 때만 `no advection + sparse L2`를 추가해 두 모듈의 중복효과를 확인한다.

Context 자체의 기여는 다음 ring-only ablation으로 추가 분리한다.

```text
full context encoder
ring-only context encoder: target과 겹치는 중앙 영역을 context-fusion branch에서 mask
no context fusion
```

Ring-only 실험에서도 optical flow와 advection artifact는 모든 arm에 동일한 full context로 만든다. Context encoder의 중앙 mask를 advection 입력에도 적용하면 fusion 기여와 flow 품질을 혼동하므로 금지한다.

### 11.2 ERA와 provider robustness

```text
radar/context only
full-trajectory ERA5 24ch oracle
target-end-causal ERA5 24ch oracle
full-trained model with target-end-causal mask only, labeled OOD occlusion
issue-causal occlusion: valid_time/interval_end <= t0, labeled OOD
(tp-present versus tp-dropout/occluded)
ERA dropout-trained model evaluated with provider absent
operational-provider hindcast after provider-specific fine-tuning and §4.2 contract
```

Full/causal/operational의 주 비교는 matched training-and-evaluation contract를 사용한다. `full >= target-end-causal >= operational`은 사전 가설로만 기록하고 결과를 그 순서에 맞추기 위해 calibration하거나 모델을 선택하지 않는다. Target-end-causal와 issue-causal OOD 차이는 미래 환경장 token에 대한 sensitivity 진단이며 인과효과로 부르지 않는다.

### 11.3 CorrDiff 기여

```text
regression-only A_reg_zinv_tau
full CorrDiff ensemble mean
full CorrDiff ensemble median
full CorrDiff probabilistic scores
```

Diffusion이 texture만 복원하고 위치·확률 skill을 악화시키는지 CRPS, FSS, spectrum과 calibration을 함께 확인한다. 포함 여부의 판정은 §3.5의 사전등록 diffusion 포함 gate를 따른다.

### 11.4 Residual scaling과 calibration

Lead별로 다음을 기록한다.

```text
s_oof[tau,condition_signature] from matching OOF residual
b[tau,condition_signature]: OOF-to-full residual location
c[tau,condition_signature]: OOF-to-full total residual scale
d[tau,condition_signature,sampler_core_signature]: optional finite-step mean correction
gamma[tau,condition_signature,ensemble_signature]: mean-preserving member spread
mean(r_tau) before and after b,c and optional d
mean_e0 = Mean_W(r_full - (b + c*rbar0))
bias_fraction = abs(mean_e0) / RMSE_W(r_full - (b + c*rbar0))
location_support_ratio = abs(b + d) / z_wet

z_before_location_tau_n = mu_z_full_tau + c * r0_tau_n
z_after_location_tau_n  = z_before_location_tau_n + b + d
location_flip_fraction = weighted_mean_{W_cal,n}(
    1[(z_before_location_tau_n >= z_wet)
      != (z_after_location_tau_n >= z_wet)]
)
Var(r_tau) / Var(z_tau)
1 - MSE(mu_z_tau, z_tau) / Var(z_tau)
spread-skill ratio before b,c, after b,c/d, and after gamma
calibration block count, ESS, pooling level and signature hashes
observed wet/dry-stratified residual and member z_hat distributions
(0, z_wet) band mass and above-z_wet leak mass per lead, pre-censoring
leak-sensitivity ratio z_wet / s_oof[tau,a]
```

`location_support_ratio > 0.5`인 calibration cell은 dry/wet support 침범 위험 flag를 붙인다. `location_flip_fraction`은 모든 lead·condition·sampler calibration cell에서 기본 기록하며 `gamma` 적용 전, valid calibration pixel과 member 축에서 계산한다. 두 값은 진단 전용이며 calibration/test를 본 뒤 `b`, `d`, censoring threshold 또는 model을 사후 수정하는 기준으로 사용하지 않는다. 관측 dry 층에서 member `z_hat`이 `(0, z_wet)` 밴드로 번진 질량은 censoring이 0으로 복원하므로 무해하고, `z_wet`을 넘는 leak 빈도가 주 진단이며 §8.2 저확률 `q_tau > p_tau` gap의 주요 후보 기구 중 하나다 — 이 gap에는 regression `p` calibration과 관측 wet에서의 under-threshold member도 기여한다. 두 질량과 `z_wet/s_oof` 비를 lead별로 기록하며, 이 진단이 §13.4 ladder 판정의 입력이다.

`d-enabled`는 `d=0` reference와 §3.4의 동일 primary score·guardrail로 비교하는 사전 등록 candidate다. `bias_fraction`은 기구 진단이지 단독 activation threshold가 아니다. 사용 여부는 model-selection validation에서 동결하며 calibration/test에서 새로 활성화하지 않는다. 선택된 arm에서는 `d` 적용 뒤 ensemble-mean error로 `gamma`를 fit한다.

Lead-normalized residual과 lead-dependent `sigma_data`를 동시에 적용하는 실험은 기본 ablation에서 제외한다.

### 11.5 OOF prediction과 encoder feature

OOF regression output이 full-data deployment model을 얼마나 잘 대리하는지는 두 단계로 기록한다. Outer-train cross-fit 진단과 model-selection validation에서 다음 값을 lead·regime·강도별로 비교해 K와 EDM-A/B를 선택한다.

```text
mean and quantiles of p_full - p_fold
mean and quantiles of mu_z_full - mu_z_fold
full/fold residual RMS ratio
residual radial-spectrum ratio
3-fold versus 5-fold residual-RMS change, when available
```

K와 모델을 동결한 뒤 calibration split에서는 같은 fold/full 차이를 다시 계산하되 `b,c`, 사전 선택된 경우의 `d`를 fit하고 shift를 audit하는 용도로만 사용한다. Calibration 결과로 K, encoder arm, `d` 사용 여부 또는 architecture를 바꾸지 않는다.

Diffusion condition encoder의 in-sample성을 다음 두 학습으로 분리한다.

```text
EDM-A: OOF p/mu/residual + advection + static
       full-train deployment encoder pyramid excluded
EDM-B: OOF p/mu/residual + frozen full-train deployment encoder pyramid
```

EDM-B가 v1.1.3b reference이고 EDM-A가 encoder in-sample성의 민감도 진단이다. EDM-A는 조건 용량도 작으므로 train score만 비교하지 않는다. Deployment pyramid 포함 여부는 model-selection validation의 §3.4 규칙으로만 결정한다. Calibration/test의 두 arm 결과는 동결된 결정을 진단·보고할 수 있지만 primary arm을 바꾸는 근거로 사용하지 않는다. EDM-B가 diffusion train에서만 크게 좋고 validation에서는 그렇지 않으면 encoder의 in-sample feature 효과를 의심한다. 동일 용량의 self-supervised/frozen encoder control은 후속 보강 arm으로 둘 수 있지만 도입 여부도 validation에서만 정한다.

### 11.6 Validity mask, sampling weight와 mask-shape audit

Regression/diffusion loss와 calibration objective는 architecture §11·§12.2.1의 동일한 미래 `M_target_tau`와 full-item importance weight `omega_i`를 사용한다. `M_target_tau`는 target-dependent objective와 metric 전용이며 model condition, noisy-state multiplication, cache와 inference API에서 발견되면 contract failure다.

```text
whole-scan missing으로 drop된 sample 수
pixel-QC invalid fraction, rho_invalid_max와 valid M_target_tau pixel 수
outer-train에서 rho_invalid_max로 제외된 item 수
validation/calibration/test에서 rho_invalid_max 초과 sample 비율
M_target_tau condition/API presence assertion = false
item = (t0, tau, condition_signature)
P_draw(item) = P(t0) * P(tau|t0) * P(signature|t0,tau)
P_target(item) and omega_i = P_target(item)/P_draw(item)
omega_i min/median/p95/max
weight clipping = none in v1.1.3b
weighted sample ESS = (sum omega_i)^2 / sum omega_i^2
timeline-uniform versus weighted event-balanced training flag
```

`rho_invalid_max`는 outer-train 학습 item 선정 전용이다. Validation, calibration과 final test에서 threshold 초과 sample 전체를 제거하지 않고 `M_target_tau` pixel만 제외한다. 최종 continuous-timeline arm의 calibrated occurrence를 목표로 할 때 class-weighted BCE, focal loss 또는 full-item draw probability를 보정하지 않은 wet/lead/signature oversampling은 허용하지 않는다.

Masked EDM의 neutral fill artifact를 다음 고정 audit로 평가한다.

```text
distance from valid pixel to nearest invalid target pixel, Chebyshev metric:
    near = 1-2 px
    mid  = 3-8 px
    far  = >8 px or no invalid pixel

low_sigma_band:
    declared training sigma distribution의 lower 20% boundary 이하

report by distance bin:
    intensity-matched clean-estimate MSE
    signed denoising bias
    MSE ratio versus far bin
    valid pixel and independent block counts
```

이 audit는 outer-train/model-selection 진단이며 fill 방식의 사후 수정 기준이 아니다. 실질적 artifact가 확인되면 새 protocol version에서만 imputation 또는 sample policy를 바꾼다.

### 11.7 Verification-time embedding ablation

다음 두 arm을 matched training으로 사전등록한다.

```text
e_time enabled   # v1.1.3b reference
e_time disabled  # same architecture/capacity; zero e_time enters the unchanged ConditionTimeFusionMLP
```

Enabled checkpoint에서 inference 시 `e_time=0`으로 만드는 결과는 OOD occlusion일 뿐 공식 ablation이 아니다. 두 arm은 각각 regression, OOF artifact와 diffusion을 학습한다. Disabled arm도 동일한 `ConditionTimeFusionMLP`와 parameter capacity를 유지하되 `e_time` 자리에 고정 zero vector를 사용하며, downstream ERA/regression/diffusion 인터페이스에는 두 arm 모두 `e_cond`만 전달한다. 두 arm은 §3.4의 동일 fair-CRPS primary endpoint와 guardrail로 비교한다.

Solar-hour bin은 mean-solar time 기준으로 고정한다.

```text
[00,06), [06,12), [12,18), [18,24) hours
```

Lead·solar-hour bin·season별 all-valid fair CRPS, wet-frequency bias, raw/calibrated reliability, initiation/decay case와 5 mm/8 km FSS를 기록한다. `e_time`은 e_time-enabled 공식 provider track과 배포 checkpoint family에서 radar-only 여부와 무관하게 항상 존재하고 condition signature에 포함하지 않는다. 이 문장의 유일한 예외는 위에서 별도로 재학습한 matched e_time-disabled checkpoint family다.


### 11.8 Joint 12-lead regression 대조 arm

Lead-conditioned 구조의 회귀 단계 이득을 검증하기 위해, 동일 encoder와 유사 parameter budget으로 12개 30분 누적장을 한 번의 forward로 출력하는 joint deterministic regression을 사전등록한다.

```text
joint arm:
    shared trunk + 12 x (p, m) output head
    trunk의 per-lead AdaLN 변조 불가
    시간 조건은 t0 기반(또는 head-level per-lead) 시간 feature로 대체

비교 항목:
    lead별 mu_z MSE와 A_reg_zinv MAE
    condition-bank 캐시 기준 per-lead 한계비용 대비 joint 1-forward 실측 latency
    인접 lead motion consistency
    expected-total 진단 (아래 정의)
```

이 비교에서는 lead-conditioning 효과와 `e_time` 주입 위치 효과가 교락됨을 결과 해석에 명기한다. Motion consistency는 인접 lead 예측장 쌍의 FFT cross-correlation argmax 변위(±32 km 탐색)를 관측 쌍의 변위와 비교하는 벡터 오차로 고정한다. Latency 비교는 naive 12회 forward가 아니라 §16.1 condition-bank 캐시를 사용한 per-lead 한계비용 실측을 기준으로 한다. Joint arm이 우세하더라도 diffusion 인터페이스는 바뀌지 않는다: joint 출력에서 lead별 `mu_z_tau`, `p_tau`를 추출해 per-lead residual diffusion을 그대로 조건화할 수 있으므로 채택 비용은 회귀 단계 교체로 한정된다.

중간 강도 비교군으로 shared encoder 1회 계산 + 12개 경량 lead-conditioned decoder query(각 decoder 내부에서 per-lead `e_cond` 허용)를 optional 사전등록 arm으로 둔다. 이 arm은 출력 구조의 효율과 trunk의 lead-awareness를 분해하는 유일한 비교군이며, Stage-0 범위에서 추가 regression 학습 비용이 드는 유일한 항목이다.

Expected-total 진단은 다음으로 고정한다: joint/regression 측은 `sum_tau A_mean_direct_tau`, CorrDiff 측은 `sum_tau A_ensmean_tau`. `sum_tau A_reg_zinv_tau`는 z-공간 평균 역변환의 합이라 기대 총량이 아니므로 사용하지 않는다. 서로 다른 lead의 member가 coherent scenario가 아니어도 marginal expectation의 합은 정의된다는 Phase 1 경계와 일치한다.

Motion-consistency FFT 규약에 다음을 고정한다: zero-padding으로 circular wrap 방지, 최소 wet-support 미달 쌍은 NA, dry 또는 모호한 correlation peak의 NA 판정 규칙, ±32 km 탐색 경계에 peak가 걸린 쌍의 비율 보고, subpixel peak interpolation 사용 여부 선언.

## 12. Source 사용 진단

Gate norm 하나만으로 source 사용량을 판단하지 않는다. 모델 단계, scale과 lead별로 다음을 기록한다.

```text
||g_source * Attention_source|| / ||target_feature||
attention entropy
source occlusion performance delta
ERA-present versus ERA-absent performance
tp-present versus tp-absent performance
full-trajectory versus target-end-causal temporal-access delta
target-end-causal versus issue-causal OOD delta
e_time enabled versus matched disabled delta
advection origin-in-domain and residence-confidence strata
```

짧은 lead에서 context 기여가 감소하고 긴 lead에서 ERA 기여가 증가할 것이라는 예상은 가설이다. Gate 궤적을 특정 방향으로 강제하거나 이를 성공조건으로 사용하지 않는다.

## 13. Calibration 규약

Calibration split은 regression/diffusion weight, architecture, K, solver, `d` 사용 여부, step 수 또는 member 수 선택에 사용하지 않는다. Model-selection validation에서 mapping family와 pooling rule까지 동결한 뒤, 사전 고정된 `final_primary_signature=32x12`의 frozen model·sampler에 대해 다음 parameter만 fit한다.

### 13.1 Residual location, total scale, sampler bias와 member spread

표기는 다음으로 고정한다.

```text
a = condition_signature
u = sampler_core_signature
  = (diffusion_or_distilled_checkpoint, solver, EDM_steps, sigma_schedule)
v = ensemble_signature = (u, N_members)
W_cal = M_target_tau * omega_cal
```

Calibration split을 declared target distribution으로 전수 평가하면 `omega_cal=1`이다. Subsampling했다면 학습과 마찬가지로 known `P_target/P_draw` ratio를 사용한다. 먼저 outer-train OOF residual RMS로 정규화된 diffusion member를 복원한다.

```text
r0_tau_n = s_oof[tau,a] * r_tilde_hat_tau_n
```

`s_oof`는 outer-train에서만 추정하며 calibration에서 재추정하지 않는다. 각 frozen fold regression과 full deployment regression을 calibration sample에 적용해 다음 true residual을 만든다.

```text
r_fold_k = z_tau - mu_z_fold_k
r_full   = z_tau - mu_z_full

H_k[tau,a] = sum over held-out OOF item i in fold k (
                 omega_i * M_target_tau_i
             )
pi_k[tau,a] = H_k / sum_j(H_j)

m_k = weighted_mean_Wcal(r_fold_k)
q_k = weighted_mean_Wcal(r_fold_k^2)

mean_fold_mix   = sum_k(pi_k * m_k)
second_fold_mix = sum_k(pi_k * q_k)
var_fold_mix    = max(second_fold_mix - mean_fold_mix^2, 0)

c[tau,a] = sqrt(
    weighted_var_Wcal(r_full) / max(var_fold_mix, epsilon)
)
b[tau,a] = weighted_mean_Wcal(r_full) - c[tau,a] * mean_fold_mix
```

`weighted_var_Wcal(x)`는 `weighted_mean_Wcal((x-weighted_mean_Wcal(x))^2)`인 weighted population central moment다. `gamma` 내부 member-axis `sample_variance(ddof=1)`와 혼동하지 않는다.

`pi_k`는 해당 lead·condition signature가 OOF artifact에서 차지하는 실제 weighted valid mass이며 fold가 정확히 균형일 때만 `1/K`다. 이 mixture moment는 fold 내부 분산뿐 아니라 fold별 residual mean 차이도 포함한다.

`b,c`를 고정한 뒤 finite-step sampler mean bias를 계산한다.

```text
rbar0_tau = mean_n(r0_tau_n)
e0_tau = r_full - (b[tau,a] + c[tau,a] * rbar0_tau)

mean_e0 = weighted_mean_Wcal(e0_tau)
rmse_e0 = sqrt(weighted_mean_Wcal(e0_tau^2))
bias_fraction = abs(mean_e0) / max(rmse_e0, epsilon)

if d_enabled:
    d[tau,a,u] = mean_e0
else:
    d[tau,a,u] = 0

r1_tau_n = b[tau,a] + d[tau,a,u] + c[tau,a] * r0_tau_n
```

`d_enabled`와 그 provisional bias 판정규칙은 model-selection validation에서 이미 동결되어 있어야 한다. Calibration의 `bias_fraction`은 audit이며 이 값으로 `d`를 새로 켜지 않는다. `b`는 OOF→full residual location, `c>0`는 total residual scale mapping, `d`는 sampler-core별 finite-step mean correction이다. 이를 고정한 뒤 exact ensemble signature `v`의 member anomaly만 조정한다.

```text
rbar1_tau = mean_n(r1_tau_n)
err_tau   = z_tau - (mu_z_full_tau + rbar1_tau)
S2_tau    = sample_variance_n(r1_tau_n, ddof=1)

gamma[tau,a,v]
    = sqrt(
        sum(W_cal * err_tau^2)
        / max(sum(W_cal * (1 + 1/N) * S2_tau), epsilon)
      )

r2_tau_n = rbar1_tau + gamma[tau,a,v] * (r1_tau_n-rbar1_tau)
z_hat_tau_n = mu_z_full_tau + r2_tau_n
```

`gamma>0`는 ensemble mean을 보존하는 spread-only factor다. 적용 순서는 `s_oof -> b,c -> optional d -> gamma -> mu_z_full addition -> physical inverse transform -> A_wet censoring`이며 바꿀 수 없다. Calibration 전 identity는 `b=0,c=1,d=0,gamma=1`이다.

### 13.2 Probability mapping

Probability calibration은 `p_tau`와 각 threshold의 raw ensemble fraction `q_T`를 각각 관측 사건에 맞춘다. 단지 `q_tau=p_tau`를 만들기 위한 보정은 금지한다. v1.1.3b의 기본 family는 다음 monotone logit-linear mapping이다.

```text
Cal(x) = sigmoid(alpha + beta * logit(clip(x, 1e-6, 1-1e-6)))
beta = softplus(beta_raw) > 0
fit objective = W_cal-weighted Bernoulli negative log likelihood
```

`p_cal`은 `A_wet` occurrence에 대해 regression checkpoint와 condition signature별로 fit한다. `q_cal_T`는 `T in {0.1,1,5}`와 조건·ensemble signature별로 fit하며, 10 mm threshold가 §7.2의 support 조건을 만족하면 보조로만 fit한다. 다른 mapping family를 쓰려면 model-selection validation 전에 후보군을 사전 등록하고 outer-train OOF로 임시 parameter를 fit해 선택한 뒤, calibration split에서는 선택된 family의 parameter만 새로 fit한다.

Probability mapping만으로 raw member field의 wet/dry support가 바뀌었다고 주장하지 않는다. 생성측 latent threshold를 실험하더라도 공식 output과 verification의 사건 threshold `A_wet`은 바꾸지 않는다.

### 13.3 Key와 sparse-cell fallback

Calibration record key는 다음과 같다.

```text
b,c:       tau, condition_signature, regression_checkpoint_pair
d:         tau, condition_signature, sampler_core_signature
gamma:     tau, condition_signature, ensemble_signature
p_cal:     tau, A_wet, condition_signature, regression_checkpoint
q_cal_T:   tau, threshold_T, condition_signature, ensemble_signature
```

다른 diffusion checkpoint, distilled checkpoint, solver, EDM step 수 또는 sigma schedule의 `d/gamma/q_cal`을 재사용하지 않는다. Member 수까지 다른 `gamma/q_cal`도 재사용하지 않는다. Condition cell이 희소하면 결과를 본 뒤 임의로 합치지 않고 다음 순서로 fallback한다.

```text
for independent block g:
    B_g = sum of W_cal over valid entries in block g
    block_ESS = (sum_g B_g)^2 / sum_g(B_g^2)

full lead x condition-signature cell 사용 조건:
    independent event/dry/marginal blocks >= 30 and block ESS >= 20

부족하면 순서대로 pool:
    lead x provider x era_present
    lead x provider
    lead only
```

Probability threshold `T`에서 valid entry 중 하나라도 `A_obs>=T`인 block을 positive-support block, 하나라도 `A_obs<T`인 block을 negative-support block으로 센다. 한 block이 둘 모두에 포함될 수 있다. Probability cell은 위 조건과 함께 positive-support와 negative-support block이 각각 20개 이상이어야 한다. 각 record에 block 수, positive/negative block 수, ESS, pooling level, model/checkpoint/config signature hash를 저장한다. Hierarchical shrinkage를 사용하려면 방법과 prior를 model-selection 단계에서 먼저 동결한다.

`s_oof`의 sparse-signature fallback도 같은 순서를 쓰되 block 수와 ESS는 calibration이 아니라 outer-train OOF artifact에서 계산한다. `s_oof`의 pooling level은 이후 `b,c,d,gamma` record와 함께 저장한다.

`lead only` cell조차 block/ESS 조건을 만족하지 못하면 해당 mapping은 identity(`b=0, c=1, d=0, gamma=1`, probability mapping은 raw 통과)로 두고 결과에 `uncalibrated` 표기와 terminal-fallback flag를 기록한다. Identity 강제는 보정 실패의 은폐가 아니라 명시이며, 이 flag가 붙은 cell의 score는 calibrated cell과 같은 표에서 구분 표기한다.

`s_oof` 자체의 terminal 실패는 identity로 대체할 수 없다. `s_oof`의 lead-only cell마저 outer-train OOF block/ESS 조건을 만족하지 못하면 normalized diffusion 출력을 residual scale로 복원하는 것 자체가 불가능하므로, 해당 cell은 diffusion 사용 불가로 선언하고 regression-only forecast를 사용하며(`p_cal`이 지원되면 `p_cal`, 아니면 raw `p`), `diffusion_scale_unsupported` flag를 기록한다. 임의의 전역 residual scale 대입은 금지한다.

§3.5 gate와의 상호작용: calibrated gate는 calibrated cell mass에서 판정하되, 해당 lead의 `uncalibrated` 또는 `diffusion_scale_unsupported` weight 비율이 선언 상한 `1%`를 넘으면 그 lead의 calibrated gate는 실패로 처리한다. Raw pass-through 확률을 `q_cal`/`p_cal` 이름으로 보고하지 않으며, aggregate에서 calibrated와 uncalibrated mass를 분리 표기한다. 상한 `1%`는 사전 선언 기본값으로 model-selection 시작 전에 비준한다.

### 13.4 Occurrence 불일치 escalation

관측 대비 occurrence 불일치가 큰 경우 다음 순서로 확장한다.

```text
1. lead/provider별 p_tau와 raw q_tau probability calibration
2. residual mean/scale 및 finite-step sampling bias 재점검
3. denoised clean estimate의 occurrence-consistency auxiliary loss
4. 그래도 member occurrence가 실패할 때만 explicit wet-mask/hurdle generator 검토
```

1단계의 calibrated probability는 raw member wet fraction과 이름을 분리해 `q_cal`처럼 표기한다. Probability mapping만으로 member field 자체의 wet/dry 구성이 바뀌었다고 주장하지 않는다.

Ladder 4단 진입 조건은 지금 상수로 고정하지 않되, 규칙 개발과 규칙 적용의 데이터를 분리한다. 1–3단의 실패 판정 규칙 — 어느 stratum·lead에서 calibrated reliability와 occurrence gap이 어떤 margin을 넘으면 실패로 보는지, §11.4 leak 질량을 어떻게 반영하는지 — 은 outer-train 내부(OOF artifact와 outer-train pilot)에서만 개발해 margin과 pooling까지 동결한다. Model-selection validation은 이미 동결된 규칙을 적용해 hurdle/non-hurdle candidate를 비교하는 데만 사용한다. Validation 결과를 보고 규칙을 새로 정하면 규칙 선택, hurdle 도입 결정과 candidate 평가가 같은 데이터에서 일어나는 선택 누수가 된다. Calibration/test에서는 규칙 변경을 금지한다.

Calibration 방식, parameter 수와 exact signature를 기록하고 test에는 한 번만 적용한다. Test 결과를 보고 calibration을 다시 조정하지 않는다.

## 14. Lead 간 독립성의 평가 한계

Phase 1 member index는 lead 사이에서 같은 시나리오를 나타내지 않는다. 따라서 member `n`의 12개 lead를 합쳐 6시간 총누적 ensemble이나 유역별 scenario를 만들고 CRPS를 계산하면 안 된다.

12개 lead의 물리단위 ensemble mean을 합친 deterministic expected-total field는 별도 진단으로 계산할 수 있지만, 이를 coherent 6시간 probabilistic forecast라고 부르지 않는다.

## 15. 최소 보고표

최종 보고에는 최소한 다음 표와 그림을 포함한다.

```text
1. lead별 MAE/RMSE/bias: persistence, advection, pySTEPS, transformed regression, direct physical mean/q50, CorrDiff mean/median
2. threshold 및 lead별 CSI/POD/FAR/bias/ETS
3. threshold 및 2/8/32 km별 FSS
4. lead별 pixel 및 8/32 km neighborhood-aggregated fair CRPS/CRPSS, Brier/BSS와 fixed-bin reliability
5. lead별 spread-skill와 randomized-tie rank histogram
6. member 및 ensemble-mean radial spectrum
7. advection/L2/ring-only, ERA tp/dropout, regression/diffusion와 e_time ablation
8. event/strict-dry/marginal paired-block bootstrap 95% interval
9. fixed case-study category 결과, 겨울 cold-surface proxy 포함
10. radar-only, full-trajectory oracle, target-end-causal oracle와 operational-provider 결과의 분리
11. OOF/full output shift, fold-mixture s_oof/b/c, sampler d/gamma, abs(b+d)/z_wet, location_flip_fraction과 EDM-A/B 진단
12. p_tau와 q_tau의 signed/absolute occurrence gap, solar-hour·cold-surface strata 포함
13. selection 16x8, final-primary 32x12, operational 8x4-distilled signature별 latency/throughput/memory
14. loss-only M_target_tau, rho_invalid_max scope, full-item target/draw weight와 weighted ESS audit
15. event/strict-dry/marginal partition, speckle, boundary embargo와 unassigned_eligible_item_fraction assertion
16. Appendix A AWS 30분 누적 comparison annex
17. operational-provider effective forecast-cycle/availability age와 required rollout-lead 분포
18. screening/finalist training-seed stability, sigma_seed_pair, paired_seed_delta_std와 common-ensemble-seed 기록
19. Stage-0 regression funnel 결과와 cull 기록
20. §3.5 절대 baseline gate(AND-합성)와 diffusion 포함 gate 판정, uncalibrated/unsupported mass 비율, escalation 발동 여부
21. OOF-full condition-swap 진단(집계 swap_delta 대 sigma_selected, 상대효과, lead·stratum CI)
22. joint 12-lead regression 대조와 motion-consistency 벡터 오차
23. wet/dry 층별 leak 질량과 z_wet/s_oof 감수성 비
24. EDM-A/B E_SSR·RI dispersion 비교와 pair tie-break 경로
25. training_draw_manifest hash와 OOF materialization coverage
26. block-support 진단: 독립 event block 수, effective count, 최대 block weight, low-support 표기
27. deployment_training_seed checkpoint의 단독 score와 3-seed 평균 병기
```

## Appendix A. AWS 30분 누적 comparison annex

AWS 자료가 준비된 경우 다음 분석을 사전등록된 annex로 수행한다. 이 annex는 model selection, calibration, threshold, grouping 또는 checkpoint 선택에 사용하지 않는다.

```text
time alignment:
    KMA/AWS timestamp와 HSR verification interval을 동일 30분 구간으로 정렬

station QC:
    missing, gauge reset, physically impossible accumulation과 metadata 오류 제외

spatial collocation:
    station containing 500 m target pixel
    plus fixed 2 km neighborhood radar mean as sensitivity

metrics:
    mean bias, median bias, MAE, correlation
    wet-event contingency at 0.1/1/5 mm per 30 min
    quantile and seasonal bias
    cold-surface proxy versus warm-surface strata
```

Station selection, QC, collocation radius와 minimum sample count는 AWS 결과를 보기 전에 annex config에 고정한다. 결과는 HSR-derived target의 대표성 한계를 해석하는 용도이며 AWS를 이용해 이미 동결된 model 또는 HSR score를 재보정하지 않는다.

## 16. v1.1.2 보완 기록

```text
Manifest:
    significant wet component를 event seed에 추가
    strict dry와 marginal/background를 분리
    speckle·weak context를 제거하지 않고 operational estimand에 포함

Time condition:
    t_c 중심 mean-solar/annual e_time reference와 matched ablation 추가

Scores:
    pixel·neighborhood·primary CRPS를 finite-ensemble fair CRPS로 변경
    서로 다른 N의 raw q Brier 직접 비교 금지

Validity and truth:
    rho_invalid_max를 outer-train 학습 선정으로 제한
    HSR-derived truth와 AWS annex, cold-surface proxy를 명시

Reproducibility and provider:
    common-seed screen + reference/finalist 3-seed policy
    issue-causal OOD diagnostic
    operational cycle/latency contract와 effective forecast age 보고
```

## 17. v1.1.2a freeze-consistency patch 기록

```text
Verification-time interface:
    e_time은 모든 공식 provider/deployment family에서 사용
    separately trained matched-disabled family만 예외
    disabled family도 동일 ConditionTimeFusionMLP 용량을 유지
    downstream module에는 e_cond만 전달

Finalist governance:
    mandatory guardrail을 동일 3-seed 평균으로 재검증
    5 mm/8 km FSS guardrail field를 primary A_ensmedian으로 명시

Calibration and seed audit:
    location_flip_fraction을 필수 기록으로 승격
    paired_seed_delta_std를 common-seed diagnostic으로 추가
    practical-tie 결정은 사전 고정 sigma_seed_pair를 유지
```

## 18. v1.1.3 통합 기록

```text
Data contract:
    pixel QC 판정 계약을 architecture §2.2 단일 기준으로 연결
    manifest seed 계산에 sentinel 제외 선행
    trapezoid 고정과 scan 시간의미 확인의 blocking 등록

Selection governance:
    Stage-0 regression funnel 사전등록, diffusion-side sweep은 artifact 공유
    §3.5 절대 baseline gate: lead 대역별 radar/pySTEPS/coarse-tp/climatology 짝
    diffusion 포함 gate로 §11.3을 진단에서 판정으로 승격
    EDM-A 자동 3-seed finalist와 A/B dispersion non-inferiority

Diagnostics:
    OOF-full condition-swap (sigma_seed_pair 판정)
    wet/dry 층별 (0,z_wet)·초과 leak 질량과 z_wet/s_oof 비
    hurdle trigger의 절차적 동결 (임의 상수 금지)
    joint 12-lead regression 대조와 motion-consistency metric
    group duration/JJA block 보고, terminal calibration fallback 정의

Convention:
    repo canonical path = docs/k_corrdiff_architecture.md,
    docs/k_corrdiff_evaluation.md; 버전은 header 전용,
    접미 사본은 아카이브 전용
```

## 19. v1.1.3a 교정 기록

```text
§3.5:  baseline AND-합성, CRPS_det = |x-y| 정의, FSS 절대 margin -0.01,
       q_cal 대 p_cal 명명 분리
§3.5:  diffusion 포함 gate의 E_PSD(member-spectrum 평균)·E_freq5 scalar화,
       보조 metric CI 판정
§3.4.1: EDM-A/B의 E_SSR·RI 정의, 절대 margin +0.05, 두 metric 모두 요구,
       pair tie-break = CRPS → dispersion → latency
§13.3: s_oof terminal 실패 시 regression-only fallback과
       diffusion_scale_unsupported, fallback mass 상한 1% gate 연동
§13.4: hurdle 실패규칙 개발을 outer-train 내부로 한정, validation은 적용 전용
architecture §12.1/§12.3/§16.5/§29:
       training_draw_manifest, 통제된 condition-swap,
       benchmark의 evidence 재규정, freeze ledger
```

## 20. v1.1.3b governance 마감 기록

```text
§3.4.1: deployment_checkpoint_rule — seed 11103 사전 고정, best-seed 금지,
        checkpoint hash binding, 단독 score 병기
§3.4.1: EDM-A/B 종결 경로 — B 부적격 시 A 조건부 승격, 아니면
        regression-only 배포, lexicographic tie 규칙, E_SSR pooled+eps
§3.4.0: Stage-0 E1/E2 분리 표기, candidate당 diffusion 1회 정정,
        ERA family occlusion 진단 추가
§3.5:  절대 gate deterministic baseline = A_q50_direct,
        inclusion gate parent = A_reg_zinv, band 등가중 집계,
        lead-averaged BSS gate, selection-interval 명명
§5.7:  coarse-tp 30분 interval-overlap 변환 계약
§11.8: 중간 비교군(경량 per-lead decoder), expected-total 정의,
        FFT 위생 5종
§3.2.2: block-support 보고와 low-support CI 표기 (독립 block < 20)
architecture: item-level draw key(global_example_index),
        weighted fair CRPS의 A_mix, fold-독립 noise와 fold별 CRN 진단 분리,
        sigma_selected threshold, BLOCKING-3 governance_constants,
        z_wet/s_oof proxy 강등, radar-quality static metadata backlog
```
