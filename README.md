# MCI 분류 파이프라인 (ADNI-3, 3D MRI) — ROI 기반 3D CNN 고도화 기록

---

# AD 3D CNN - 2-Stage Inference + ROI Attention

뇌 MRI NIfTI를 대상으로 2단계 분류(CN vs CI, MCI vs AD)와 ROI 기반 CAM/SHAP 시각화를 수행하는 노트북 기반 파이프라인입니다.

CI는 MCI+AD를 의미합니다.

---

## 0. 실험 흐름 요약

1) **ROI: 3D 시각화를 통한 주요 진단 부위 지정**  
- ROI 중심좌표/크기(σ)/가중치(weight)를 정의하고, MRI 슬라이스(축/관상/시상)에 **가중치 맵 오버레이** 로 좌표가 해부학적으로 타당한지 반복 검증.

2) **Classic CNN으로 ROI 부위별 중요도 조절**  
- ROI를 **학습 가능한 모듈이 아니라 고정 prior** 로 두고, 입력 `x`에 대해 `x ⊙ weight_map` 형태의 **ROISpatialAttention**을 적용(“classic”한 방식의 공간 가중치).

3) **Training 파라미터 조정**  
- `TARGET_SHAPE` 축소(메모리 안정화), `BATCH_SIZE` + `GRAD_ACCUM_STEPS` 로 effective batch 확대  
- `AdamW` + `CosineAnnealingLR`, gradient clipping, early stopping, 불균형 대응(WeightedRandomSampler / class weight / FocalLoss)

4) **모델 고도화(Attention/Transformer/ResNet/MedicalNet/LSTM/GRU/Hybrid)**  
- 기본 3D CNN → Channel Attention 추가 → (선택) Res2Net3D / ResNet3D(MedicalNet pretrain) →  
  CNN+Transformer / CNN+LSTM/GRU / Transformer+LSTM/GRU 등 시퀀스 모델 실험  
- 최종적으로는 **ROI Spatial Attention + Channel Attention 기반 모델**을 최종안으로 사용.

5) **삼진분류(CN/MCI/AD)도 동일한 과정으로 진행했으나 MCI가 잘 안 잡힘**  
- 동일한 ROI prior + 학습 루틴을 3-class로 확장했지만, **MCI 예측이 붕괴(특정 클래스 쏠림)** 하는 현상을 확인.


## 1. ROI 정의 (최종 22개 ROI)

- 좌표계: `TARGET_SHAPE = (D, H, W) = (96, 112, 96)` 기준  
- 좌표 의미(코드 주석 기준)
  - `D`: Inferior → Superior
  - `H`: Posterior → Anterior
  - `W`: Left → Right

### ROI 리스트 (center / σ / weight)

| ROI                 | Center [D,H,W]   | σ(크기) |Weight(가중치) | Description                         |
|:--------------------|:-----------------|-----:  |---------:   |:------------------------------------|
| entorhinal_L        | 54,48,34         |  3.5   |      2      | 좌측 엔토라이날                         |
| entorhinal_R        | 54,48,62         |  3.5   |      2      | 우측 엔토라이날                         |
| hippocampus_L       | 48,54,36         |  4     |      1.9    | 좌측 해마                             |
| hippocampus_R       | 48,54,60         |  4     |      1.9    | 우측 해마                             |
| parahippocampal_L   | 48,42,32         |  4     |      1.8    | 좌측 해마방회                          |
| parahippocampal_R   | 48,42,64         |  4     |      1.8    | 우측 해마방회                          |
| inferior_horn_L     | 35,56,30         |  8     |      1.6    | 좌측 하각 - 해마 위축 지표               |
| inferior_horn_R     | 35,56,66         |  8     |      1.6    | 우측 하각                             |
| posterior_cingulate | 30,35,48         |  5     |      1.6    | 후대상피질                             |
| precuneus           | 65,30,48         |  6     |      1.6    | 설전부 - 상두정엽                       |
| amygdala_L          | 42,60,34         |  3.5   |      1.5    | 좌측 편도체                            |
| amygdala_R          | 42,60,62         |  3.5   |      1.5    | 우측 편도체                            |
| fusiform_L          | 38,58,28         |  6     |      1.5    | 좌측 방추상회                          |
| fusiform_R          | 38,58,68         |  6     |      1.5    | 우측 방추상회                          |
| inferior_temporal_L | 42,60,22         |  8     |      1.5    | 좌측 하측두회                          |
| inferior_temporal_R | 42,60,72         |  8     |      1.5    | 우측 하측두회                          |
| lateral_ventricle   | 35,56,48         | 12     |      1.4    | 측뇌실 - 위축 보상 확장                  |
| middle_temporal_L   | 48,52,24         |  8     |      1.4    | 좌측 중측두회                          |
| middle_temporal_R   | 48,52,72         |  8     |      1.4    | 우측 중측두회                          |
| parietal_L          | 68,45,26         |  8     |      1.3    | 좌측 두정엽 - temporo-parietal 확장     |
| parietal_R          | 68,45,70         |  8     |      1.3    | 우측 두정엽                            |
| frontal             | 55,85,48         | 10     |      1.1    | 전두엽                                |

---

## 2. ROI Weight Map 생성 로직 (핵심 아이디어)

### 2.1 3D Gaussian 기반 가중치 맵

각 ROI를 중심점 `center`와 `sigma`를 갖는 3D Gaussian으로 모델링합니다.

- 거리 제곱: `dist_sq = (d-cd)^2 + (h-ch)^2 + (w-cw)^2`
- 가중치 기여: `exp(-dist_sq / (2*sigma^2)) * (roi_weight - base_weight)`
- 최종 맵: 모든 ROI 기여를 합산하여 `weight_map`을 구성

### 2.2 Classic ROI Spatial Attention (입력에 곱)

`ROISpatialAttention` 은 아래처럼 **입력 볼륨에 가중치 맵을 element-wise로 곱**합니다.

- `forward(x) = x * weight_map`
- 학습 파라미터가 없는 **고정 prior** 로써 동작

---

## 3. ROI 위치 검증(시각화) & ROI 기반 해석(Grad-CAM)

### 3.1 ROI Weight Map 오버레이 (좌표 검증)

가중치 맵이 해부학적 위치에 제대로 얹히는지 확인하기 위해:
- Axial(여러 D slice), Coronal(H 중앙), Sagittal(W 좌/우) 슬라이스에
- `gray` MRI + `hot` weight_map(α blending) 를 겹쳐 시각화

### 3.2 Grad-CAM(3D) + ROI anomaly score(정량)

모델이 실제로 ROI에 반응하는지 확인하기 위해 3D Grad-CAM을 생성하고,
각 ROI 영역(gaussian mask)에서 활성화 통계를 계산합니다.

- `GradCAM3D`: target layer의 activations/gradients hook → CAM 생성
- `compute_roi_anomaly_scores`
  - `mean_activation`, `max_activation`, `anomaly_pct(%)` 산출

---

## 4. 모델 구성 및 고도화 과정

### 4.1 Baseline (3D CNN)

- `MCIClassifier3D`
  - `ConvBlock3D(1→32→64→128→256)` + GAP + MLP classifier

### 4.2 Channel Attention 추가 (최종 채택)

- `MCIClassifierWithAttention`
  - Baseline 3D CNN encoder 뒤에 `ChannelAttention(256)` 추가
  - **ROI Spatial Attention + Channel Attention** 조합이 최종안으로 사용

### 4.3 Res2Net3D / ResNet3D (Backbone 확장)

- `Res2Net3D`: multi-scale bottleneck(Res2Net) 기반 3D backbone
- `ResNet3D(resnet50_3d)`: Bottleneck3D 기반 3D ResNet

### 4.4 MedicalNet 사전학습(ResNet50 3D) + 2-stage 학습

- `load_pretrained_weights` 로 MedicalNet 체크포인트 로딩(불일치 파라미터 필터링)
- Stage 1: backbone freeze → head-only 학습 (`freeze_backbone`)
- Stage 2: 전체 unfreeze → fine-tune (`unfreeze_all`)

### 4.5 Sequence/Hybrid 모델 (Transformer/LSTM/GRU)

MRI의 깊이(D)를 sequence로 보고 다음을 실험했습니다.

- `MCIClassifierCNNTransformer`
- `MCIClassifierCNNLSTM`, `MCIClassifierCNNGRU`
- `MCIClassifierTransLSTM`, `MCIClassifierTransGRU`

핵심 아이디어(예: TransLSTM):
- 3D CNN encoder 출력 `x`에서 `H,W` 평균 → `[B, D, C]` 시퀀스 생성
- sinusoidal positional encoding 추가 → Transformer encoder 통과
- LSTM/GRU로 sequence 요약 → classifier

---

## 5. 학습 설정(대표 값) & 파라미터 튜닝 포인트

### 5.1 대표 설정(2-class, attention 실험 기준)

- Target shape: `(96, 112, 96)`
- Batch size: `8`, grad accum `4` → effective `32`
- Optimizer: `AdamW(lr=1e-4)`
- Loss: `CrossEntropyLoss()`
- WeightedRandomSampler: `True`
- Gradient clipping: `clip_grad_norm_(1.0)`

### 5.2 불균형 대응(loss / sampler)

불균형이 심할 때 다음 옵션을 실험했습니다.
- `WeightedRandomSampler`
- `Weighted CrossEntropyLoss(weight=class_weights)`
- `FocalLoss(alpha, gamma)`

---

## 6. 3-class(CN/MCI/AD) 확장과 이슈

### 6.1 구현 방식

- label mapping: `{"CN":0, "MCI":1, "AD":2}`
- 모델/ROI prior/학습루틴은 2-class와 동일하게 재사용
- 평가: macro F1, per-class F1, confusion matrix, multi-class AUC(OVR)

### 6.2 관찰된 문제 (MCI를 전혀 못 잡는 현상)

실험 메모 기준으로,
- 모델이 **CN/AD 쪽으로 쏠리거나**, MCI를 거의 예측하지 않는 현상이 발생했습니다.

가능 원인(현상 설명용):
- CN↔MCI, MCI↔AD 사이 **영상학적 경계가 연속적**(overlap)이라 decision boundary가 불안정
- MCI 내부 이질성(EMCI/LMCI 혼재), 레이블 노이즈/메타데이터 매칭 불완전
- class imbalance(특히 AD가 소수) + ROI prior가 특정 방향(AD)으로 bias를 줄 가능성
