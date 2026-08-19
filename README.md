# K-ART-4L · 한국 미술 작품 4-Layer 자동 해설 및 유사작 검색 엔진

> 한 점의 그림을 **도상 · 기법 · 물질성 · 색채** 네 층위로 분해해 해설을 생성하고,
> 층위별 임베딩을 결합해 "무엇이 닮았는가"를 설명 가능한 형태로 검색하는 멀티모달 시스템

<p align="left">
  <img src="https://img.shields.io/badge/AIFFEL-Online%2018th-000000" alt="AIFFEL">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License">
</p>

---

## 목차

- [왜 이 프로젝트인가](#왜-이-프로젝트인가)
- [4-Layer 프레임워크](#4-layer-프레임워크)
- [시스템 아키텍처](#시스템-아키텍처)
- [Layer-Query Attention (LQA)](#layer-query-attention-lqa)
- [데이터](#데이터)
- [실험 결과](#실험-결과)
- [설치 및 실행](#설치-및-실행)
- [프로젝트 구조](#프로젝트-구조)
- [한계와 향후 과제](#한계와-향후-과제)
- [팀](#팀)
- [참고 문헌](#참고-문헌)

---

## 왜 이 프로젝트인가

미술관 현장에서 작품 해설은 여전히 사람의 손으로 쓰인다. 문제는 **해설이 왜 그렇게 쓰였는지가 기록되지 않는다**는 점이다. "정적인 화면 구성"이라는 한 문장이 구도에서 나온 판단인지, 채도에서 나온 판단인지, 재료의 물성에서 나온 판단인지가 남지 않는다.

기존 이미지 캡셔닝 모델을 미술 작품에 적용하면 이 문제가 그대로 재현된다. 모델은 단일 벡터로 그림 전체를 뭉뚱그려 표현하고, 생성된 문장의 근거를 되짚을 수 없다. 유사작 검색도 마찬가지다. 시각적으로 비슷하다는 결과는 나오지만, **어떤 차원에서 비슷한지**는 알 수 없다.

K-ART-4L은 이 문제를 **표현을 층위로 나누는 것**으로 접근한다. 작품을 네 개의 독립된 층위로 인코딩하면 해설은 층위별 근거를 갖게 되고, 검색은 "색채는 유사하지만 기법은 다른 작품"처럼 축을 지정할 수 있게 된다.

---

## 4-Layer 프레임워크

미술사 기술(記述)의 관행 — 파노프스키의 도상해석학과 재료·기법 중심 보존과학 기술 체계 — 을 계산 가능한 네 축으로 재구성했다.

| Layer | 이름 | 포착 대상 | 대표 특징 |
|:-----:|------|-----------|-----------|
| **L1** | 도상 (Iconography) | 무엇이 그려졌는가 | 소재, 모티프, 주제, 상징, 인물·사물 배치 |
| **L2** | 기법 (Technique) | 어떻게 그려졌는가 | 필치, 붓자국 방향성, 레이어링, 마티에르 |
| **L3** | 물질성 (Materiality) | 무엇으로 만들어졌는가 | 지지체, 매체, 표면 질감, 물성적 흔적 |
| **L4** | 색채 (Color) | 어떤 색으로 조직되었는가 | 색상 분포, 채도·명도 구조, 대비, 조화 관계 |

각 층위는 **독립적으로 인코딩되지만 상호 제약한다.** 예컨대 L3의 재료 판단은 L2의 기법 표현 가능 범위를 제한하고, L4의 색채 구조는 L3의 매체 특성에 종속된다. 이 상호 제약을 모델링하기 위해 도입한 것이 LQA다.

---

## 시스템 아키텍처

```
                        ┌─────────────────┐
                        │   Input Image   │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  Vision Backbone │   (ViT / CLIP image encoder)
                        │   shared trunk   │
                        └────────┬────────┘
                                 │  patch tokens
              ┌──────────┬───────┴───────┬──────────┐
              │          │               │          │
         ┌────▼───┐ ┌───▼────┐    ┌─────▼───┐ ┌───▼────┐
         │ L1 Head│ │ L2 Head│    │ L3 Head │ │ L4 Head│
         │ 도상   │ │  기법  │    │ 물질성  │ │  색채  │
         └────┬───┘ └───┬────┘    └─────┬───┘ └───┬────┘
              │         │               │         │
              └─────────┴───────┬───────┴─────────┘
                                │
                    ┌───────────▼────────────┐
                    │ Layer-Query Attention  │  ← 층위 간 상호 제약
                    │        (LQA)           │
                    └───────────┬────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
        ┌────────▼─────────┐        ┌──────────▼─────────┐
        │  Caption Decoder │        │  Retrieval Index   │
        │  층위별 해설 생성 │        │  FAISS · 가중 결합  │
        └────────┬─────────┘        └──────────┬─────────┘
                 │                             │
        ┌────────▼─────────┐        ┌──────────▼─────────┐
        │  4-Layer 해설문   │        │  유사작 + 근거 층위 │
        └──────────────────┘        └────────────────────┘
```

**핵심 설계 결정**

1. **공유 백본 + 층위별 헤드** — 층위마다 별도 백본을 두면 파라미터가 4배가 되고, 데이터가 희소한 조건에서 각 헤드가 충분히 학습되지 않는다. 백본을 공유하고 헤드만 분기시켜 표현 학습의 효율을 확보했다.
2. **검색 시 층위 가중치 노출** — 사용자가 `w = (w₁, w₂, w₃, w₄)` 를 조정해 "색채 중심 검색", "기법 중심 검색"을 선택할 수 있다. 검색 결과에는 어느 층위가 유사도에 기여했는지가 함께 반환된다.

---

## Layer-Query Attention (LQA)

네 층위를 완전히 독립적으로 다루면 서로 모순되는 해설이 생성된다(예: L3가 수묵을 지목했는데 L2가 임파스토 붓질을 기술하는 경우). 반대로 완전히 결합하면 층위 분리의 의미가 사라진다.

LQA는 각 층위 표현을 **query**로, 나머지 층위 표현을 **key/value**로 두어 층위 간 정보를 선택적으로 참조하게 한다.

```python
# core/lqa.py  (개념 코드)
class LayerQueryAttention(nn.Module):
    """각 레이어가 나머지 레이어를 조회하여 표현을 보정한다."""

    def __init__(self, dim: int, n_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.n_layers = n_layers
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        # 층위 정체성을 보존하기 위한 학습 가능 임베딩
        self.layer_emb = nn.Parameter(torch.randn(n_layers, dim) * 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, n_layers, dim) — 층위별 표현
        return: (B, n_layers, dim) — 상호 제약이 반영된 표현
        """
        z = z + self.layer_emb.unsqueeze(0)      # 층위 식별자 주입
        ctx, _ = self.attn(query=z, key=z, value=z)
        return self.norm(z + ctx)                # residual: 원 층위 정보 보존
```

`residual` 연결을 유지하는 것이 중요하다. 이것이 없으면 attention이 네 층위를 평균으로 수렴시켜 층위 구분이 무너진다. <!-- TODO: ablation 수치로 뒷받침 -->

---

## 데이터

### 마주친 벽

이 프로젝트의 최대 난관은 모델이 아니라 데이터였다. **한국 미술 작품에 대해 층위별로 주석이 달린 공개 데이터셋은 존재하지 않는다.** 착수 후 확인한 사실이며, 이로 인해 초기 설계를 여러 차례 수정해야 했다.

| 후보 소스 | 규모 | 층위 주석 | 판정 |
|-----------|------|-----------|------|
| 국립현대미술관 소장품 공개 API | <!-- TODO --> | ✗ | 이미지 + 기본 메타만 |
| 공공데이터포털 미술관 소장품 | <!-- TODO --> | ✗ | 메타데이터 품질 편차 큼 |
| WikiArt / SemArt (해외) | 약 8만+ | 부분적 | 도메인 불일치 (서양화 편중) |
| e-뮤지엄 통합 검색 | <!-- TODO --> | ✗ | 라이선스 확인 필요 |

### 대응 전략

정면 돌파(대규모 주석 구축)가 기간 내 불가능하다고 판단하고, 세 갈래로 우회했다.

1. **약지도 학습 (weak supervision)** — 기존 소장품 메타데이터(재료, 기법, 크기 필드)를 L2·L3의 노이즈 있는 레이블로 활용
2. **규칙 기반 부트스트래핑** — L4(색채)는 이미지에서 직접 산출 가능하므로 컬러 히스토그램·군집 기반으로 의사 레이블 생성
3. **소규모 골드셋 구축** — 큐레이터 도메인 지식을 투입해 <!-- TODO: N --> 점에 대해 4층위 수기 주석을 작성, 평가 전용으로 사용

> **기록해 둘 것** — 3번은 팀에 미술 전공자가 있었기에 가능했던 선택이다. 도메인 전문성이 데이터 부재를 부분적으로 상쇄할 수 있다는 것이 이 프로젝트의 부수적 발견이었다.

---

## 실험 결과

<!-- TODO: 아래 표를 실제 실험 수치로 교체 -->

### 해설 생성

| 모델 | BLEU-4 | ROUGE-L | CIDEr | 층위 일관성* |
|------|:------:|:-------:|:-----:|:-----------:|
| Baseline (단일 캡셔닝) | – | – | – | – |
| 4-Layer (LQA 없음) | – | – | – | – |
| **4-Layer + LQA** | – | – | – | – |

\* 층위 일관성: 생성된 네 해설 간 모순 문장 비율의 역수 (자체 정의 지표)

### 유사작 검색

| 방식 | Recall@10 | mAP | 층위 설명 가능성 |
|------|:---------:|:---:|:---------------:|
| CLIP 단일 임베딩 | – | – | ✗ |
| 4-Layer 균등 가중 | – | – | ✓ |
| **4-Layer + 가중 조정** | – | – | ✓ |

### 정성 평가

<!-- TODO: 검색 결과 스크린샷 또는 예시 이미지 삽입
     예) docs/figures/retrieval_example.png -->

---

## 설치 및 실행

### 환경

```bash
git clone https://github.com/utopiz44/k-art-4l.git
cd k-art-4l

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 데이터 준비

```bash
# 소장품 메타데이터 수집 및 이미지 다운로드
python scripts/collect_data.py --source mmca --out data/raw

# 4층위 전처리 (색채 의사 레이블 생성 포함)
python scripts/preprocess.py --in data/raw --out data/processed
```

### 학습

```bash
python train.py \
    --config configs/lqa_base.yaml \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4
```

### 추론 — 단일 작품 해설

```bash
python infer.py --image samples/example.jpg --mode caption
```

<details>
<summary>출력 예시</summary>

```json
{
  "iconography": "화면 중앙에 원형 구조가 반복 배치되며 ...",
  "technique":   "붓의 방향이 수평으로 일관되게 유지되어 ...",
  "materiality": "지지체의 결이 안료 층 사이로 노출되어 ...",
  "color":       "저채도 중간 명도의 색군이 화면 대부분을 차지하며 ..."
}
```
</details>

### 추론 — 유사작 검색

```bash
python search.py \
    --image samples/example.jpg \
    --weights 0.2 0.2 0.2 0.4 \   # L1 L2 L3 L4 가중치 (색채 중심)
    --top-k 10
```

---

## 프로젝트 구조

```
k-art-4l/
├── configs/                 # 실험 설정 (YAML)
│   ├── lqa_base.yaml
│   └── ablation/
├── core/
│   ├── backbone.py          # 공유 비전 백본
│   ├── heads.py             # 층위별 인코딩 헤드 (L1–L4)
│   ├── lqa.py               # Layer-Query Attention
│   ├── decoder.py           # 해설 생성 디코더
│   └── retrieval.py         # FAISS 인덱싱 및 가중 검색
├── data/
│   ├── raw/                 # 원본 (gitignore)
│   ├── processed/           # 전처리 결과 (gitignore)
│   └── goldset/             # 수기 4층위 주석 평가셋
├── scripts/
│   ├── collect_data.py
│   ├── preprocess.py
│   └── build_index.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_color_layer.ipynb
│   └── 03_results.ipynb
├── docs/
│   └── figures/
├── train.py
├── infer.py
├── search.py
└── requirements.txt
```

---

## 한계와 향후 과제

**한계 (솔직하게)**

- 층위별 주석 데이터의 부재로 L1(도상)·L2(기법)는 약지도 신호에 의존한다. 골드셋 규모가 작아 평가의 통계적 신뢰구간이 넓다.
- 학습 데이터가 특정 시기·장르에 편중되어 있어, 전통 회화와 현대 설치·미디어 작업에 대한 일반화는 검증되지 않았다.
- L3(물질성)는 2D 이미지만으로 판단하는 데 원리적 한계가 있다. 표면 질감의 상당 부분이 촬영 조명 조건에 좌우된다.

**향후 과제**

- [ ] 골드셋 확대 및 복수 주석자 간 일치도(IAA) 측정
- [ ] LQA ablation 정밀화 — residual, layer embedding, head 수에 대한 체계적 실험
- [ ] 층위별 근거 시각화 (attention rollout 기반 하이라이트)
- [ ] **K-VALUE 연계** — 4층위 표현을 미술시장 가치평가 모델의 입력 피처로 전이. 작품의 시각적 속성이 가격 형성에 기여하는 경로를 층위 단위로 분해하는 것이 목표
- [ ] 미술관 도슨트 교육 보조 도구로의 실사용 검증

---

## 팀

| 이름 | 역할 |
|------|------|
| 추병곤 | <!-- TODO: 실제 담당 역할 --> 문제 정의, 4-Layer 프레임워크 설계, 도메인 주석 골드셋 구축 |
| <!-- TODO --> | |
| <!-- TODO --> | |

**퍼실리테이터**: <!-- TODO -->
**진행 기간**: <!-- TODO: YYYY.MM ~ YYYY.MM --> · 모두의연구소 AIFFEL 온라인 18기

---

## 참고 문헌

1. Panofsky, E. *Studies in Iconology: Humanistic Themes in the Art of the Renaissance*. Oxford University Press, 1939.
2. Radford, A. et al. "Learning Transferable Visual Models From Natural Language Supervision." *ICML*, 2021.
3. Garcia, N. & Vogiatzis, G. "How to Read Paintings: Semantic Art Understanding with Multi-Modal Retrieval." *ECCV Workshops*, 2018.
4. Johnson, J. et al. "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data*, 2019.
5. <!-- TODO: 실제 참조한 문헌으로 보완 -->

---

## 라이선스

MIT License. 단, 학습에 사용된 소장품 이미지의 저작권은 각 소장 기관 및 작가에게 있으며 본 저장소에 포함되지 않습니다.

---

<p align="center"><i>Learning by Doing — AIFFEL Online 18th</i></p>
