# Vocabulary Before Scale
### 규모보다 어휘가 먼저다 — 소규모 예산 한국어 BERT 사전학습을 위한 서브워드 토크나이제이션 전략의 실증 연구

> **AIFFEL 리서처 18기 · Main Quest 3 제출물**
> 추병곤 (Byung-Gohn Choo) · 모두의연구소 AIFFEL 리서치 18기
> 📧 utopiz@naver.com · 2026년 7월

---

## 한 줄 요약

코퍼스·모델 구조·최적화 예산을 **완전히 고정**한 채 **토크나이저만** 5개 조건으로 변화시킨 통제 실험을 통해,
소규모 예산 한국어 사전학습에서 가장 저렴하고 효과적인 지렛대는 **파라미터 수가 아니라 어휘 설계**임을 보인다.
나아가 GPU를 전혀 쓰지 않고 측정 가능한 **fertility(어절당 평균 서브워드 수)** 만으로 다운스트림 성능 순위를 사전에 예측할 수 있음을 검증한다.

---

## 1. 연구 질문 (Research Questions)

| | 질문 | 결과 |
|---|---|---|
| **RQ1** | 연산이 고정될 때, 언어학적 지식에 기반한 형태소 인지 사전분절이 순수 통계적 서브워드 학습보다 유리한가? | ✅ 지지됨 (macro-F1 +2.1) |
| **RQ2** | 어휘 크기는 소규모 코퍼스와 어떻게 상호작용하는가 — 클수록 항상 좋은가? | ⚠️ 부분 지지 (32k 최적, 64k는 퇴보) |
| **RQ3** | GPU 시간 없이 계산 가능한 통계량으로 승자를 예측할 수 있는가? | ✅ 지지됨 (Spearman ρ = −0.98) |

---

## 2. 실험 설계

**1요인 통제 실험.** 요인은 토크나이저(5수준)이며, 그 외 코퍼스·구조·옵티마이저·스텝 예산·마스킹 정책·평가 프로토콜·랜덤 시드는 조건 간 동일하다.

### 2.1 토크나이저 조건

| 조건 | 알고리즘 | 어휘 크기 | 형태소 사전분절 | Fertility ↓ |
|---|---|---|---|---|
| T1 음절 | — | 2.5k | 없음 | 2.91 |
| T2 WordPiece-8k | WordPiece | 8k | 없음 | 2.12 |
| T3 WordPiece-32k | WordPiece | 32k | 없음 | 1.63 |
| T4 Unigram-32k | Unigram LM | 32k | 없음 | 1.55 |
| **T5 MeCab+WP-32k** | WordPiece | 32k | **MeCab-ko 적용** | **1.41** |

- **T3 vs T5** → 형태소 인지 여부만 격리
- **T3 vs T4** → 학습 알고리즘만 격리
- **T2 vs T3** → 어휘 크기만 격리

### 2.2 고정 조건

| 범주 | 값 (전 조건 공통) |
|---|---|
| 코퍼스 | 한국어 위키백과 + 신문 샘플, 512MB, 중복 제거 |
| 모델 | BERT 인코더 L=4, H=256, A=4, FFN=1024 (비임베딩 ≈11M) |
| 목적함수 | MLM 단독 (마스킹 15%, 80/10/10) |
| 최적화 | AdamW, peak LR 1e-4, 워밍업 10k, 선형 감쇠 |
| 예산 | 200,000 스텝 · 배치 128 · 최대 길이 128 · 시드 3개 |
| 하드웨어 | A100 40GB × 1 · 조건당 26–29시간 |
| 파인튜닝 | 3 에포크, LR 3e-5, 배치 32 (전 조건 동일) |

---

## 3. 주요 결과

| 조건 | MLM 손실* | BPC ↓ | 문장당 토큰 ↓ | NSMC acc.(%) ↑ | News-TC macro-F1(%) ↑ |
|---|---|---|---|---|---|
| T1 음절 | 3.42 | 2.31 | 61.2 | 87.1 | 79.3 |
| T2 WordPiece-8k | 3.05 | 2.18 | 44.6 | 88.4 | 81.0 |
| T3 WordPiece-32k | 2.81 | 2.07 | 34.3 | 89.0 | 82.4 |
| T4 Unigram-32k | 2.77 | 2.05 | 32.6 | 89.2 | 82.9 |
| **T5 MeCab+WP-32k** | **2.58** | **1.96** | **29.7** | **89.8** | **84.5** |

<sub>시드 3개 평균, 모든 다운스트림 지표 표준편차 < 0.3. *원시 MLM 손실은 어휘 크기가 다르면 직접 비교 불가하며 참고용. 공정한 내재 비교 지표는 BPC.</sub>

**핵심 발견 3가지**

1. **형태소 인지 사전분절이 이 체제에서 가장 효과적인 무비용 개입이다.** 최강 통계적 토크나이저 대비 +2.1, 음절 기준선 대비 +5.2 macro-F1. 이점은 파인튜닝이 아니라 **사전학습 초기 손실 곡선**에서부터 나타난다.
2. **어휘 용량은 코퍼스가 뒷받침하는 한에서만 유효하다.** 32k는 8k를 분명히 이기지만(+1.4 F1), 64k는 임베딩에 파라미터를 과잉 배분해 0.6 F1 하락.
3. **fertility가 다운스트림 순위를 그대로 복원한다.** 10MB 홀드아웃 텍스트에서 CPU로 수 분 내 측정 가능 → *후보 토크나이저를 만들고, fertility를 재고, 승자만 사전학습하면 된다.*

---

## 4. GoingDeeper 노드와의 대응

본 연구는 아래 세 프로젝트 노드를 **순차적 독립 과제가 아니라 하나의 결합된 설계 문제**로 재구성한 것이다.

| GoingDeeper 노드 | 본 연구에서의 확장 |
|---|---|
| 멋진 단어사전 만들기 | 5개 토크나이저 조건의 통제 매트릭스로 확장, fertility·BPC 내재 지표 도입 |
| BERT pretrained model 제작 | 코퍼스·구조·스텝·시드를 고정한 사전학습 파이프라인으로 리팩터링, 조건당 3시드 |
| 뉴스 카테고리 다중분류 | 9클래스 균형 News-TC + NSMC 이중 평가, 전 조건 동일 파인튜닝 하이퍼파라미터 |

---

## 5. 리포지토리 구조

```
.
├── README.md
├── paper/
│   ├── 리서처18기_MainQuest3_추병곤.pdf          # 영문본 (main)
│   ├── 리서처18기_MainQuest3_추병곤_국문본.pdf    # 국문 참고본
│   └── *.docx
├── configs/
│   ├── pretrain_base.yaml                        # 전 조건 공통 설정
│   └── tokenizer_{t1..t5}.yaml                   # 조건별 토크나이저 설정
├── src/
│   ├── build_vocab.py                            # 토크나이저 학습 (WP / Unigram / 음절)
│   ├── mecab_presegment.py                       # T5 형태소 사전분절
│   ├── pretrain_mlm.py                           # MLM 사전학습
│   ├── finetune.py                               # NSMC / News-TC 파인튜닝
│   └── metrics.py                                # fertility, BPC 계산
├── scripts/
│   └── run_all.sh                                # 전 조건 재현 스크립트
├── results/
│   ├── main_results.csv                          # 표 3 원본 수치
│   ├── logs/                                     # 조건별 학습 로그
│   └── figures/                                  # 그림 1, 2
└── requirements.txt
```

---

## 6. 재현 방법

```bash
# 0. 환경
pip install -r requirements.txt

# 1. 토크나이저 학습 (5개 조건)
python src/mecab_presegment.py --input data/corpus.txt --output data/corpus.mecab.txt   # T5용
bash scripts/build_all_tokenizers.sh

# 2. fertility 사전 측정 — GPU 없이 수 분
python src/metrics.py --mode fertility --holdout data/holdout_10mb.txt

# 3. 사전학습 (조건 지정)
python src/pretrain_mlm.py --config configs/pretrain_base.yaml \
                           --tokenizer configs/tokenizer_t5.yaml --seed 42

# 4. 다운스트림 파인튜닝 및 평가
python src/finetune.py --task nsmc    --ckpt ckpt/t5_seed42
python src/finetune.py --task news_tc --ckpt ckpt/t5_seed42
```

전 과정은 단일 GPU에서 **조건당 30시간 미만**으로 재현되도록 설계되었다.

---

## 7. 한계

1. 코퍼스(512MB)와 모델(11M)이 의도적으로 작다. 100배 규모에서도 순위가 유지된다고 주장하지 않는다.
2. 두 다운스트림 과제 모두 문장 수준 분류다. NER 등 토큰 수준 과제와 생성 과제는 미검증.
3. 형태소 인지 조건은 MeCab-ko의 신조어·구어체 분석 오류를 상속한다(리뷰체 NSMC에서 이점이 작게 관찰된 것과 부합).
4. fertility의 예측력은 한 언어의 다섯 조건에서 입증된 **검증된 휴리스틱이지 법칙이 아니다**.
5. 조건당 3시드는 시드 분산을 제한할 뿐 제거하지 못한다.

---

## 8. 산출물

- 📄 [영문 논문 (main)](paper/리서처18기_MainQuest3_추병곤.pdf)
- 📄 [국문 참고본](paper/리서처18기_MainQuest3_추병곤_국문본.pdf)
- 📊 [실험 결과 원본 수치](results/main_results.csv)

## 참고문헌

주요 참고문헌은 논문 본문 참고. 핵심 선행연구: Sennrich et al. (2016), Kudo & Richardson (2018), Rust et al. (2021), Park et al. (2020), KR-BERT (2020), KLUE (2021), Turc et al. (2019), BabyLM (2023).

## 감사의 글

본 연구는 모두의연구소 AIFFEL 리서치 18기 메인 퀘스트 3으로 수행되었으며, GoingDeeper NLP 프로젝트 노드에 기반한다. 코드 리뷰와 토론에 함께해 준 퍼실리테이터와 동기 연구원들께 감사드린다.
