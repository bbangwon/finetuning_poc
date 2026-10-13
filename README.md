# 🚀 Gemma 4 LoRA Fine-Tuning PoC (Apple MLX)

Apple Silicon(M4 Pro) 환경에서 **MLX 프레임워크**를 활용하여 경량 언어 모델(**Gemma 4 E2B**)을 LoRA(Low-Rank Adaptation) 방식으로 파인튜닝하고 성능을 검증한 개념 증명(PoC) 프로젝트입니다.

---

## 📌 프로젝트 개요

- **대상 모델**: `Google Gemma 4 E2B (4-bit MLX)`
- **파인튜닝 기법**: LoRA (Low-Rank Adaptation, Rank=8, Alpha=16)
- **학습 환경**: Apple M4 Pro (12-Core, 24GB Unified Memory, Metal GPU 가속)
- **학습 파라미터 비율**: 약 0.147%
- **어댑터 용량**: **26.03 MB** (`adapters/adapters.safetensors`, 초경량 탈부착 가능)
- **과제 도메인**: 가상 테크 기업 '넥스트웨이브 코퍼레이션(NextWave Corp.)' 사내 복무 규정 Q&A

---

## 📊 정량적 성과 요약

| 구분 | Base 모델 (어댑터 OFF) | LoRA Fine-tuned (어댑터 ON) | 개선도 |
| :--- | :---: | :---: | :---: |
| **사내 규정 준수 정확도** | **30.0%** (3/10) | **70.0%** (7/10) | **+40.0%p** |
| **어댑터 파일 용량** | - | **26.03 MB** (초경량) | 독립 서빙 / 탈부착 가능 |
| **학습 소요 시간** | - | **1분 27초** (120 Iters) | M4 Pro 로컬 완결 |

> 상세 평가 결과 및 한글 질의응답 비교표는 [results/finetuning_comparison_report.md](results/finetuning_comparison_report.md) 및 [docs/slm_lora_learning_guide.md](docs/slm_lora_learning_guide.md)를 참고하세요.

---

## 📁 디렉토리 구조

```text
finetuning_poc/
├── configs/
│   └── lora_config.yaml         # MLX LoRA 학습 하이퍼파라미터 설정
├── data/
│   ├── manual.md                # 사내 규정 원본 매뉴얼
│   ├── generate_dataset.py      # Q&A 합성 데이터셋 생성 스크립트
│   ├── train.jsonl              # 학습 데이터셋 (16건)
│   ├── valid.jsonl              # 검증 데이터셋 (4건)
│   ├── test.jsonl               # 테스트 데이터셋 (4건)
│   └── test_questions.jsonl     # 성능 비교 평가용 10개 질문
├── adapters/                    # LoRA 어댑터 가중치 및 설정 파일
│   ├── adapter_config.json
│   └── adapters.safetensors
├── src/
│   ├── train.py                 # MLX LoRA 파인튜닝 실행 스크립트
│   └── compare.py               # 어댑터 ON/OFF 성능 1:1 비교 스크립트
├── results/
│   └── finetuning_comparison_report.md  # 최종 비교 분석 보고서
├── docs/
│   └── slm_lora_learning_guide.md       # SLM LoRA 파인튜닝 학습 가이드
└── pyproject.toml
```

---

## 🛠️ 실행 방법

### 1. 환경 설정

```bash
# 가상환경 활성화 및 의존성 설치
python -m venv .venv
source .venv/bin/activate
pip install mlx-lm rich pyyaml
```

### 2. 데이터셋 생성

```bash
python data/generate_dataset.py
```

### 3. LoRA 파인튜닝 학습

```bash
python src/train.py
```

### 4. 어댑터 ON/OFF 성능 비교 평가

```bash
python src/compare.py
```
