# Inversa

> A **contamination-resistant, self-scaling, machine-verified** benchmark of mathematical ability —
> measured not by *solving* problems but by **constructing** them (given an answer, build a problem
> that yields it). Scored by a sympy oracle with **no LLM judge**.

기존 벤치마크는 "AI가 문제를 **푸는가**"(문제→답)를 잰다. 그런데 공개 시험셋은 학습 데이터에 **유출(오염)** 되어 점수가 부풀려지고, 모델이 강해지면 **포화**된다. Inversa는 **역방향** — "답을 주면 그 답을 낳는 *문제를 구성*할 수 있는가"(답→문제) — 를 잰다. 즉석 무작위 입력으로 문제를 만들게 하므로 **오염될 고정 문항이 없고**, 난이도를 **프로그램으로** 올릴 수 있다. 점수는 **Inversa Generative Score (IGS)** — sympy로 기계 검증, 주관·LLM 심판 없음.

---

## 핵심 결과 (요약 — 전 과정은 [`docs/paper/inversa-paper.ko.md`](docs/paper/inversa-paper.ko.md))

전부 이 저장소의 실제 측정값이며 `data/results/`에 있다.

- **forward는 오염으로 부풀려진다 (우리 데이터):** GSM8K(유출) vs GSM-Symbolic(신선·동일난이도) = **+3.2%**, 95% CI [+1.0, +5.7], 9개 중 8개 모델 양수. 반면 역과제는 **체계적 격차 없음**(+0.0%, [−10.5, +11.0]) — 설계상 오염 불가.
- **IGS는 검증된 능력 측정기다:** 어려운 비포화 forward 벤치(AIME)와 **Spearman +0.93** [+0.65, +1.0] 일치 → 같은 수학 능력을 잰다. *별개 능력은 아니며*(원래 가설 반증), 그 가치는 **오염 면역 + 자동 확장 + gold-standard 일치**라는 구조적 우위.
- **난이도는 스스로 확장된다:** 구성 난이도를 올리면(뫼비우스 → r²·r³ → r⁴·합성) 상단 천장이 깨져 **어떤 모델도 100%에 도달 못 함**(최강 opus-4.8도 88%). 검증 가능 수학 한계 안에서 프런티어까지 변별.
- **거대 리더보드:** OpenRouter 카탈로그(텍스트 346개)에서 추린 102개 시도 → **72개 측정**(19개 패밀리, IGS 0.03~1.00). 그림: [`docs/paper/figures/fig_leaderboard.png`](docs/paper/figures/fig_leaderboard.png).

### 리더보드 상위 (표준 IGS, 발췌)
| # | 모델 | IGS |
|---|---|---|
| 1~5 (1.0 동률, 하드 tiebreak로 분리) | gpt-5.2 · gemini-3.1-flash-lite · deepseek-v4-pro · qwen3.7-plus · grok-build-0.1 | 1.00 |
| 6 | openai/gpt-5.1 | 0.98 |
| 7 | x-ai/grok-4.3 | 0.98 |
| 8 | openai/gpt-oss-120b | 0.97 |
| … | … | … |
| 72 | anthropic/claude-3-haiku | 0.03 |

전체 순위: [`data/results/leaderboard_merged.json`](data/results/leaderboard_merged.json) / 보기: `leaderboard.html`.

---

## 빠른 시작 — 벤치마크 엔진

```bash
# .env 에 OPENROUTER_API_KEY 저장 후
python -m inversa.cli_bench --models "anthropic/claude-opus-4.8,openai/gpt-4o-mini,..." \
       --max-tokens 2048 --max-workers 8
# -> data/results/igs_benchmark.{json,html} : 순위 IGS + 문항별 증거(글라스박스)
```

- `--models` 쉼표 구분. `--max-workers` 동시도. `--deadline` 초과 시 느린 모델 포기(완료분은 증분 저장).
- 점수 = **IGS = mean(pose validity, transform validity)**. 둘 다 sympy 유일근 검증.

### ⚙️ 시스템 자원 (RAM)
대규모 병렬 측정(예: ~100개 모델 동시)은 sympy 검증 워커·HTTP 클라이언트가 메모리를 점유한다. **피크 시 시스템 RAM 약 8–10GB**(가용 최대치 범위 내)를 사용하도록 운영했다. RAM이 부족한 환경에서는 `--max-workers`를 낮추거나(예: 4~6) 모델을 배치로 나눠 실행할 것. (워커를 과도하게 높이면 sympy의 GIL-점유 검증이 동시에 겹쳐 프로세스가 멈출 수 있으므로 6~12 권장.)

---

## 작동 원리 (한눈에)

![pipeline](docs/paper/figures/fig_pipeline.png)

1. **타깃(답)** 을 주고 → 2. 모델에게 **문제 구성**을 시키고 → 3. 모델이 `#### x**3-27=0` 식을 출력 →
4. **추출**(마커/파싱/LaTeX·유니코드 정리) → 5. **sympy 검증**(유일 실근 == 타깃?) → 6. **IGS 채점**.

상세 메커니즘(실제 프롬프트·출력 예·채점 worked example)은 논문 §2 참고.

---

## 한계 (정직)

- N 중간 규모, 일부 동률 → 상관에 부트스트랩 CI를 붙였으나 폭은 넓다.
- 단일 도메인(단변수 대수); 도메인 일반화·외부 예측타당도 미검증(논문 §4.3 E4/E5).
- 리더보드의 미측정 ~30개는 계정 미접근/provider 비호환(non-serverless·404·엔드포인트)·추론모델 latency 때문 — 측정 한계로 표기.
- 일부 초월식 검증은 sympy GIL-행 위험(타임아웃+수치 폴백으로 완화, 고동시도 시 워커 수 주의).

---

## 문서 · 재현성

- **논문:** [`docs/paper/inversa-paper.md`](docs/paper/inversa-paper.md) (영문 정본) · [`docs/paper/inversa-paper.ko.md`](docs/paper/inversa-paper.ko.md) (한국어)
- **핵심 코드:** `tasks/structural.py`(구성 과제) · `verifiers/math_equation.py`(검증 오라클) · `scoring.py`(IGS) · `cli_bench.py`(엔진)
- **실험 스크립트:** `scripts/run_forward_gap.py`(오염), `run_e10.py`(AIME 변별), `run_transform_hard.py`+`gen_transform_*`(난이도 확장), `make_figures.py`(그림)
- **테스트:** 118개 단위테스트(검증기·채점). `data/banks/` 문제은행, `data/results/` 결과.

**참고문헌:** GSM-Symbolic (Mirzadeh et al., Apple, ICLR 2025, arXiv:2410.05229) · GSM1k (Zhang et al., Scale AI, 2024, arXiv:2405.00332) · arXiv:2311.04850.
