# Inversa

> *문제를 **푸는** 능력이 아니라, 답을 받고 그 답을 낳는 **문제를 짓는** 능력으로 AI의 추론을 측정하는 심리측정 벤치마크.*

기존 벤치마크는 "AI가 문제를 푸는가"(≈ 본 적 있는가, 암기·coverage)를 측정한다. Inversa는 **inverse/생성** 방향 — "답을 주면 그 답을 낳는 유효한 문제를, 통제된 난이도·제약 하에 지을 수 있는가"(≈ 구조를 이해하는가) — 를 측정한다. 채점은 **기계 검증 가능한 형식 도메인(수학·코드·논리)** 에 못박아 주관적 오라클 함정을 피하고, **IRT**로 보정해 *생성-추론 능력 θ_gen* 을 추정한다.

**핵심 검정 질문:** θ_gen 이 풀이 능력 θ_solve 와 **괴리**하는가? 괴리한다면 기존 벤치마크가 놓치는 실제 추론 차원이 존재함을 측정으로 증명한다.

## 문서
- 헌장 + 설계 (최상위 근거): [`docs/specs/2026-06-03-inversa-design.md`](docs/specs/2026-06-03-inversa-design.md)

## 상태 (2026-06-03)
slice 1 (validity) · 2 (난이도 calibration) · 2b (model-vs-model adversarial) · 3 (괴리 실험 + HTML 리포트) **구현 완료. 유닛 75개 통과, main/GitHub 반영.**

- **측정 도구:** 작동 · 객관(sympy 기계검증) · 투명(전 과정 공개). 모델 *변별* 실측 — 어려운 라디칼 방정식에서 haiku 75% vs opus/sonnet 100%.
- **핵심 가설 (θ_gen ≠ θ_solve):** *미결.* N=3 동일계열(Claude) 강모델로는 검정력 부족(Spearman +0.5, *suggestive·결정 불가*). 결정적 증명엔 더 많은 모델 계열 · 더 어려운/넓은 도메인 · 더 큰 세트 필요.
- 즉 — *작동하고 모델을 가르는 도구는 증명됐고, 헤드라인 가설은 검증 가능한 채 열려 있음.*

### 실행 / 결과 보기
```
.venv\Scripts\python.exe -m inversa.cli_experiment --solve-bank data\banks\solve_set_hard.json --targets 3,7,12 --levels 1,2,3,4,5
```
결과: `data/results/*.html` (브라우저로 열어 표·막대·괴리 판정·증거 확인) + `*.json`. (`.env`의 `ANTHROPIC_API_KEY` 자동 사용.)

## 무엇이 *아닌가* (anti-drift)
개방·주관 도메인 품질 평가 / AI "성격" 측정 / solve-벤치마크 데이터 생성 도구 / 모델 학습·파인튜닝 / 범용 평가 플랫폼 — 전부 명시적 비목표. (근거: 설계 문서 §8.)
