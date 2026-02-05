# ADR-001: 검색 에이전트 모델 선택

## Status

Accepted

## Context

검색 에이전트에 사용할 LLM 모델을 선택해야 한다. 후보 모델:

- Qwen 4B
- Jan-v3-4B
- LFM2.5-1.2B-Instruct

모델 선택 기준:

1. Tool calling 기능 지원 여부
2. 응답 속도
3. 결과 품질

## Decision

**Jan-v3-4B** 모델을 선택한다.

### 선택 근거

1. **LFM2.5 탈락**: Tool Web search를 호출하지 않아 검색 에이전트로 사용 불가
2. **속도 비교**: Jan-v3-4B(33.4s)가 Qwen 4B(44.9s) 대비 약 25% 빠름
3. **품질**: 두 모델의 결과 품질은 비슷함

## Consequences

### 긍정적

- 더 빠른 검색 응답 시간
- Tool calling 기능 정상 동작

### 부정적

- Jan-v3-4B의 표준편차(6.655s)가 Qwen 4B(4.804s)보다 큼
- 응답 시간의 일관성이 다소 낮음

### 참고

- 상세 벤치마크 결과: [benchmark-model-comparison.md](./benchmark-model-comparison.md)
