# LLM 에이전트 패턴 (Agent Patterns)

> 실수를 학습하고 스스로 교정하는 LLM 에이전트 아키텍처.

## 학습 순서

| # | 모듈 | 핵심 패턴 | 디렉토리 |
|---|---|---|---|
| 1 | 메모리 스키마 | SQLite 기반 에피소딕·시맨틱·절차적 메모리 | `1-memory-schema/` |
| 2 | Reflexion | 실패 후 언어적 자기 반성 루프 | `2-reflexion/` |
| 3 | 메모리 검색 | 최신성·관련성·중요도 3축 하이브리드 검색 | `3-memory-retrieval/` |
| 4 | 메모리 거버넌스 | PII 마스킹, 안전 게이팅, 접근 제어 | `4-memory-governance/` |
| 5 | 생성+반성 | 생성 후 자기 평가 및 개선 | `5-generation-reflection/` |
| 6 | 반복 정제 | 품질 기준 달성까지 반복 개선 | `6-iterative-refinement/` |
| 7 | LLM-TDD | 테스트 먼저 작성 후 코드 생성·반복 | `7-llm-tdd/` |
| 8 | 스킬 라이브러리 | 성공 코드 저장·검색·재사용 (Voyager 스타일) | `8-skill-library/` |
| 9 | 자기 교정 에이전트 | 모든 모듈 통합 캡스톤 에이전트 | `9-self-correcting-agent/` |

## 빠른 실행

```bash
# 모듈별 실행
python agent-ex/7-llm-tdd/llm_tdd.py
python agent-ex/8-skill-library/skill_library.py
python agent-ex/9-self-correcting-agent/agent.py
```

## 참고 논문

| 논문 | 핵심 기여 | 링크 |
|---|---|---|
| **Reflexion** (Shinn et al., 2023) | 언어적 강화학습 기반 에이전트 자기 반성 | https://arxiv.org/abs/2303.11366 |
| **Generative Agents** (Park et al., 2023) | 에피소딕 메모리를 갖춘 시뮬레이션 에이전트 | https://arxiv.org/abs/2304.03442 |
| **ReAct** (Yao et al., 2022) | 추론(Reasoning)과 행동(Acting)의 통합 | https://arxiv.org/abs/2210.03629 |
| **MemGPT** (Packer et al., 2023) | 계층적 메모리 관리 시스템 | https://arxiv.org/abs/2310.08560 |
| **Voyager** (Wang et al., 2023) | 스킬 라이브러리 기반 평생학습 에이전트 | https://arxiv.org/abs/2305.16291 |
| **LATS** (Zhou et al., 2023) | 언어 에이전트 트리 탐색(Monte Carlo) | https://arxiv.org/abs/2310.04406 |
