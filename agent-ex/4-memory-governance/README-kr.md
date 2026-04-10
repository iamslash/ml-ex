# 메모리 거버넌스 (Memory Governance)

> 안전한 에이전트 메모리 관리: PII 마스킹, TTL 기반 만료, 충돌 감지, 안전 게이팅

## 실행 방법

```bash
cd agent-ex/4-memory-governance
python memory_governance.py
```

의존성: Python 표준 라이브러리만 사용 (외부 패키지 불필요)

## 핵심 개념

| 컴포넌트 | 역할 |
|---|---|
| `PIIMasker` | 이메일, 전화번호, IP, API 키를 정규식으로 탐지·마스킹 |
| `MemoryAger` | TTL 기반 만료 + 중요도 가중치로 수명 연장 |
| `ConflictDetector` | 키워드 겹침 + 부정어 패턴으로 모순 교훈 탐지 |
| `SafetyGate` | 프롬프트 인젝션 방어용 면책 문구 자동 삽입 |
| `GovernedMemoryStore` | 위 4가지를 파이프라인으로 통합한 메모리 저장소 |

### 파이프라인 흐름

```
저장: 입력 → PII 마스킹 → 충돌 검사 → 저장소
조회: 저장소 → 만료 필터 → 충돌 주석 → 안전 게이트 → 출력
```

### 중요도 가중 TTL

```
effective_ttl = base_ttl × importance
```

중요한 기억(importance=5.0)은 동일한 기본 TTL에서 5배 오래 유지됩니다.

## 참고 논문/자료

- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) - 계층적 메모리 관리
- [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) - 에이전트 안전성 평가
- [Prompt Injection Attacks and Defenses](https://arxiv.org/abs/2306.05499) - 안전 게이팅 패턴
- [GDPR 가이드라인](https://gdpr.eu/) - PII 처리 규정
