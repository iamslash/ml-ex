# Voyager 스킬 라이브러리 (Skill Library)

> 성공한 코드 스니펫을 저장·검색·재사용하는 Voyager 스타일 스킬 메모리 시스템.

## 실행 방법

```bash
cd agent-ex/8-skill-library
python skill_library.py
```

의존성: Python 표준 라이브러리만 사용 (numpy 불필요)

## 핵심 개념

| 컴포넌트 | 역할 |
|---|---|
| `Skill` | 코드·테스트·태그·성공률을 보유하는 스킬 단위 |
| `SkillLibrary.add_skill` | 새 스킬을 라이브러리에 등록 |
| `SkillLibrary.search_skills` | 설명 유사도(bag-of-words cosine)로 관련 스킬 검색 |
| `SkillLibrary.get_skill` | 이름으로 정확히 스킬 조회 |
| `SkillLibrary.verify_skill` | 테스트 코드를 실행해 `success_rate` 갱신 |
| `SkillLibrary.compose_skills` | 여러 스킬을 하나로 합성 |

### 유사도 검색 알고리즘

```
query → tokenize → TF 벡터
skill description + tags → tokenize → TF 벡터
similarity = cosine(query_vec, skill_vec)
```

외부 라이브러리 없이 순수 Python으로 bag-of-words cosine 유사도 구현.

### 데모 흐름

1. 5개 스킬 추가: `sort_list`, `binary_search`, `parse_csv`, `retry_with_backoff`, `validate_email`
2. `"find element in sorted array"` 검색 → `binary_search` 최상위 결과
3. 모든 스킬 검증 (exec 기반 테스트 실행)
4. `sort_list` + `binary_search` → `sorted_search` 합성
5. 라이브러리 전체 통계 출력

### 핵심 설계 포인트

- `exec` 샌드박스로 스킬 코드를 격리 실행하여 검증
- `usage_count` 추적으로 인기 스킬 파악
- `success_rate` = 누적 검증 통과 횟수 / 전체 실행 횟수

## 참고 논문/자료

- **Voyager** (Wang et al., 2023) - LLM 기반 평생학습 에이전트와 스킬 라이브러리
  - https://arxiv.org/abs/2305.16291
- **ToolFormer** (Schick et al., 2023) - 도구 사용 학습
  - https://arxiv.org/abs/2302.04761
- **CodeAct** (Wang et al., 2024) - 코드 실행을 통한 에이전트 행동
  - https://arxiv.org/abs/2402.01030
