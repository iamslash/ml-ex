# Tool Sandbox (도구 샌드박스)

> LLM이 호출할 수 있는 도구를 제한하고, 위험한 명령과 경로 접근을 차단.

## 실행 방법

```bash
cd harness-ex/3-tool-sandbox
python tool_sandbox.py
```

## 핵심 개념

### 도구 권한 제어

```
에이전트 → 도구 호출 요청
              ↓
    ToolRegistry: allowlist 확인
              ↓ 허용
    FileAccessController: 경로 확인
              ↓ 허용
    CommandFilter: 위험 명령 확인
              ↓ 안전
         실행 → 결과 반환
```

### 3중 방어

| 계층 | 역할 | 예시 |
|------|------|------|
| ToolRegistry | 도구 수준 허용/차단 | delete_file = blocked (high risk) |
| FileAccessController | 경로 수준 접근 제한 | `../../etc/passwd` = blocked |
| CommandFilter | 명령 수준 위험 패턴 | `rm -rf`, `eval(`, `sudo` = blocked |

### 감사 로그

모든 도구 호출 시도를 기록: 성공/차단/에러, 시각, 도구명, 인자.

## 출처 및 참고 자료

- OpenAI Agents SDK: tool permissions, sandboxing
- Docker sandbox 패턴: 파일/네트워크 격리
