# 툴 샌드박스 (Tool Sandbox)

> 도구 권한 제어, 경로 순회 차단, 위험 명령어 필터링으로 LLM 에이전트의 도구 실행을 안전하게 제한하는 모듈

---

## 실행 방법

```bash
cd harness-ex/3-tool-sandbox
python tool_sandbox.py
```

의존성 없음 — Python 3.9+ 표준 라이브러리 (`fnmatch`, `os`, `dataclasses`, `datetime`)만 사용합니다.

---

## 핵심 개념

### 전체 아키텍처

```
LLM 에이전트
    │
    │  tool_name + args
    ▼
┌─────────────────────────────────────┐
│           ToolRegistry               │
│                                     │
│  ┌─────────────┐  ┌──────────────┐  │
│  │  allowlist  │  │  blocklist   │  │
│  │  read_file  │  │ delete_file  │  │
│  │  write_file │  │              │  │
│  │  search_web │  │              │  │
│  └──────┬──────┘  └──────┬───────┘  │
│         │                │          │
│    허용된 경우        차단된 경우      │
│         │                │          │
│         ▼                ▼          │
│    tool.fn()       PermissionError  │
│         │                           │
│  AuditLog에 기록   AuditLog에 기록   │
└─────────────────────────────────────┘
    │
    ├── FileAccessController (경로 검증)
    │     allowed_patterns: ["./workspace/*", "/tmp/sandbox/*"]
    │
    └── CommandFilter (명령어 필터)
          blocked: ["rm -rf", "curl", "wget", "eval(", ...]
```

### Tool 데이터 구조

```
Tool
├── name         : str          ("read_file")
├── description  : str          ("Read a file from the workspace")
├── fn           : Callable     (실제 실행 함수)
├── risk_level   : str          ("low" | "medium" | "high")
└── allowed      : bool         (True = allowlist, False = blocklist)
```

### 등록된 도구 목록

```
도구명          위험도    기본 허용
────────────────────────────────
read_file       low       YES
write_file      medium    YES
execute_code    high      YES  ← CommandFilter 추가 검사
search_web      medium    YES
delete_file     high      NO   ← blocklist (항상 차단)
```

### FileAccessController — 경로 순회 차단

```
allowed_patterns = ["./workspace/*", "/tmp/sandbox/*"]

요청 경로                           결과
──────────────────────────────────────────────────────
./workspace/notes.txt              [OK]    허용 패턴 내
./workspace/../../etc/passwd       [DENY]  순회 공격 감지
/tmp/sandbox/output.json           [OK]    허용 패턴 내
/tmp/sandbox/../../../root/.ssh    [DENY]  순회 공격 감지

동작 원리:
  os.path.realpath(path) → 절대 경로 해석
  → allowed_patterns 각각과 비교
  → 패턴 밖이면 PermissionError
```

### CommandFilter — 위험 명령어 목록

```
차단 패턴:
  "rm -rf"        삭제 공격
  "curl"          외부 데이터 송수신
  "wget"          외부 파일 다운로드
  "sudo"          권한 상승
  "chmod 777"     과도한 권한 부여
  "eval("         동적 코드 실행
  "__import__"    동적 임포트
  "os.system"     쉘 명령 실행
  "subprocess"    서브프로세스 생성
  "exec("         코드 실행

입력: "import os; os.system('rm -rf /')"
       ────────────────────────────────
       [DENIED] Blocked pattern detected: 'os.system'
```

### 감사 로그 (Audit Log) 형식

```
[OK]   2025-01-15T10:23:01Z | read_file             | Result: [SIMULATED CONTENT ...]
[DENY] 2025-01-15T10:23:01Z | delete_file           | Tool blocked (risk_level=high)
[DENY] 2025-01-15T10:23:01Z | read_file             | Path not in allowed directories
[ERR]  2025-01-15T10:23:01Z | execute_code          | Dangerous code blocked: ...
```

### 런타임 정책 변경

```
reg.allow("tool_name")  →  blocklist에서 제거, allowlist에 추가
reg.block("tool_name")  →  allowlist에서 제거, blocklist에 추가

보안 사고 대응 흐름:
  1. 사고 감지
  2. reg.block("execute_code")  ← 즉시 차단
  3. 조사 완료
  4. reg.allow("execute_code")  ← 복구
```

---

## 실패 트레이스 (Deterministic Failure)

**DEMO 4** — 런타임 정책 변경으로 인한 차단:

```
[STEP 1] execute_code is currently ALLOWED
  tool='execute_code'  args={'code': 'result = 1 + 1'}
  [OK] [SIMULATED EXEC: result = 1 + 1]

[STEP 2] Security incident detected — blocking execute_code
  reg.block("execute_code")

[STEP 3] Same call now BLOCKED
  tool='execute_code'  args={'code': 'result = 1 + 1'}
  [DENIED] Tool 'execute_code' is blocked [risk=high].

[TRACE] ToolRegistry.execute() → PermissionError
        Cause: 'execute_code' moved to blocklist via reg.block().
        Fix: re-enable after incident review with reg.allow().
```

**DEMO 2** — 경로 순회 공격 차단:

```
tool='read_file'  path='./workspace/../../etc/passwd'
  os.path.realpath() → '/etc/passwd'
  allowed_patterns 비교 → 패턴 불일치
  [DENIED] Path not in allowed directories: './workspace/../../etc/passwd'
```

---

## 출처 및 참고 자료

| 개념 | 출처 |
|------|------|
| LLM 에이전트 도구 권한 제어 | [Anthropic — Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) |
| 최소 권한 원칙 (Least Privilege) | [OWASP Access Control Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Access_Control_Cheat_Sheet.html) |
| 경로 순회 공격 | [OWASP Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal) |
| LLM 에이전트 샌드박스 설계 | [Anthropic — Computer Use Safety](https://docs.anthropic.com/en/docs/build-with-claude/computer-use) |
| 감사 로그 패턴 | [NIST SP 800-92 — Guide to Log Management](https://csrc.nist.gov/publications/detail/sp/800-92/final) |

> 이 모듈의 `_read_file()`, `_write_file()`, `_execute_code()` 함수는
> 실제 파일시스템이나 코드 실행 없이 시뮬레이션 결과를 반환합니다.
> `FileAccessController`의 경로 검증 로직은 `os.path.realpath()`를 사용하여
> 실제 경로 순회 공격을 탐지합니다.
