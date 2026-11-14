# Stage 2: email sending (no LLM, no agent)

Status: draft, 2026-11-14. Covers the delivery part of acceptance criteria 4–6 from
`CLAUDE.md` (an email lands in MailHog, addressed to a department, with the right
`Reply-To`). Branch: `feat/stage-2-email-sending`.

Measurements quoted below were taken on the dev machine during the spec investigation,
against `mailhog/mailhog:v1.0.1` on the compose network, with Python 3.14.0,
aiosmtplib 5.1.3, email-validator 2.3.0 and langchain-core 1.8.0 in throwaway
environments (nothing was added to the project).

## Goal

An async function that sends the original user message to one of the five allowed
department addresses, with `Reply-To` set to the requester's address, and that stage 3
can wrap as the agent's email tool without changes:

1. invalid input (unknown department, malformed or multi-address requester email, empty
   message) is rejected with `ValueError` **before** any SMTP I/O;
2. every delivery failure (DNS, connection refused, timeout, refused sender, recipient
   or data) surfaces as one dedicated exception, `EmailDeliveryError`;
3. no header can be added or altered through the requester email or the message;
4. a real email sent from the host reaches MailHog, and an e2e test proves To,
   Reply-To, Subject and body via MailHog's HTTP API, locally and in CI.

## Non-goals

- The LLM, the agent, its prompt and the tool wrapper (stage 3).
- The HTTP message endpoint, its request model and HTTP error mapping (stage 3).
- Choosing a department from message content.
- `README.md`.
- Retries, queues or an outbox; TLS or authentication for SMTP (MailHog has neither).
- HTML email, attachments, Cc/Bcc.
- Logging inside the mailer. Stage 3 logs at the call site, where it knows the request.

## Design

### Decisions and evidence

**D1. `Department` is a `StrEnum` whose values are the addresses.**

- Both `StrEnum` and `Literal[...]` produce the same inline tool schema in
  langchain-core 1.8.0 (`{"enum": [five addresses], "type": "string"}`, no `$ref`), and
  in both cases LangChain's own validation rejects an unknown value before the tool body
  runs. Verified. So the stage 3 schema the model sees doesn't depend on this choice.
- `StrEnum` wins on ergonomics: `Department(value)` validates at runtime and raises
  `ValueError` (a `Literal` needs a manual `in get_args(...)` check),
  `list(Department)` feeds the stage 3 prompt, members are named (`Department.IT`),
  and `str(member)` is the address itself.

**D2. `aiosmtplib`, not `smtplib` in `asyncio.to_thread`.**

Both were tested against MailHog, an unresolvable host, a closed port and a
non-routable IP (timeout 2 s):

| Failure | aiosmtplib | smtplib |
|---|---|---|
| DNS failure | `SMTPConnectError` | `socket.gaierror` (an `OSError`) |
| Connection refused | `SMTPConnectError` | `ConnectionRefusedError` (an `OSError`) |
| Timeout (2.0 s) | `SMTPConnectTimeoutError` | `TimeoutError` (an `OSError`) |

- With aiosmtplib, every failure is an `aiosmtplib.SMTPException`, so one `except`
  maps them all. smtplib needs `except (OSError, smtplib.SMTPException)`.
- Cancellation: if stage 3 cancels the request (client gone, agent timeout), aiosmtplib
  aborts. A `to_thread` call can't be cancelled, and the thread may still send.
- MailHog's EHLO advertises only `PIPELINING` and `AUTH PLAIN`, with no `8BITMIME`.
  smtplib sends a UTF-8 body as raw 8-bit anyway, which breaks the RFC. aiosmtplib
  re-encodes it as base64. MailHog's UI decodes both base64 and quoted-printable
  (checked in `js/controllers.js`), so Polish text is readable either way.
- The cost is one pure-Python dependency with no runtime dependencies of its own.
- ruff's ASYNC rules don't flag `smtplib`, so lint doesn't decide this; event-loop
  blocking and error mapping do.

**D3. The requester email is validated with `email-validator`.**

`EmailMessage` (default policy) raises `ValueError` for CR/LF in any header value, so
newline injection is already blocked. Verified on 3.14. It still accepts
`Reply-To: a@b.com, evil@x.com` as **two** addresses, and it serializes a non-ASCII
local part as an RFC 2047 encoded-word, which is invalid in an addr-spec. So:

```python
validate_email(value, check_deliverability=False, allow_smtputf8=False).ascii_email
```

Verified behaviour:

| Input | Result |
|---|---|
| `jan.nowak@example.com` | accepted |
| `Jan.Nowak@EXAMPLE.com` | `Jan.Nowak@example.com` (domain lowercased) |
| `a@b.com, evil@x.com` | rejected |
| `a@b.com\nBcc: x@y.com` | rejected |
| `Jan <jan@example.com>` | rejected (display name) |
| `żółw@example.com` | rejected (non-ASCII local part; MailHog has no SMTPUTF8) |
| `jan@bücher.de` | accepted; `.ascii_email` gives the punycode domain |
| `jan@localhost`, `jan@x.test`, `jan@x.local` | rejected |
| `=?utf-8?q?a=40evil.com=2C_b?=@example.com` | accepted by email-validator; rejected by our `=?` check |

- RFC 2047 encoded words (`=?…?=`) are a legal local part, but `EmailMessage` decodes
  them when the header is set, turning one validated address into another address, a
  list or a display name (found in review, reproduced). So `normalize_requester_email`
  also rejects `=?`, and `build_message` checks that `Reply-To` parses back to exactly
  the validated address.

- `EmailNotValidError` subclasses `ValueError`. `check_deliverability=False` means no
  DNS lookups, so it works offline.
- Pydantic's `EmailStr` is **not** equivalent: it accepts `Jan <jan@example.com>`
  (silently strips the name) and `żółw@example.com`. Stage 3's request model must reuse
  `normalize_requester_email` (see open question 2).
- The cost is `email-validator` plus `dnspython` and `idna`.

**D4. Code builds the subject; the body is the original message, unchanged.**

- Subject: `[message-router] Message from <normalized requester email>`. No caller text
  reaches any header, and the requester part is already validated.
- Body: exactly the `message` argument, `text/plain; charset=utf-8`, with no footer.
- In stage 3 the model supplies only `department`. The requester email and the message
  are bound from the HTTP request. A single tool argument is the most reliable shape for
  a 0.6B model, and the model can't touch `Reply-To` or the body.

**D5. Publish MailHog SMTP on `127.0.0.1:1025`.** This reverses part of stage 1 D4.

- The e2e test runs on the host and must reach SMTP. Port 1025 is free on the dev
  machine (checked with `ss -ltn`). Loopback only, for the same reason as 8000 and 8025:
  Docker port rules bypass ufw.
- This doesn't conflict with the "container wiring" rule. That rule governs app
  settings: `SMTP_HOST: mailhog` is the only new compose env value. `ports:` is compose's
  own config, not a setting. `SMTP_PORT` is not set in compose, because 1025 is both the
  pydantic default and MailHog's port inside the network.
- Side benefit: a host run (`uv run uvicorn …`) now sends to the compose MailHog with
  the default settings.

**D6. Timeout 10 s, no retries.**

- The `smtp_timeout` setting defaults to 10 s. aiosmtplib applies it to connect and to
  each SMTP command, so a server that stalls mid-session could take a few times that. A
  total cap, if needed, belongs to stage 3's request budget (open question 4). MailHog
  answers in milliseconds (measured: ~10 ms per send).
- No retries: sending is not idempotent. If the server accepted DATA but the reply was
  lost, a retry duplicates the email. MailHog is local, so a failure means it is down,
  and an immediate retry won't help.

**D7. The envelope is explicit: `sender=settings.smtp_from, recipients=[department]`.**

Without explicit recipients, aiosmtplib derives them from the To/Cc/Bcc headers.
Passing them explicitly guarantees the requester (Reply-To) is never an envelope
recipient.

**D8. Errors: `ValueError` for bad input, `EmailDeliveryError` for delivery.**

- Bad input is a caller or model error. Stage 3 can return it to the model as a tool
  error ("choose one of …") or as a 422.
- `EmailDeliveryError` (raised `from` the aiosmtplib exception) is an infrastructure
  failure that stage 3 maps to 502/503. Its message names the department and the
  aiosmtplib error class and text, and never includes the body or the requester email.

**D9. CI runs the email e2e test against a MailHog-only stack.**

`docker compose up -d --wait mailhog` starts just MailHog: no model, no API build.
`--wait` blocks until its healthcheck passes, so pytest can't race the SMTP port. Cost:
pulling `mailhog/mailhog:v1.0.1` is a 145.5 MB download (572 MB unpacked, measured),
plus about 2 s to become healthy. Every PR then checks real SMTP delivery and the
headers.

### Files

| File | Change |
|---|---|
| `pyproject.toml`, `uv.lock` | `uv add aiosmtplib email-validator` |
| `src/message_router/config.py` | four SMTP settings |
| `src/message_router/departments.py` | new: `Department` |
| `src/message_router/mailer.py` | new: validation, message building, sending, `EmailDeliveryError` |
| `docker-compose.yml` | `api`: `SMTP_HOST: mailhog`; `mailhog`: publish `127.0.0.1:1025` |
| `tests/test_mailer.py` | new: unit tests (stubbed `aiosmtplib.send`) |
| `tests/e2e/test_email.py` | new: e2e against MailHog |
| `.github/workflows/ci.yml` | MailHog-only e2e step |
| `CLAUDE.md` | "Commands" updates |

The module is named `mailer.py`, not `email.py`, so it can't shadow the stdlib `email`
package when the package directory is on `sys.path`. The Dockerfile is unchanged:
`uv sync --locked` picks up the new dependencies.

Implementation order: dependencies → settings, `departments.py`, `mailer.py` and unit
tests → compose (+ `docker compose config`) → e2e test → CI → `CLAUDE.md` →
verification.

### Settings: `src/message_router/config.py`

```python
    # Host-run defaults reach the compose MailHog published on 127.0.0.1:1025;
    # docker-compose.yml sets only SMTP_HOST: mailhog.
    smtp_host: str = "localhost"
    smtp_port: int = Field(default=1025, gt=0, lt=65536)
    # Seconds; aiosmtplib applies it to connect and to each SMTP command.
    smtp_timeout: float = Field(default=10.0, gt=0)
    smtp_from: str = "message-router@example.com"
```

### `src/message_router/departments.py`

```python
class Department(StrEnum):
    HUMAN_RESOURCES = "human-resources@example.com"
    HELP_DESK = "help-desk@example.com"
    IT = "it@example.com"
    KADRY = "kadry@example.com"
    OTHER = "other@example.com"  # fallback for unrecognised requests
```

### `src/message_router/mailer.py` (indicative)

```python
SUBJECT_PREFIX = "[message-router]"


class EmailDeliveryError(Exception):
    """The SMTP server was unreachable, timed out, or refused the email."""

    def __init__(self, department: Department, reason: str) -> None: ...


def normalize_requester_email(value: str) -> str:
    """Return one ASCII addr-spec, normalised. Raises ValueError (D3)."""


def build_message(
    department: Department, requester_email: str, message: str, *, sender: str
) -> EmailMessage:
    """Pure, no I/O. Validates everything; raises ValueError."""
    # department = Department(department)    -> ValueError for unknown values
    # requester = normalize_requester_email(requester_email)
    # if not message.strip(): raise ValueError(...)
    # From=sender, To=department, Reply-To=requester,
    # Subject=f"{SUBJECT_PREFIX} Message from {requester}",
    # Date=formatdate(localtime=True),
    # Message-ID=make_msgid(domain=<domain of sender>)
    # set_content(message)                   # text/plain; charset=utf-8


async def send_department_email(
    department: Department, requester_email: str, message: str, *, settings: Settings
) -> str:
    """Build and send; return the Message-ID.

    Raises ValueError for invalid input (before any I/O) and EmailDeliveryError
    for any SMTP failure.
    """
    # msg = build_message(..., sender=settings.smtp_from)
    # try:
    #     await aiosmtplib.send(msg, sender=settings.smtp_from,
    #                           recipients=[department.value],
    #                           hostname=settings.smtp_host, port=settings.smtp_port,
    #                           timeout=settings.smtp_timeout)
    # except aiosmtplib.SMTPException as exc:
    #     raise EmailDeliveryError(department, f"{type(exc).__name__}: {exc}") from exc
```

Notes:

- `make_msgid()` without `domain` calls `socket.getfqdn()`, which can block for seconds
  on a host with broken DNS, so always pass the sender's domain. MailHog doesn't add a
  `Date` header (observed), so we set it.
- `ValidatedEmail.ascii_email` is typed `str | None`. With `allow_smtputf8=False` it is
  always set, so narrow it explicitly for pyright (raise `ValueError` on `None`, don't
  use `assert`).
- The function takes `settings` explicitly, not `get_settings()`: unit tests pass a
  `Settings(...)`, and stage 3 binds it when building the tool.
- `start_tls` stays at aiosmtplib's default (opportunistic). MailHog doesn't advertise
  STARTTLS, so the connection is plain.

### `docker-compose.yml`

```yaml
  api:
    environment:
      OLLAMA_URL: *ollama-url
      OLLAMA_MODEL: *ollama-model
      SMTP_HOST: mailhog
  mailhog:
    ports:
      - "127.0.0.1:8025:8025"      # UI + HTTP API
      - "127.0.0.1:1025:1025"      # SMTP, for host runs and host e2e tests
```

### CI: `.github/workflows/ci.yml`

New steps after "Run tests":

```yaml
      # Only MailHog: no model download, no API build. --wait blocks until healthy.
      - name: Start MailHog
        run: docker compose up -d --wait mailhog

      - name: Email e2e test against MailHog
        run: uv run pytest -m e2e tests/e2e/test_email.py -q

      - name: MailHog logs on failure
        if: failure()
        run: docker compose logs mailhog
```

No teardown step: the runner is discarded after the job.

### `CLAUDE.md`: "Commands"

- Add: email e2e only (needs only MailHog):
  `docker compose up -d --wait mailhog && uv run pytest -m e2e tests/e2e/test_email.py -q`.
- Host run line: replace "no SMTP" with "SMTP goes to the compose MailHog on
  `localhost:1025` if it is running".

## Data & migrations

None. No database. MailHog keeps messages in memory and loses them on restart.

## External calls & cost

- No LLM calls and no token cost.
- SMTP: one connection per send to MailHog (local, ~10 ms).
- MailHog HTTP API (`/api/v2/search`, `DELETE /api/v1/messages/{id}`), used only by
  the e2e test.
- New runtime dependencies: `aiosmtplib` (no dependencies of its own) and
  `email-validator` (pulls in `dnspython` and `idna`). They add a little to the API
  image.
- CI: one 145.5 MB image pull plus about 2 s of MailHog startup per run.

## Edge cases

1. **Unknown department string** (e.g. a typo, or a model output that bypassed the tool
   schema): `Department(value)` raises `ValueError` before any I/O.
2. **CR/LF in the requester email**: rejected by email-validator. `EmailMessage` would
   raise `ValueError` too (second layer).
3. **Several addresses or a display name in the requester email**: rejected, not
   silently reduced to one address.
4. **Non-ASCII local part**: rejected (no SMTPUTF8 in MailHog). An IDN domain is
   accepted and converted to punycode.
5. **Address normalization**: the domain is lowercased and the local part keeps its
   case. `Reply-To` and the subject use the normalized form.
6. **Header-like text in the message** (`\r\n\r\nBcc: x@y.com`) stays in the body.
   Headers come only from code.
7. **Body fidelity**: Polish characters, lines containing only `.` (dot-stuffed by the
   client and unstuffed by the server, verified), and leading or trailing spaces arrive
   unchanged. Two normalizations apply: line endings become CRLF on the wire, and
   `set_content` adds a final `\n` if missing. "Unchanged" means identical after
   normalizing line endings and that final newline. The tests compare it that way.
8. **Lines longer than 998 bytes**: `EmailMessage` and aiosmtplib choose a transfer
   encoding (base64 without `8BITMIME`) that MailHog's UI decodes.
9. **Empty or whitespace-only message**: `ValueError`. A maximum length is stage 3's job
   (HTTP request model).
10. **MailHog down, DNS failure, port closed, timeout**: `EmailDeliveryError` (the
    aiosmtplib types were verified, see D2). Connection refused fails immediately; a
    black-holed host fails after `smtp_timeout`.
11. **Recipient, sender or data refused**: MailHog accepts everything, so this is
    covered only by unit tests with the real aiosmtplib exception classes.
12. **Cancellation mid-send**: aiosmtplib aborts. Whether the server already accepted
    the email is unknown, and that is acceptable for a PoC.
13. **Host port 1025 busy**: `docker compose up` fails with "address already in use",
    as with 8000 and 8025.
14. **The requester is never an envelope recipient** (D7), so no copy goes to them.
15. **The e2e test runs while a person is using MailHog, or other emails exist**:
    isolated by the unique Message-ID, and the test deletes only its own message.

## Test plan

Unit tests (`uv run pytest -m "not e2e" -x -q`; all `@pytest.mark.unit`; no network)
in `tests/test_mailer.py`. A fixture replaces `aiosmtplib.send` (via
`monkeypatch.setattr`) with an async fake that records its arguments and can raise.
Settings are built as `Settings(ollama_model="test-model")`.

1. `Department` has exactly the five addresses; `Department("ceo@example.com")` raises
   `ValueError`.
2. `build_message` headers: To, From, Reply-To, Subject (exact string), Date and
   Message-ID are present; no Cc/Bcc; `text/plain; charset="utf-8"`.
3. Body fidelity: a multi-line message with Polish characters, a lone `.` line and
   header-like lines round-trips through `get_content()` (normalized as in edge case 7).
4. Header injection in the body: after `as_bytes()` and reparsing, the message has no
   `Bcc` header and the text is still in the body.
5. Invalid requester email (parametrized: CRLF, LF, comma list, display name, non-ASCII
   local part, no `@`, `jan@localhost`, empty) raises `ValueError`, and the fake send
   is **not** called.
6. Invalid department through `send_department_email` raises `ValueError`; the fake
   send is not called.
7. An empty or whitespace-only message raises `ValueError`; the fake send is not called.
8. Requester normalization: `Jan.Nowak@EXAMPLE.com` becomes `Jan.Nowak@example.com`
   in Reply-To and Subject; an IDN domain is converted to punycode.
9. The send call gets `sender=smtp_from`, `recipients=[department.value]`,
   `hostname`, `port` and `timeout` from settings. The return value equals the message's
   Message-ID.
10. Failure mapping (parametrized with real aiosmtplib exception instances:
    `SMTPConnectError`, `SMTPConnectTimeoutError`, `SMTPServerDisconnected`,
    `SMTPRecipientsRefused`, `SMTPDataError`) raises `EmailDeliveryError` with
    `__cause__` set to the original. The message names the department and doesn't
    contain the body.
11. Settings defaults (with `SMTP_*` removed from the environment via `monkeypatch`):
    `localhost`, `1025`, `10.0`, `message-router@example.com`.

e2e tests (`@pytest.mark.e2e`; need MailHog running; run locally and in CI) in
`tests/e2e/test_email.py`:

1. `test_email_delivered_to_mailhog`
   - Arrange: `token = uuid4().hex`; a tricky message (Polish characters, a `.` line,
     a `Bcc:`-looking line, the token); requester `jan.nowak@example.com`.
   - Act: `await send_department_email(Department.IT, ..., settings=Settings(ollama_model="unused"))`.
     Explicit kwargs override only the model, so `SMTP_*` still come from the defaults
     or the environment.
   - Find: poll `GET /api/v2/search?kind=containing&query=<returned Message-ID>` (up
     to ~5 s, 0.1 s interval) with httpx2's async client. Not by the token: the body
     is base64 and MailHog's search doesn't decode it (verified: 0 hits by body token,
     1 by Message-ID). Expect exactly one item. Delete it in a
     fixture finalizer with `DELETE /api/v1/messages/{ID}`. **Never**
     `DELETE /api/v1/messages`, which would wipe the inbox someone may be looking at.
   - Assert, after parsing `Raw.Data` with
     `email.message_from_string(..., policy=policy.default)` (MailHog's
     `Content.Headers` hold raw RFC 2047 text, so don't compare them directly):
     To `it@example.com`; Reply-To the requester; Subject exact; Message-ID equals
     the returned value; no Bcc header; body equals the message (normalized). Envelope
     `Raw.To == ["it@example.com"]` and `Raw.From == smtp_from`.
2. `test_unreachable_smtp_raises_delivery_error`: settings point to a closed loopback
   port (found by binding a socket to port 0 and closing it), and the call raises
   `EmailDeliveryError` quickly. This uses only loopback, but it lives here because it
   does real I/O.

`tests/e2e/test_stack.py` is unchanged; the full `uv run pytest -m e2e -q` still needs
the whole stack.

### Done criteria (run and show output)

```bash
uv run ruff format . && uv run ruff check --fix .
uv run pyright
uv run pytest -m "not e2e" -x -q
uv run pre-commit run --all-files
docker compose config        # required by CLAUDE.md for compose/settings changes
```

## End-to-end verification steps

```bash
# 1. Compose wiring: only SMTP_HOST in api env; both MailHog ports on loopback
docker compose config -q && echo "OK: compose config valid"
docker compose config | grep -nE 'SMTP_|host_ip|published|target'
#    expect: SMTP_HOST: mailhog (no SMTP_PORT); mailhog 1025 and 8025 on 127.0.0.1

# 2. What CI runs: MailHog only + email e2e
docker compose up -d --wait mailhog
uv run pytest -m e2e tests/e2e/test_email.py -q          # 2 passed
curl -s http://localhost:8025/api/v2/messages | python3 -c 'import json,sys; print(json.load(sys.stdin)["total"])'
#    expect: unchanged from before the run (the test deletes its own message)

# 3. Manual send from the host; open http://localhost:8025 and read it
OLLAMA_MODEL=unused uv run python -c 'import asyncio
from message_router.config import get_settings
from message_router.departments import Department
from message_router.mailer import send_department_email
print(asyncio.run(send_department_email(Department.HELP_DESK, "jan.nowak@example.com", "Nie działa mi komputer", settings=get_settings())))'
curl -s 'http://localhost:8025/api/v2/search?kind=to&query=help-desk@example.com' \
  | python3 -c 'import json,sys,email; from email import policy
for i in json.load(sys.stdin)["items"]:
    m = email.message_from_string(i["Raw"]["Data"], policy=policy.default)
    print(m["To"], "|", m["Reply-To"], "|", m["Subject"], "|", m.get_content().strip())'
#    expect: help-desk@example.com | jan.nowak@example.com | [message-router] Message from jan.nowak@example.com | Nie działa mi komputer
#    UI: Polish characters readable in the message view

# 4. Container wiring: send from inside the API container (proves SMTP_HOST: mailhog)
docker compose up -d --build --wait
docker compose exec api env | grep '^SMTP_'               # SMTP_HOST=mailhog only
docker compose exec api python -c 'import asyncio
from message_router.config import get_settings
from message_router.departments import Department
from message_router.mailer import send_department_email
print(asyncio.run(send_department_email(Department.KADRY, "jan.nowak@example.com", "Chciałbym zgłosić urlop na jutro", settings=get_settings())))'
#    expect: a Message-ID; a new email to kadry@example.com in the UI

# 5. Failure mapping against a real stopped MailHog
docker compose stop mailhog
OLLAMA_MODEL=unused uv run python -c '<same snippet as step 3>'; echo "exit=$?"
#    expect: EmailDeliveryError ... SMTPConnectError ... immediately; exit=1
docker compose start mailhog

# 6. Stage 1 e2e still green with the full stack
uv run pytest -m e2e -q

# 7. Fresh clone, no .env (after the implementation is committed)
#    as stage 1 step 1-4 with --branch feat/stage-2-email-sending; plus
docker compose ps --format '{{.Service}}: {{.Ports}}'
#    expect: mailhog 127.0.0.1:1025->1025/tcp, 127.0.0.1:8025->8025/tcp
```

## Open questions

None block stage 2. Items 1-4 are deliberately deferred to the stage 3 spec: they depend
on the agent and endpoint design, and stage 2's interface (`ValueError`,
`EmailDeliveryError`, `normalize_requester_email`) already supports every option.

1. **Stage 3 tool shape:** expose only `department` to the model and bind
   `requester_email`, `message` and `settings` per request (a closure-built tool or
   LangChain's injected-argument mechanism). Check the installed LangChain version's
   docs then.
2. **Stage 3 request validation:** use `normalize_requester_email` through an
   `AfterValidator`, not plain `EmailStr`. They differ on display names and non-ASCII
   local parts (D3). To keep `format: email` in Swagger, add it with
   `json_schema_extra`.
3. **Stage 3 error mapping:** the HTTP status for `EmailDeliveryError` (502 or 503),
   and whether the agent may call the tool again after a delivery failure (risk of
   duplicate emails).
4. **Total send budget:** aiosmtplib's timeout is per operation. Stage 3 should set one
   request-level deadline around the whole agent run (`asyncio.timeout`), since LLM
   latency dominates; no change to the mailer is needed.
5. **CI e2e cost:** resolved. The MailHog pull is 145.5 MB (not ~7 MB). Decided: keep the
   email e2e test in CI (D9).
