# AI Message Router

Proof of concept routera wiadomości opartego na AI. API przyjmuje adres e-mail nadawcy
i wiadomość, agent AI z lokalnym modelem (Ollama) wybiera dział i wysyła do niego
e-mail, wywołując narzędzie, z nagłówkiem `Reply-To` ustawionym na nadawcę. E-maile
przechwytuje MailHog.

## Uruchomienie

Wymagania: Docker z Compose, około 4 GB wolnej pamięci RAM i 3 GB miejsca na dysku na
model.

```bash
docker compose up -d
```

Pierwsze uruchomienie trwa kilka minut (budowa obrazu API i pobranie modelu, 2,5 GB).
Komenda kończy się, gdy model jest załadowany, a API gotowe na żądania.

| Co | Adres |
|---|---|
| Swagger | <http://localhost:8000/api/v1/docs> |
| MailHog (przechwycone e-maile) | <http://localhost:8025> |

Zatrzymanie: `docker compose down` (`-v` usuwa też pobrany model).

### Docker Desktop (macOS, Windows)

Na macOS i Windowsie kontenery działają w maszynie wirtualnej Docker Desktop, która
dostaje tylko część pamięci RAM hosta. Trzeba jej przydzielić co najmniej 6 GB. To
ustawienie hosta, `docker-compose.yml` nie może go zmienić.

- macOS: Docker Desktop → Settings → Resources → Memory.
- Windows (WSL2): w pliku `%UserProfile%\.wslconfig` w sekcji `[wsl2]` wpisać
  `memory=6GB`, potem `wsl --shutdown` i ponownie uruchomić Docker Desktop.

Pamięć dostępną dla Dockera (w bajtach) pokazuje
`docker info --format '{{.MemTotal}}'`. Jeśli jest jej za mało, `docker compose up -d`
kończy się błędem serwisu `ollama-pull`, a przyczynę pokazuje
`docker compose logs ollama-pull`.

### Zajęty port

Stack wystawia na `127.0.0.1` porty 8000 (API), 8025 (MailHog UI) i 1025 (MailHog
SMTP). Jeśli któryś jest zajęty, `docker compose up -d` kończy się błędem
`Bind for 127.0.0.1:8000 failed: port is already allocated`. Proces, który zajmuje
port (tu 8000), można znaleźć tak:

```bash
ss -ltnp | grep :8000                  # Linux
lsof -i :8000                          # macOS
netstat -ano | findstr :8000           # Windows, potem: tasklist /FI "PID eq <PID>"
```

Można zatrzymać ten proces albo zmienić w `docker-compose.yml` tylko port po stronie
hosta (lewa strona mapowania), np. `"127.0.0.1:8080:8000"`. Wtedy w adresach powyżej
i w przykładowym cURL-u trzeba użyć nowego portu.

## Przykładowe zapytanie

Linux, macOS:

```bash
curl -s -X POST http://localhost:8000/api/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"email": "jan.nowak@example.com", "message": "I forgot my password and cannot log in"}'
```

Windows, wiersz poleceń `cmd` (`curl.exe` jest wbudowany w Windows 10 i 11):

```bat
curl -s -X POST http://localhost:8000/api/v1/messages ^
  -H "Content-Type: application/json" ^
  -d "{\"email\": \"jan.nowak@example.com\", \"message\": \"I forgot my password and cannot log in\"}"
```

Windows, PowerShell (tu `curl` bywa aliasem `Invoke-WebRequest`, więc prościej użyć
`Invoke-RestMethod`):

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/messages `
  -ContentType 'application/json' `
  -Body '{"email": "jan.nowak@example.com", "message": "I forgot my password and cannot log in"}'
```

Odpowiedź:

```json
{"department":"help-desk@example.com","fallback":false,"message_id":"<179487979763.1.13250721382118444477@example.com>"}
```

W MailHog pojawia się e-mail do `help-desk@example.com` z
`Reply-To: jan.nowak@example.com`.

## Działy

Model wybiera dział według zakresów poniżej. Trafiają one do promptu systemowego z
`DEPARTMENT_SCOPES` w [`src/message_router/departments.py`](src/message_router/departments.py).

| Dział | Zakres |
|---|---|
| `kadry@example.com` | sprawy kadrowo-administracyjne: urlopy, L4, umowy o pracę, wypłaty, zaświadczenia |
| `human-resources@example.com` | rekrutacja, szkolenia, rozwój, benefity, konflikty w zespole |
| `help-desk@example.com` | wsparcie pierwszej linii: niedziałający komputer, reset hasła, drukarka, klient poczty |
| `it@example.com` | infrastruktura, serwery, wnioski o dostęp i uprawnienia, zakup lub wdrożenie oprogramowania |
| `other@example.com` | wszystko inne (fallback) |

## Decyzje architektoniczne

- **Python, FastAPI, LangChain:** FastAPI daje walidację żądań i Swaggera, LangChain
  (`create_agent` + `ChatOllama`) daje agenta z wywoływaniem narzędzi.
- **Model wybiera tylko dział:** jedyny argument narzędzia `send_email` to enum pięciu
  adresów. Adres nadawcy i treść trafiają do narzędzia z kodu, więc model nie może
  wysłać e-maila poza listę ani zmienić `Reply-To`.
- **Jedno wywołanie modelu na żądanie:** wynik narzędzia kończy pracę agenta
  (`return_direct`). Typowe żądanie trwa 1,5–2 s na CPU (zależnie od procesora),
  wiadomość bliska limitu 2000 znaków około 8 s.
- **Zabezpieczenie awaryjne:** Ollama nie pozwala wymusić wywołania narzędzia. Jeśli
  model go nie wywoła, API samo wysyła e-mail na `other@example.com` i zwraca
  `fallback: true`. W ewaluacji nie zdarzyło się to ani razu (0 z 64).
- **Model `qwen3:4b-instruct-2507-q4_K_M` na `alpine/ollama` (CPU):** najmniejszy
  model, który przeszedł ewaluację: 100% wywołań narzędzia, 90,6% trafnych działów.
  Metodyka, porównanie modeli i analiza błędów są w
  `docs/evals/README.md`.
- **Gotowość po `up -d`:** jednorazowy serwis `ollama-pull` pobiera model i ładuje go
  do RAM, a API startuje dopiero po nim i rozgrzewa cache promptu. Model zostaje w
  pamięci (`OLLAMA_KEEP_ALIVE=-1`).
- **Prompt injection:** wiadomość jest ujęta w znaczniki `<message>`, a model ma
  zakaz wykonywania zawartych w niej poleceń.
