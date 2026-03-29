# HubEstudos Backend

Backend FastAPI para processamento de PDFs de concursos com IA (OpenAI GPT-4o) e Supabase.

## Pré-requisitos

- Python 3.12+
- Conta Supabase com tabelas criadas (schema abaixo)
- Chave de API da OpenAI

## Setup Local

```bash
# 1. Clone e entre na pasta
cd hub-estudos-backend

# 2. Crie ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instale dependências
pip install -r requirements.txt

# 4. Configure variáveis de ambiente
cp .env.example .env
# Edite .env com suas credenciais

# 5. Rode o servidor
uvicorn app.main:app --reload --port 8000
```

A API estará em `http://localhost:8000`
Documentação automática: `http://localhost:8000/docs`

## Variáveis de Ambiente

| Variável | Descrição |
|---|---|
| `SUPABASE_URL` | URL do projeto Supabase (ex: `https://xxx.supabase.co`) |
| `SUPABASE_SERVICE_ROLE_KEY` | Service Role Key do Supabase (Settings → API) |
| `AI_API_KEY` | Chave da OpenAI (`sk-...`) |
| `AI_MODEL` | Modelo a usar (padrão: `gpt-4o`) |

## Endpoints

### Health
| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status da API e conexão com Supabase |

### Pipeline
| Método | Rota | Descrição |
|---|---|---|
| POST | `/api/pipeline/process` | Inicia pipeline completo para um PDF |
| GET | `/api/pipeline/status/{pdf_upload_id}` | Polling do status de processamento |
| POST | `/api/generate-analysis` | Gera justificativas e peguinhas (batch) |
| POST | `/api/extract-syllabus` | Extrai conteúdo programático de texto |

### Questões
| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/questions/{study_plan_id}` | Lista questões de um plano |
| GET | `/api/questions/{study_plan_id}/{question_id}/analysis` | Justificativas e peguinhas de uma questão |

## Fluxo Principal

```
Frontend (Lovable)
    │
    ├─ Upload PDF → Supabase Storage (bucket: pdfs)
    ├─ Insere registro em pdf_uploads (status: pending)
    ├─ POST /api/pipeline/process { pdf_upload_id }
    │       └─ Resposta imediata: { status: "processing" }
    │
    └─ Polling: GET /api/pipeline/status/{id}
            └─ Quando status = "completed" → questões já no Supabase
```

## Rodar com Docker

```bash
docker build -t hub-estudos-backend .
docker run -p 8000:8000 --env-file .env hub-estudos-backend
```

## Testes

```bash
pytest tests/ -v
```

## Schema Supabase

Execute no SQL Editor do Supabase:

### Migration — campos de progresso e cache de texto

Execute **após** criar as tabelas base:

```sql
-- Campos de progresso granular (pipeline escalável)
alter table pdf_uploads
  add column if not exists progress int default 0,
  add column if not exists processing_stage text,
  add column if not exists last_heartbeat_at timestamptz,
  add column if not exists processing_started_at timestamptz,
  add column if not exists completed_at timestamptz,
  add column if not exists cancel_requested boolean default false,
  -- Cache de texto extraído (evita re-download/OCR em reprocessamentos)
  add column if not exists text_content text,
  -- Nome do concurso extraído pelo parser (sem expor a banca)
  -- Ex: "Câmara de Goiânia - GO - Revisor de Texto (2026)"
  add column if not exists concurso_name text,
  -- Quantidade de questões extraídas (preenchido ao completar)
  add column if not exists questions_count int;
```

### Suporte na Edge Function `process-callback`

Além de `save_text_content`, adicione o campo `concurso_name` ao handler `update_pdf_status`:

```typescript
// No case "update_pdf_status", aceitar o campo concurso_name
case "update_pdf_status": {
  const {
    pdf_id, status, progress, processing_stage, error_message,
    processing_started_at, completed_at, cancel_requested,
    concurso_name,        // ← NOVO: nome do concurso sem banca
    questions_count,      // ← NOVO: contagem de questões
  } = data;

  const update: Record<string, unknown> = {};
  if (status !== undefined)               update.status = status;
  if (progress !== undefined)             update.progress = progress;
  if (processing_stage !== undefined)     update.processing_stage = processing_stage;
  if (error_message !== undefined)        update.error_message = error_message;
  if (processing_started_at !== undefined) update.processing_started_at = processing_started_at;
  if (completed_at !== undefined)         update.completed_at = completed_at;
  if (cancel_requested !== undefined)     update.cancel_requested = cancel_requested;
  if (concurso_name !== undefined)        update.concurso_name = concurso_name;
  if (questions_count !== undefined)      update.questions_count = questions_count;

  await supabase.from("pdf_uploads").update(update).eq("id", pdf_id);
  return { ok: true };
}

// action: "save_text_content"
case "save_text_content": {
  const { pdf_id, text_content } = data;
  await supabase.from("pdf_uploads").update({ text_content }).eq("id", pdf_id);
  return { ok: true };
}
```

Suporte ao campo `save_text_content` na Edge Function `process-callback`:
```typescript
// action: "save_text_content"
case "save_text_content": {
  const { pdf_id, text_content } = data;
  await supabase.from("pdf_uploads").update({ text_content }).eq("id", pdf_id);
  return { ok: true };
}
```

---

## Schema Supabase

Execute no SQL Editor do Supabase:

```sql
-- Tabela de uploads
create table pdf_uploads (
  id uuid primary key default gen_random_uuid(),
  study_plan_id uuid not null,
  user_id uuid,
  file_path text not null,
  file_name text,
  type text check (type in ('exam', 'syllabus')),
  status text default 'pending' check (status in ('pending','processing','completed','error')),
  error_message text,
  created_at timestamptz default now()
);

-- Disciplinas
create table subjects (
  id uuid primary key default gen_random_uuid(),
  study_plan_id uuid not null,
  name text not null,
  created_at timestamptz default now()
);

-- Questões
create table questions (
  id uuid primary key default gen_random_uuid(),
  study_plan_id uuid not null,
  subject_id uuid references subjects(id),
  source_pdf_id uuid references pdf_uploads(id),
  statement text not null,
  alternatives jsonb,
  correct_answer text,
  topic text,
  difficulty text,
  created_at timestamptz default now()
);

-- Justificativas
create table justifications (
  id uuid primary key default gen_random_uuid(),
  question_id uuid not null references questions(id) on delete cascade,
  alternative text,
  is_correct boolean,
  justification text,
  created_at timestamptz default now()
);

-- Peguinhas
create table tricky_points (
  id uuid primary key default gen_random_uuid(),
  question_id uuid not null references questions(id) on delete cascade,
  description text,
  misleading_alternative text,
  deduction_tip text,
  created_at timestamptz default now()
);

-- Conteúdo programático
create table syllabus_topics (
  id uuid primary key default gen_random_uuid(),
  study_plan_id uuid not null,
  subject_name text,
  topic_title text,
  parent_topic_id uuid references syllabus_topics(id),
  order_index int default 0,
  is_completed boolean default false,
  created_at timestamptz default now()
);
```
