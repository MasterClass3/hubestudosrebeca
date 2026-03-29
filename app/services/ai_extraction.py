import json
import logging
import re
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable
import anthropic

from app.config import get_settings
from app.services.callback_service import get_client
from app.services.ai_analysis import generate_analysis_parallel

logger = logging.getLogger(__name__)

BATCH_SIZE = 2           # 1-2 questões por chamada de IA → mais rápido em paralelo
CHAR_CHUNK_SIZE = 12_000  # Fallback: chunking por caracteres
AI_CALL_TIMEOUT = 120.0

EXTRACTION_PROMPT = """
Você é um especialista em concursos públicos brasileiros. Analise o texto abaixo extraído de um PDF de prova.

FORMATO DO PDF:
- Questões identificadas por "Questão" (o número pode aparecer como caractere especial/nulo — ignore)
- Cada questão tem: Enunciado, Alternativas (A B C D ou A B C D E), RESPOSTA CORRETA, ALTERNATIVAS ERRADAS, DICA DE ELIMINAÇÃO
- As justificativas e dicas já estão no texto — EXTRAIA-AS diretamente, não invente

INSTRUÇÕES:
1. Extraia TODAS as questões encontradas no trecho
2. Para cada questão, capture:
   - statement: texto do Enunciado
   - alternatives: array com cada alternativa separada no formato [{"letter":"A","text":"..."}]
   - correct_answer: letra da RESPOSTA CORRETA (ex: "C")
   - topic: disciplina inferida (Português, Matemática, Direito Constitucional, Informática, etc.)
   - difficulty: "easy", "medium" ou "hard"
   - justifications: array com uma justificativa por alternativa, extraída de RESPOSTA CORRETA e ALTERNATIVAS ERRADAS
   - tricky_points: array com dicas de DICA DE ELIMINAÇÃO (pode ser vazio)

Retorne SOMENTE JSON válido neste formato:
{
  "questions": [
    {
      "statement": "texto do enunciado",
      "alternatives": [
        {"letter": "A", "text": "texto alternativa A"},
        {"letter": "B", "text": "texto alternativa B"},
        {"letter": "C", "text": "texto alternativa C"},
        {"letter": "D", "text": "texto alternativa D"}
      ],
      "correct_answer": "C",
      "topic": "Língua Portuguesa",
      "difficulty": "medium",
      "justifications": [
        {"alternative": "A", "is_correct": false, "justification": "explicação de por que A está errada"},
        {"alternative": "B", "is_correct": false, "justification": "explicação de por que B está errada"},
        {"alternative": "C", "is_correct": true,  "justification": "explicação de por que C está correta"},
        {"alternative": "D", "is_correct": false, "justification": "explicação de por que D está errada"}
      ],
      "tricky_points": [
        {
          "description": "texto da dica de eliminação",
          "misleading_alternative": "A",
          "deduction_tip": "como descartar sem saber o conteúdo"
        }
      ]
    }
  ]
}

Se não houver questões no trecho, retorne {"questions": []}.

TEXTO DO PDF:
"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1).strip()
    return json.loads(raw)


# ── Chunking strategies ──────────────────────────────────────────────────────

# Patterns tentados em ordem de confiança decrescente.
# Cada entrada: (pattern, min_ratio_for_acceptance)
# min_ratio = fração mínima de blocos que precisam parecer "questão real" para aceitar o split.
_BOUNDARY_PATTERNS: list[tuple[re.Pattern, float]] = [
    # Tier 1 — "Questão N" ou "QUESTÃO N" com dígito (alta confiança)
    (re.compile(r"(?m)^[ \t]*(?:Quest[aã]o|QUEST[AÃ]O)[ \t]+\d"), 0.3),
    # Tier 2 — "Questão" isolado na linha (número pode estar na linha seguinte como null-char)
    (re.compile(r"(?m)^[ \t]*(?:Quest[aã]o|QUEST[AÃ]O)[ \t]*$"), 0.3),
    # Tier 3 — "Questão " genérico
    (re.compile(r"(?m)^[ \t]*Quest[aã]o[ \t]"), 0.5),
    # Tier 4 — "01. " ou "(01) " — comum em simulados/bancos de questões
    (re.compile(r"(?m)^[ \t]*(?:\(\d{1,3}\)|\d{1,3}\.)\s"), 0.7),
    # Tier 5 — "01) " — outro estilo numerado
    (re.compile(r"(?m)^[ \t]*\d{1,3}\)\s"), 0.7),
]

# Uma questão "real" tem alternativas (A) ... (E), A) ... E) ou é estilo CESPE (Certo/Errado)
_QUESTION_INDICATOR = re.compile(
    r"(?m)"
    r"(?:^[ \t]*[A-Ea-e][\)\.] )"      # A) ou A. no início de linha
    r"|(?:\([A-Ea-e]\))"                 # (A) em qualquer posição
    r"|(?i:^[ \t]*Certo\b)"             # CESPE Certo/Errado
    r"|(?i:Enunciado)",                  # Rótulo explícito
    re.MULTILINE,
)


def _split_questions(text: str) -> list[str]:
    """
    Tenta dividir o texto nos limites de cada questão.
    Usa padrões em ordem de confiança. Retorna blocos individuais ou [] se nenhum funcionar.
    """
    for pattern, min_ratio in _BOUNDARY_PATTERNS:
        positions = [m.start() for m in pattern.finditer(text)]
        if len(positions) < 2:
            continue

        blocks = []
        for i, pos in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            block = text[pos:end].strip()
            if block:
                blocks.append(block)

        real = [b for b in blocks if _QUESTION_INDICATOR.search(b)]
        ratio = len(real) / len(blocks) if blocks else 0

        if real and ratio >= min_ratio:
            logger.info(
                f"Smart chunking OK: pattern='{pattern.pattern[:40]}' "
                f"total={len(blocks)} real={len(real)} ratio={ratio:.0%}"
            )
            return real

        logger.debug(
            f"Smart chunking SKIP: pattern='{pattern.pattern[:40]}' "
            f"total={len(blocks)} real={len(real)} ratio={ratio:.0%} (min={min_ratio:.0%})"
        )

    return []


def _char_chunks(text: str) -> list[str]:
    """Fallback: divide por quantidade de caracteres."""
    if len(text) <= CHAR_CHUNK_SIZE:
        return [text]
    return textwrap.wrap(text, CHAR_CHUNK_SIZE, break_long_words=False, break_on_hyphens=False)


def _build_batches(text: str, batch_size: int) -> tuple[list[str], bool, int]:
    """
    Retorna (batches, usado_smart_chunking, total_questoes_estimado).
    Quando smart=True, total_questoes_estimado é exato.
    Quando smart=False, é -1 (desconhecido).
    """
    clean = text.replace("\x00", " ").replace("\ufffd", "")

    questions = _split_questions(clean)
    if questions:
        batches = [
            "\n\n---\n\n".join(questions[i: i + batch_size])
            for i in range(0, len(questions), batch_size)
        ]
        return batches, True, len(questions)

    logger.info("Smart chunking: nenhum padrão encontrado — usando fallback por caracteres")
    return _char_chunks(clean), False, -1


# ── Processamento de um único batch ──────────────────────────────────────────

def _extract_chunk(
    api_key: str,
    model: str,
    chunk: str,
    chunk_idx: int,
    total_chunks: int,
    pdf_upload_id: str,
) -> tuple[int, list[dict]]:
    """Processa um batch em thread separada. Retorna (chunk_idx, questions)."""
    logger.info(
        f"[Extraction:{pdf_upload_id}] batch {chunk_idx+1}/{total_chunks} "
        f"iniciando ({len(chunk)} chars)"
    )
    t0 = time.time()

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            timeout=AI_CALL_TIMEOUT,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT + chunk}],
        )
    except anthropic.APITimeoutError:
        raise RuntimeError(f"Timeout no batch {chunk_idx+1}/{total_chunks}")
    except anthropic.APIError as e:
        raise RuntimeError(f"Erro API Claude batch {chunk_idx+1}: {e}")

    raw = response.content[0].text if response.content else ""
    if not raw.strip():
        logger.warning(f"[Extraction:{pdf_upload_id}] batch {chunk_idx+1} retornou vazio")
        return chunk_idx, []

    try:
        questions = _parse_json(raw).get("questions", [])
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(
            f"[Extraction:{pdf_upload_id}] batch {chunk_idx+1} JSON inválido: {e} "
            f"| raw[:200]={raw[:200]}"
        )
        return chunk_idx, []

    elapsed = time.time() - t0
    logger.info(
        f"[Extraction:{pdf_upload_id}] batch {chunk_idx+1} OK — "
        f"{len(questions)} questões em {elapsed:.1f}s"
    )
    return chunk_idx, questions


# ── Entry point principal ─────────────────────────────────────────────────────

def extract_and_save_questions(
    raw_text: str,
    study_plan_id: str,
    source_pdf_id: str,
    pdf_upload_id: str = "",
    heartbeat_fn: Callable[[int, str], None] | None = None,
    pdf_bytes: bytes = b"",
    user_id: str = "",
) -> list[str]:
    """
    Extrai questões em paralelo e faz batch insert.

    Estratégia:
    - Smart chunking (por limite de questão) → BATCH_SIZE questões por chamada
    - Fallback para chunking por caracteres se regex não encontrar padrão
    - ThreadPoolExecutor com extraction_parallelism workers
    - Heartbeat granular: "Extraindo questão ~X de ~Y" quando smart=True

    Metas de tempo:
    - 50 questões:  < 1.5 min
    - 100 questões: < 3 min
    - 200 questões: < 5 min
    """
    settings = get_settings()
    cb = get_client()

    t_start = time.time()

    batches, smart, total_questions_estimate = _build_batches(raw_text, BATCH_SIZE)
    total_batches = len(batches)

    logger.info(
        f"[Extraction:{pdf_upload_id}] START — {total_batches} batches "
        f"(smart={smart}, batch_size={BATCH_SIZE}, questoes_estimadas={total_questions_estimate}) | "
        f"modelo={settings.ai_extraction_model} | workers={settings.extraction_parallelism}"
    )

    # ── Paralelizar chamadas ao Claude ────────────────────────────────────
    completed_count = 0
    questions_so_far = 0
    lock = threading.Lock()
    results_by_idx: dict[int, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=settings.extraction_parallelism) as executor:
        futures = {
            executor.submit(
                _extract_chunk,
                settings.anthropic_api_key,
                settings.ai_extraction_model,
                batch,
                i,
                total_batches,
                pdf_upload_id,
            ): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            try:
                idx, questions = future.result()
                results_by_idx[idx] = questions
            except Exception as e:
                idx = futures[future]
                logger.error(f"[Extraction:{pdf_upload_id}] batch {idx+1} falhou: {e}")
                results_by_idx[idx] = []
                questions = []

            with lock:
                completed_count += 1
                questions_so_far += len(questions)
                done = completed_count
                q_done = questions_so_far

            if heartbeat_fn:
                progress = 35 + int((done / total_batches) * 47)  # 35% → 82%
                if smart and total_questions_estimate > 0:
                    stage = f"Extraindo questão ~{q_done} de ~{total_questions_estimate}"
                else:
                    stage = f"Processando bloco {done}/{total_batches} (~{q_done} questões)"
                heartbeat_fn(progress, stage)

    # Reagrupa na ordem original
    all_questions: list[dict[str, Any]] = []
    for i in range(total_batches):
        all_questions.extend(results_by_idx.get(i, []))

    t_ai = time.time() - t_start
    logger.info(
        f"[Extraction:{pdf_upload_id}] IA concluída — "
        f"{len(all_questions)} questões | {total_batches} batches | AI={t_ai:.1f}s"
    )

    if not all_questions:
        return []

    if heartbeat_fn:
        heartbeat_fn(83, "Criando disciplinas")

    # ── Upsert de disciplinas (sem duplicata) ─────────────────────────────
    t_db = time.time()
    subject_cache: dict[str, str] = {}
    topics = list({q.get("topic", "Geral") for q in all_questions})
    for topic in topics:
        subject_cache[topic] = cb.upsert_subject(topic, study_plan_id)

    if heartbeat_fn:
        heartbeat_fn(84, "Extraindo imagens do PDF")

    # ── Extração de imagens (não-bloqueante) ──────────────────────────────
    # Para o caminho IA, as questões são numeradas sequencialmente (1, 2, 3…)
    # O image_service detecta posições via fitz, que pode não coincidir 100%
    # com a numeração da IA. Usamos o mapa por número como melhor heurística.
    q_image_map: dict[int, list[dict]] = {}
    if pdf_bytes and user_id:
        try:
            from app.services.image_service import extract_question_images
            if heartbeat_fn:
                heartbeat_fn(85, "Extraindo e enviando imagens")
            q_image_map = extract_question_images(
                pdf_bytes=pdf_bytes,
                study_plan_id=study_plan_id,
                pdf_upload_id=pdf_upload_id,
                user_id=user_id,
                cb=cb,
            )
            logger.info(
                f"[Extraction:{pdf_upload_id}] imagens em {len(q_image_map)} questões"
            )
        except Exception as img_err:
            logger.warning(
                f"[Extraction:{pdf_upload_id}] extração de imagens falhou "
                f"(não bloqueante): {img_err}"
            )

    if heartbeat_fn:
        heartbeat_fn(86, f"Salvando {len(all_questions)} questões no banco")

    # ── Batch insert de questões ──────────────────────────────────────────
    from app.services.image_service import has_image_reference

    questions_payload = []
    for i, q in enumerate(all_questions):
        q_num   = i + 1   # numeração sequencial para lookup no image_map
        imgs    = q_image_map.get(q_num, [])
        topic   = q.get("topic", "Geral")
        difficulty = q.get("difficulty", "medium")

        # Questão menciona imagem mas nenhuma foi encontrada → alerta visual
        if not imgs and has_image_reference(q.get("statement", "")):
            difficulty = "image_missing"

        entry: dict = {
            "subject_id":    subject_cache.get(topic),
            "statement":     q.get("statement", ""),
            "alternatives":  q.get("alternatives", []),
            "correct_answer": q.get("correct_answer", ""),
            "topic":         topic,
            "difficulty":    difficulty,
        }
        # Só inclui image_urls quando há imagens — evita erro SQL se a coluna
        # ainda não existir na tabela questions
        if imgs:
            entry["image_urls"] = imgs
        questions_payload.append(entry)
    question_ids = cb.insert_questions(questions_payload, study_plan_id, source_pdf_id)

    if not question_ids:
        logger.warning(f"[Extraction:{pdf_upload_id}] insert_questions não retornou IDs")
        return []

    if heartbeat_fn:
        heartbeat_fn(90, "Salvando justificativas")

    # ── Batch insert de justificativas e peguinhas ────────────────────────
    all_justifications = []
    all_tricky_points = []

    for q, question_id in zip(all_questions, question_ids):
        for j in q.get("justifications", []):
            if j.get("alternative") and j.get("justification"):
                all_justifications.append({
                    "question_id": question_id,
                    "alternative": j["alternative"],
                    "is_correct": j.get("is_correct", False),
                    "justification": j["justification"],
                })
        for tp in q.get("tricky_points", []):
            if tp.get("description"):
                all_tricky_points.append({
                    "question_id": question_id,
                    "description": tp["description"],
                    "misleading_alternative": tp.get("misleading_alternative"),
                    "deduction_tip": tp.get("deduction_tip", ""),
                })

    if all_justifications:
        cb.insert_justifications(all_justifications)

    if heartbeat_fn:
        heartbeat_fn(94, "Salvando peguinhas")

    if all_tricky_points:
        cb.insert_tricky_points(all_tricky_points)

    t_total = time.time() - t_start
    logger.info(
        f"[Extraction:{pdf_upload_id}] FINISH — "
        f"{len(question_ids)} questões, {len(all_justifications)} justificativas, "
        f"{len(all_tricky_points)} peguinhas | "
        f"total={t_total:.1f}s (AI={t_ai:.1f}s, DB={time.time()-t_db:.1f}s)"
    )
    return question_ids


# ── Caminho rápido: questões pré-extraídas pelo smart parser ─────────────────

def save_parsed_questions(
    parsed_questions: list,         # list[smart_parser.ParsedQuestion]
    study_plan_id: str,
    source_pdf_id: str,
    pdf_upload_id: str = "",
    heartbeat_fn: Callable[[int, str], None] | None = None,
    pdf_bytes: bytes = b"",
    user_id: str = "",
) -> list[str]:
    """
    Salva questões que já foram extraídas via regex pelo smart parser (sem IA).
    Usa IA em paralelo apenas para justificativas e peguinhas.

    Fluxo:
      smart_parser → save_parsed_questions → generate_analysis_parallel
      (regex, ~0s)    (DB insert, ~1s)          (Claude, paralelo)

    Metas de tempo para 20 questões:
      - Insert:    < 2s
      - Análise:   < 30s (8 threads × Claude Haiku)
      - Total:     < 35s
    """
    if not parsed_questions:
        return []

    settings = get_settings()
    cb = get_client()
    t_start = time.time()
    total = len(parsed_questions)

    logger.info(
        f"[SaveParsed:{pdf_upload_id}] {total} questões pré-extraídas | "
        f"workers={settings.extraction_parallelism}"
    )

    if heartbeat_fn:
        heartbeat_fn(40, f"Salvando {total} questões extraídas")

    # ── Upsert de disciplinas ─────────────────────────────────────────────
    subject_cache: dict[str, str] = {}
    topics = list({q.topic for q in parsed_questions})
    for topic in topics:
        subject_cache[topic] = cb.upsert_subject(topic, study_plan_id)

    if heartbeat_fn:
        heartbeat_fn(50, "Extraindo imagens do PDF")

    # ── Extração de imagens (não-bloqueante) ──────────────────────────────
    # Para o smart parser, questões têm número explícito (q.number) —
    # o image_service usa esse mesmo número como chave do mapa.
    q_image_map: dict[int, list[dict]] = {}
    if pdf_bytes and user_id:
        try:
            from app.services.image_service import extract_question_images
            if heartbeat_fn:
                heartbeat_fn(52, "Extraindo e enviando imagens")
            q_image_map = extract_question_images(
                pdf_bytes=pdf_bytes,
                study_plan_id=study_plan_id,
                pdf_upload_id=pdf_upload_id,
                user_id=user_id,
                cb=cb,
            )
            logger.info(
                f"[SaveParsed:{pdf_upload_id}] imagens em {len(q_image_map)} questões"
            )
        except Exception as img_err:
            logger.warning(
                f"[SaveParsed:{pdf_upload_id}] extração de imagens falhou "
                f"(não bloqueante): {img_err}"
            )

    if heartbeat_fn:
        heartbeat_fn(55, f"Inserindo {total} questões no banco")

    # ── Batch insert de questões ──────────────────────────────────────────
    from app.services.image_service import has_image_reference

    questions_payload = []
    for q in parsed_questions:
        imgs       = q_image_map.get(q.number, [])
        difficulty = q.difficulty

        # Questão menciona imagem mas nenhuma foi encontrada → alerta visual
        if not imgs and has_image_reference(q.statement):
            difficulty = "image_missing"

        entry: dict = {
            "subject_id":    subject_cache.get(q.topic) or None,
            "statement":     q.statement,
            "alternatives":  q.alternatives,
            "correct_answer": q.correct_answer,
            "topic":         q.topic,
            "difficulty":    difficulty,
        }
        # Só inclui image_urls quando há imagens — evita erro SQL se a coluna
        # ainda não existir na tabela questions
        if imgs:
            entry["image_urls"] = imgs
        questions_payload.append(entry)

    logger.info(
        f"[SaveParsed:{pdf_upload_id}] payload amostra — "
        f"subject_id={questions_payload[0].get('subject_id')!r} "
        f"topic={questions_payload[0].get('topic')!r} "
        f"image_urls={questions_payload[0].get('image_urls')!r}"
    )
    question_ids = cb.insert_questions(questions_payload, study_plan_id, source_pdf_id)

    if not question_ids:
        logger.warning(f"[SaveParsed:{pdf_upload_id}] insert_questions não retornou IDs")
        return []

    t_insert = time.time() - t_start
    logger.info(
        f"[SaveParsed:{pdf_upload_id}] {len(question_ids)} questões inseridas em {t_insert:.1f}s"
    )

    if heartbeat_fn:
        heartbeat_fn(58, f"Gerando justificativas e peguinhas ({total} questões)")

    # ── Análise paralela: justificativas + peguinhas ──────────────────────
    questions_data = [
        {
            "statement": q.statement,
            "alternatives": q.alternatives,
            "correct_answer": q.correct_answer,
        }
        for q in parsed_questions
    ]

    all_justifications, all_tricky_points = generate_analysis_parallel(
        question_ids=question_ids,
        questions_data=questions_data,
        parallelism=settings.extraction_parallelism,
        heartbeat_fn=heartbeat_fn,
        heartbeat_base=58,
    )

    if heartbeat_fn:
        heartbeat_fn(93, "Salvando justificativas e peguinhas")

    if all_justifications:
        cb.insert_justifications(all_justifications)
    if all_tricky_points:
        cb.insert_tricky_points(all_tricky_points)

    t_total = time.time() - t_start
    logger.info(
        f"[SaveParsed:{pdf_upload_id}] FINISH — "
        f"{len(question_ids)} questões, {len(all_justifications)} justificativas, "
        f"{len(all_tricky_points)} peguinhas | total={t_total:.1f}s"
    )
    return question_ids
