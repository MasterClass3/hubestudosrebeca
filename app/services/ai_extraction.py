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

logger = logging.getLogger(__name__)

CHUNK_SIZE = 12_000
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


def _chunk_text(text: str) -> list[str]:
    text = text.replace("\x00", " ").replace("\ufffd", "")
    if len(text) <= CHUNK_SIZE:
        return [text]
    return textwrap.wrap(text, CHUNK_SIZE, break_long_words=False, break_on_hyphens=False)


def _extract_chunk(
    api_key: str,
    model: str,
    chunk: str,
    chunk_idx: int,
    total_chunks: int,
    pdf_upload_id: str,
) -> tuple[int, list[dict]]:
    """Processa um chunk em thread separada. Retorna (chunk_idx, questions)."""
    logger.info(f"[Extraction:{pdf_upload_id}] chunk {chunk_idx+1}/{total_chunks} iniciando ({len(chunk)} chars)")
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
        raise RuntimeError(f"Timeout no chunk {chunk_idx+1}/{total_chunks}")
    except anthropic.APIError as e:
        raise RuntimeError(f"Erro API Claude chunk {chunk_idx+1}: {e}")

    raw = response.content[0].text if response.content else ""
    if not raw.strip():
        logger.warning(f"[Extraction:{pdf_upload_id}] chunk {chunk_idx+1} retornou vazio")
        return chunk_idx, []

    try:
        questions = _parse_json(raw).get("questions", [])
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"[Extraction:{pdf_upload_id}] chunk {chunk_idx+1} JSON inválido: {e} | raw[:100]={raw[:100]}")
        return chunk_idx, []

    elapsed = time.time() - t0
    logger.info(f"[Extraction:{pdf_upload_id}] chunk {chunk_idx+1} OK — {len(questions)} questões em {elapsed:.1f}s")
    return chunk_idx, questions


def extract_and_save_questions(
    raw_text: str,
    study_plan_id: str,
    source_pdf_id: str,
    pdf_upload_id: str = "",
    heartbeat_fn: Callable[[int, str], None] | None = None,
) -> list[str]:
    """
    Extrai questões em paralelo (N chunks simultâneos) e faz batch insert.
    Tempo típico: 100 questões em ~60-90s vs ~10min sequencial.
    """
    settings = get_settings()
    cb = get_client()
    chunks = _chunk_text(raw_text)
    total = len(chunks)

    logger.info(f"[Extraction:{pdf_upload_id}] START — {total} chunks | modelo={settings.ai_extraction_model} | workers={settings.extraction_parallelism}")

    # ── Paralelizar chamadas ao Claude ──────────────────────────────────
    completed_count = 0
    lock = threading.Lock()
    results_by_idx: dict[int, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=settings.extraction_parallelism) as executor:
        futures = {
            executor.submit(
                _extract_chunk,
                settings.anthropic_api_key,
                settings.ai_extraction_model,
                chunk,
                i,
                total,
                pdf_upload_id,
            ): i
            for i, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            try:
                idx, questions = future.result()
                results_by_idx[idx] = questions
            except Exception as e:
                idx = futures[future]
                logger.error(f"[Extraction:{pdf_upload_id}] chunk {idx+1} falhou: {e}")
                results_by_idx[idx] = []

            with lock:
                completed_count += 1
                done = completed_count

            if heartbeat_fn:
                progress = 35 + int((done / total) * 47)  # 35% → 82%
                heartbeat_fn(progress, f"Identificando questões ({done}/{total})")

    # Reagrupa na ordem original dos chunks
    all_questions: list[dict[str, Any]] = []
    for i in range(total):
        all_questions.extend(results_by_idx.get(i, []))

    logger.info(f"[Extraction:{pdf_upload_id}] Extração concluída — {len(all_questions)} questões em {total} chunks")

    if not all_questions:
        return []

    if heartbeat_fn:
        heartbeat_fn(83, "Criando disciplinas")

    # ── Upsert de disciplinas (sem duplicata) ───────────────────────────
    subject_cache: dict[str, str] = {}
    topics = list({q.get("topic", "Geral") for q in all_questions})
    for topic in topics:
        subject_cache[topic] = cb.upsert_subject(topic, study_plan_id)

    if heartbeat_fn:
        heartbeat_fn(86, "Salvando questões no banco")

    # ── Batch insert de questões ────────────────────────────────────────
    questions_payload = [
        {
            "subject_id": subject_cache.get(q.get("topic", "Geral")),
            "statement": q.get("statement", ""),
            "alternatives": q.get("alternatives", []),
            "correct_answer": q.get("correct_answer", ""),
            "topic": q.get("topic", "Geral"),
            "difficulty": q.get("difficulty", "medium"),
        }
        for q in all_questions
    ]
    question_ids = cb.insert_questions(questions_payload, study_plan_id, source_pdf_id)

    if not question_ids:
        logger.warning(f"[Extraction:{pdf_upload_id}] insert_questions não retornou IDs")
        return []

    if heartbeat_fn:
        heartbeat_fn(90, "Salvando justificativas")

    # ── Batch insert de justificativas ──────────────────────────────────
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

    logger.info(
        f"[Extraction:{pdf_upload_id}] FINISH — "
        f"{len(question_ids)} questões, {len(all_justifications)} justificativas, {len(all_tricky_points)} peguinhas"
    )
    return question_ids
