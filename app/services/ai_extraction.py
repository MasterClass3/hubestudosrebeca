import json
import logging
import re
import textwrap
import time
from typing import Any, Callable
import anthropic

from app.config import get_settings
from app.services.callback_service import get_client

logger = logging.getLogger(__name__)

CHUNK_SIZE = 12_000
AI_CALL_TIMEOUT = 120.0  # segundos por chamada ao Claude

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


def _extract_from_chunk(
    client: anthropic.Anthropic,
    chunk: str,
    model: str,
    chunk_idx: int,
    total_chunks: int,
    pdf_upload_id: str,
) -> list[dict]:
    logger.info(f"[Extraction:{pdf_upload_id}] STEP_BEGIN chunk {chunk_idx+1}/{total_chunks} ({len(chunk)} chars)")
    t0 = time.time()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            timeout=AI_CALL_TIMEOUT,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT + chunk}],
        )
    except anthropic.APITimeoutError:
        logger.error(f"[Extraction:{pdf_upload_id}] STEP_ERROR chunk {chunk_idx+1} — timeout após {AI_CALL_TIMEOUT}s")
        raise RuntimeError(f"Timeout na chamada ao Claude (chunk {chunk_idx+1}/{total_chunks})")
    except anthropic.APIError as e:
        logger.error(f"[Extraction:{pdf_upload_id}] STEP_ERROR chunk {chunk_idx+1} — API error: {e}")
        raise RuntimeError(f"Erro na API Claude (chunk {chunk_idx+1}): {e}")

    raw = response.content[0].text if response.content else ""
    if not raw.strip():
        logger.warning(f"[Extraction:{pdf_upload_id}] chunk {chunk_idx+1} retornou resposta vazia")
        return []

    try:
        data = _parse_json(raw)
        questions = data.get("questions", [])
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"[Extraction:{pdf_upload_id}] STEP_ERROR chunk {chunk_idx+1} — JSON inválido: {e} | raw[:200]={raw[:200]}")
        return []

    elapsed = time.time() - t0
    logger.info(f"[Extraction:{pdf_upload_id}] STEP_SUCCESS chunk {chunk_idx+1} — {len(questions)} questões em {elapsed:.1f}s")
    return questions


def extract_and_save_questions(
    raw_text: str,
    study_plan_id: str,
    source_pdf_id: str,
    pdf_upload_id: str = "",
    heartbeat_fn: Callable[[int, str], None] | None = None,
) -> list[str]:
    """
    Extrai questões, justificativas e peguinhas do texto em uma única passagem de IA.
    Chama heartbeat_fn(progress, stage) entre chunks para manter o job vivo.
    """
    settings = get_settings()
    ai_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    cb = get_client()

    chunks = _chunk_text(raw_text)
    total = len(chunks)
    all_questions: list[dict[str, Any]] = []

    logger.info(f"[Extraction:{pdf_upload_id}] START — {total} chunks, {len(raw_text)} chars totais")

    for i, chunk in enumerate(chunks):
        # Heartbeat antes de cada chamada de IA
        if heartbeat_fn:
            progress = 35 + int((i / total) * 45)  # 35% → 80%
            heartbeat_fn(progress, f"Identificando questões ({i+1}/{total})")

        questions = _extract_from_chunk(ai_client, chunk, settings.ai_model, i, total, pdf_upload_id)
        all_questions.extend(questions)

    logger.info(f"[Extraction:{pdf_upload_id}] Extração concluída — {len(all_questions)} questões totais")

    if not all_questions:
        return []

    if heartbeat_fn:
        heartbeat_fn(82, "Salvando questões no banco")

    subject_cache: dict[str, str] = {}
    question_ids: list[str] = []

    for idx, q in enumerate(all_questions):
        topic = q.get("topic", "Geral")

        if topic not in subject_cache:
            subject_cache[topic] = cb.upsert_subject(topic, study_plan_id)

        ids = cb.insert_questions(
            [{
                "subject_id": subject_cache[topic],
                "statement": q.get("statement", ""),
                "alternatives": q.get("alternatives", []),
                "correct_answer": q.get("correct_answer", ""),
                "topic": topic,
                "difficulty": q.get("difficulty", "medium"),
            }],
            study_plan_id,
            source_pdf_id,
        )
        if not ids:
            continue
        question_id = ids[0]
        question_ids.append(question_id)

        justifications = [
            {
                "question_id": question_id,
                "alternative": j.get("alternative"),
                "is_correct": j.get("is_correct", False),
                "justification": j.get("justification", ""),
            }
            for j in q.get("justifications", [])
            if j.get("alternative") and j.get("justification")
        ]
        if justifications:
            cb.insert_justifications(justifications)

        tricky_points = [
            {
                "question_id": question_id,
                "description": tp.get("description", ""),
                "misleading_alternative": tp.get("misleading_alternative"),
                "deduction_tip": tp.get("deduction_tip", ""),
            }
            for tp in q.get("tricky_points", [])
            if tp.get("description")
        ]
        if tricky_points:
            cb.insert_tricky_points(tricky_points)

        # Heartbeat a cada 10 questões salvas
        if heartbeat_fn and (idx + 1) % 10 == 0:
            progress = 82 + int(((idx + 1) / len(all_questions)) * 8)
            heartbeat_fn(progress, f"Salvando questões ({idx+1}/{len(all_questions)})")

    logger.info(f"[Extraction:{pdf_upload_id}] FINISH — {len(question_ids)} questões salvas")
    return question_ids
