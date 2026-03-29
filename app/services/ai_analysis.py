import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import anthropic

from app.config import get_settings
from app.services.callback_service import get_client

logger = logging.getLogger(__name__)

# Rate limit para a função sequencial legada
_RATE_LIMIT_INTERVAL = 12.0
_last_call_time: float = 0.0

ANALYSIS_PROMPT = """
Você é um professor especialista em concursos públicos brasileiros.
Analise a questão abaixo e retorne um JSON com justificativas e peguinhas.

QUESTÃO:
Enunciado: {statement}
Alternativas: {alternatives}
Gabarito: {correct_answer}

Retorne SOMENTE um JSON válido no formato:
{{
  "justifications": [
    {{
      "alternative": "A",
      "is_correct": false,
      "justification": "Explicação didática de por que está errada..."
    }}
  ],
  "tricky_points": [
    {{
      "description": "Descrição da armadilha",
      "misleading_alternative": "A",
      "deduction_tip": "Como eliminar essa alternativa sem saber o conteúdo..."
    }}
  ]
}}

Dicas:
- Identifique palavras-armadilha: sempre, nunca, somente, exclusivamente, obrigatoriamente, apenas
- Explique por que cada alternativa errada parece correta à primeira vista
- Forneça dicas de dedução lógica para eliminar alternativas
"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1).strip()
    return json.loads(raw)


def _analyze_question(ai_client: anthropic.Anthropic, model: str, question: dict) -> dict:
    """Gera justificativas e peguinhas para UMA questão via Claude."""
    alternatives_text = ", ".join(
        f"{a.get('letter', a.get('letra', '?'))}) "
        f"{a.get('text', a.get('texto', ''))}"
        for a in question.get("alternatives", [])
    )
    prompt = ANALYSIS_PROMPT.format(
        statement=question.get("statement", ""),
        alternatives=alternatives_text,
        correct_answer=question.get("correct_answer", ""),
    )
    response = ai_client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text if response.content else "{}"
    return _parse_json(raw)


# ── Versão paralela ───────────────────────────────────────────────────────────

def generate_analysis_parallel(
    question_ids: list[str],
    questions_data: list[dict],
    parallelism: int = 8,
    heartbeat_fn: Callable[[int, str], None] | None = None,
    heartbeat_base: int = 60,
) -> tuple[list[dict], list[dict]]:
    """
    Gera justificativas e peguinhas em paralelo para uma lista de questões.

    Args:
        question_ids:   lista de IDs já inseridos no banco
        questions_data: lista de dicts com statement/alternatives/correct_answer
                        (mesma ordem de question_ids)
        parallelism:    número de threads simultâneas (padrão: 8)
        heartbeat_fn:   callback(progress, stage) opcional
        heartbeat_base: progresso inicial para o heartbeat (%)

    Returns:
        (all_justifications, all_tricky_points) — prontos para batch insert
    """
    settings = get_settings()
    total = len(question_ids)

    all_justifications: list[dict] = []
    all_tricky_points: list[dict] = []
    analyzed_count = 0
    lock = threading.Lock()

    def _analyze_one(q_id: str, q_data: dict, idx: int) -> None:
        nonlocal analyzed_count
        try:
            ai_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            analysis = _analyze_question(ai_client, settings.ai_extraction_model, q_data)

            justs = [
                {
                    "question_id": q_id,
                    "alternative": j.get("alternative"),
                    "is_correct": j.get("is_correct", False),
                    "justification": j.get("justification", ""),
                }
                for j in analysis.get("justifications", [])
                if j.get("alternative") and j.get("justification")
            ]
            tricks = [
                {
                    "question_id": q_id,
                    "description": tp.get("description", ""),
                    "misleading_alternative": tp.get("misleading_alternative"),
                    "deduction_tip": tp.get("deduction_tip", ""),
                }
                for tp in analysis.get("tricky_points", [])
                if tp.get("description")
            ]

            with lock:
                all_justifications.extend(justs)
                all_tricky_points.extend(tricks)
                analyzed_count += 1
                done = analyzed_count

            if heartbeat_fn and (done % 5 == 0 or done == total):
                progress = heartbeat_base + int((done / total) * (95 - heartbeat_base))
                heartbeat_fn(progress, f"Analisando questão {done}/{total}")

        except Exception as e:
            logger.error(f"[Analysis] análise falhou q_id={q_id}: {e}")
            with lock:
                analyzed_count += 1

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [
            executor.submit(_analyze_one, q_id, q_data, i)
            for i, (q_id, q_data) in enumerate(zip(question_ids, questions_data))
        ]
        for f in as_completed(futures):
            f.result()  # exceções já capturadas internamente

    logger.info(
        f"[Analysis] paralela concluída — "
        f"{len(all_justifications)} justificativas, {len(all_tricky_points)} peguinhas "
        f"para {total} questões"
    )
    return all_justifications, all_tricky_points


# ── Versão sequencial legada (mantida para compatibilidade) ──────────────────

def generate_analysis_for_questions(question_ids: list[str]) -> dict:
    """Gera justificativas e peguinhas para uma lista de questões (por ID). Sequencial."""
    settings = get_settings()
    ai_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    cb = get_client()

    results = {"processed": 0, "errors": 0, "question_ids": question_ids}

    for question_id in question_ids:
        try:
            rows = cb.read("questions", {"id": question_id})
            if not rows:
                results["errors"] += 1
                continue
            question = rows[0]

            global _last_call_time
            elapsed = time.time() - _last_call_time
            if elapsed < _RATE_LIMIT_INTERVAL:
                time.sleep(_RATE_LIMIT_INTERVAL - elapsed)

            analysis = _analyze_question(ai_client, settings.ai_model, question)
            _last_call_time = time.time()

            justifications = [
                {
                    "question_id": question_id,
                    "alternative": j.get("alternative"),
                    "is_correct": j.get("is_correct", False),
                    "justification": j.get("justification", ""),
                }
                for j in analysis.get("justifications", [])
            ]
            if justifications:
                cb.insert_justifications(justifications)

            tricky_points = [
                {
                    "question_id": question_id,
                    "description": tp.get("description", ""),
                    "misleading_alternative": tp.get("misleading_alternative"),
                    "deduction_tip": tp.get("deduction_tip"),
                }
                for tp in analysis.get("tricky_points", [])
            ]
            if tricky_points:
                cb.insert_tricky_points(tricky_points)

            results["processed"] += 1

        except Exception as e:
            results["errors"] += 1
            results.setdefault("error_details", []).append(
                {"question_id": question_id, "error": str(e)}
            )

    return results
