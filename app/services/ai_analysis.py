import json
import re
import time
import anthropic

from app.config import get_settings
from app.services.callback_service import get_client

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
    alternatives_text = ", ".join(
        f"{a['letter']}) {a['text']}" for a in question.get("alternatives", [])
    )
    prompt = ANALYSIS_PROMPT.format(
        statement=question["statement"],
        alternatives=alternatives_text,
        correct_answer=question["correct_answer"],
    )
    response = ai_client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text if response.content else "{}"
    return _parse_json(raw)


def generate_analysis_for_questions(question_ids: list[str]) -> dict:
    """Gera justificativas e peguinhas para uma lista de questões (por ID)."""
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
