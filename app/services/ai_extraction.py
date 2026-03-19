import json
import re
import textwrap
from typing import Any
import anthropic

from app.config import get_settings
from app.services.callback_service import get_client

CHUNK_SIZE = 12_000

EXTRACTION_PROMPT = """
Você é um especialista em concursos públicos brasileiros. Analise o texto abaixo e extraia TODAS as questões encontradas.

Regras:
- Detecte questões CESPE (Certo/Errado) e múltipla escolha (A a E)
- Para CESPE, use alternatives: [{"letter": "C", "text": "Certo"}, {"letter": "E", "text": "Errado"}]
- Classifique a disciplina: Português, Matemática, Raciocínio Lógico, Direito Constitucional, Direito Administrativo, Direito Penal, Direito Processual, Informática, Administração Pública, Legislação Específica, ou outra
- Estime a dificuldade como: easy, medium ou hard

Retorne SOMENTE um JSON válido no formato:
{
  "questions": [
    {
      "statement": "texto completo do enunciado",
      "alternatives": [{"letter": "A", "text": "texto da alternativa"}, ...],
      "correct_answer": "B",
      "topic": "nome da disciplina/tema",
      "difficulty": "medium"
    }
  ]
}

Se não houver questões no texto, retorne {"questions": []}.
TEXTO:
"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1).strip()
    return json.loads(raw)


def _chunk_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    return textwrap.wrap(text, CHUNK_SIZE, break_long_words=False, break_on_hyphens=False)


def _extract_questions_from_chunk(client: anthropic.Anthropic, chunk: str, model: str) -> list[dict]:
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT + chunk}],
    )
    raw = response.content[0].text if response.content else "{}"
    data = _parse_json(raw)
    return data.get("questions", [])


def extract_and_save_questions(
    raw_text: str,
    study_plan_id: str,
    source_pdf_id: str,
) -> list[str]:
    """
    Extrai questões do texto bruto via IA e salva via Edge Function.
    Retorna lista de IDs das questões criadas.
    """
    settings = get_settings()
    ai_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    cb = get_client()

    chunks = _chunk_text(raw_text)
    all_questions: list[dict[str, Any]] = []

    for chunk in chunks:
        questions = _extract_questions_from_chunk(ai_client, chunk, settings.ai_model)
        all_questions.extend(questions)

    if not all_questions:
        return []

    subject_cache: dict[str, str] = {}
    questions_payload = []

    for q in all_questions:
        topic = q.get("topic", "Geral")
        if topic not in subject_cache:
            subject_cache[topic] = cb.upsert_subject(topic, study_plan_id)

        questions_payload.append({
            "subject_id": subject_cache[topic],
            "statement": q.get("statement", ""),
            "alternatives": q.get("alternatives", []),
            "correct_answer": q.get("correct_answer", ""),
            "topic": topic,
            "difficulty": q.get("difficulty", "medium"),
        })

    return cb.insert_questions(questions_payload, study_plan_id, source_pdf_id)
