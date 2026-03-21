import json
import re
import textwrap
from typing import Any
import anthropic

from app.config import get_settings
from app.services.callback_service import get_client

# Cada chunk ~3000 tokens
CHUNK_SIZE = 12_000

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
    # Remove null bytes que o pdfplumber extrai de fontes especiais
    text = text.replace("\x00", " ").replace("\ufffd", "")
    if len(text) <= CHUNK_SIZE:
        return [text]
    return textwrap.wrap(text, CHUNK_SIZE, break_long_words=False, break_on_hyphens=False)


def _extract_from_chunk(client: anthropic.Anthropic, chunk: str, model: str) -> list[dict]:
    response = client.messages.create(
        model=model,
        max_tokens=8192,
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
    Extrai questões, justificativas e peguinhas do texto bruto em uma única passagem de IA.
    Retorna lista de IDs das questões criadas.
    """
    settings = get_settings()
    ai_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    cb = get_client()

    chunks = _chunk_text(raw_text)
    all_questions: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        questions = _extract_from_chunk(ai_client, chunk, settings.ai_model)
        all_questions.extend(questions)

    if not all_questions:
        return []

    subject_cache: dict[str, str] = {}
    question_ids: list[str] = []

    for q in all_questions:
        topic = q.get("topic", "Geral")

        # Busca ou cria disciplina
        if topic not in subject_cache:
            subject_cache[topic] = cb.upsert_subject(topic, study_plan_id)

        # Insere questão
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

        # Insere justificativas extraídas do PDF (não geradas por IA)
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

        # Insere peguinhas extraídas do PDF
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

    return question_ids
