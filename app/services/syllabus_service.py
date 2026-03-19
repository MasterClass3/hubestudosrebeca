import json
import re
import textwrap
import anthropic

from app.config import get_settings
from app.services import db_client

CHUNK_SIZE = 12_000

SYLLABUS_PROMPT = """
Você é um especialista em concursos públicos brasileiros.
Analise o texto de edital/conteúdo programático abaixo e estruture em disciplinas e tópicos hierárquicos.

Retorne SOMENTE um JSON válido no formato:
{
  "subjects": [
    {
      "subject_name": "Português",
      "topics": [
        {
          "topic_title": "Interpretação de Texto",
          "order_index": 1,
          "subtopics": []
        },
        {
          "topic_title": "Gramática",
          "order_index": 2,
          "subtopics": [
            {"topic_title": "Concordância Verbal", "order_index": 1, "subtopics": []},
            {"topic_title": "Regência", "order_index": 2, "subtopics": []}
          ]
        }
      ]
    }
  ]
}

Regras:
- Identifique nomes de disciplinas em maiúsculas ou com numeração romana
- Preserve a hierarquia original (tópicos e subtópicos)
- Se houver numeração (1.1, 1.2), use para determinar order_index
- Se não houver hierarquia clara, todos os tópicos ficam no nível raiz

TEXTO DO EDITAL:
"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1).strip()
    return json.loads(raw)


def _flatten_topics(
    study_plan_id: str,
    subject_name: str,
    topics: list[dict],
    parent_id: str | None = None,
) -> list[dict]:
    """Achata a hierarquia de tópicos para inserção em batch."""
    flat = []
    for topic in topics:
        flat.append({
            "study_plan_id": study_plan_id,
            "subject_name": subject_name,
            "topic_title": topic["topic_title"],
            "parent_topic_id": parent_id,
            "order_index": topic.get("order_index", 0),
            "is_completed": False,
            "_subtopics": topic.get("subtopics", []),
        })
    return flat


def _insert_topics_recursive(
    study_plan_id: str,
    subject_name: str,
    topics: list[dict],
    parent_id: str | None = None,
):
    """Insere tópicos recursivamente via Edge Function."""
    for topic in topics:
        subtopics = topic.pop("_subtopics", topic.get("subtopics", []))
        payload = {
            "study_plan_id": study_plan_id,
            "subject_name": subject_name,
            "topic_title": topic["topic_title"],
            "parent_topic_id": parent_id,
            "order_index": topic.get("order_index", 0),
            "is_completed": False,
        }
        result = db_client.insert_syllabus_topics([payload])
        # Tenta obter o ID do tópico inserido para usar como parent_id nos subtópicos
        new_id = None
        if isinstance(result, dict):
            data = result.get("data", [])
            if data:
                new_id = data[0].get("id")

        if subtopics:
            _insert_topics_recursive(study_plan_id, subject_name, subtopics, new_id)


def extract_and_save_syllabus(text: str, study_plan_id: str) -> dict:
    """
    Extrai conteúdo programático do texto e salva via Edge Function.
    """
    settings = get_settings()
    ai_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    chunks = (
        [text]
        if len(text) <= CHUNK_SIZE
        else textwrap.wrap(text, CHUNK_SIZE, break_long_words=False, break_on_hyphens=False)
    )

    all_subjects: list[dict] = []
    for chunk in chunks:
        response = ai_client.messages.create(
            model=settings.ai_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": SYLLABUS_PROMPT + chunk}],
        )
        raw = response.content[0].text if response.content else "{}"
        data = _parse_json(raw)
        all_subjects.extend(data.get("subjects", []))

    subject_count = 0
    topic_count = 0

    for subject in all_subjects:
        subject_name = subject.get("subject_name", "Sem disciplina")
        topics = subject.get("topics", [])
        _insert_topics_recursive(study_plan_id, subject_name, topics)
        subject_count += 1
        topic_count += len(topics)

    return {
        "subjects_created": subject_count,
        "top_level_topics_created": topic_count,
        "study_plan_id": study_plan_id,
    }
