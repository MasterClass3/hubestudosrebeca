from fastapi import APIRouter, HTTPException

from app.services import db_client

router = APIRouter()


@router.get("/questions/{study_plan_id}")
def get_questions(study_plan_id: str):
    """Retorna todas as questões de um plano de estudo."""
    questions = db_client.read("questions", {"study_plan_id": study_plan_id})
    return {"study_plan_id": study_plan_id, "count": len(questions), "questions": questions}


@router.get("/questions/{study_plan_id}/{question_id}/analysis")
def get_question_analysis(study_plan_id: str, question_id: str):
    """Retorna justificativas e peguinhas de uma questão específica."""
    rows = db_client.read("questions", {"id": question_id, "study_plan_id": study_plan_id})
    if not rows:
        raise HTTPException(status_code=404, detail="Questão não encontrada")

    justifications = db_client.read("justifications", {"question_id": question_id})
    tricky_points = db_client.read("tricky_points", {"question_id": question_id})

    return {
        "question": rows[0],
        "justifications": justifications,
        "tricky_points": tricky_points,
    }
