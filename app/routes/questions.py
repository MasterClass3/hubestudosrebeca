from fastapi import APIRouter, HTTPException
from app.services.callback_service import get_client

router = APIRouter()


@router.get("/questions/{study_plan_id}")
def get_questions(study_plan_id: str):
    cb = get_client()
    questions = cb.get_questions(study_plan_id)
    return {"study_plan_id": study_plan_id, "count": len(questions), "questions": questions}


@router.get("/questions/{study_plan_id}/{question_id}/analysis")
def get_question_analysis(study_plan_id: str, question_id: str):
    cb = get_client()
    rows = cb.read("questions", {"id": question_id, "study_plan_id": study_plan_id})
    if not rows:
        raise HTTPException(status_code=404, detail="Questão não encontrada")

    return {
        "question": rows[0],
        "justifications": cb.read("justifications", {"question_id": question_id}),
        "tricky_points": cb.read("tricky_points", {"question_id": question_id}),
    }
