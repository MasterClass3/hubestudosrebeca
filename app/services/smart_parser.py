"""
Smart Parser — detector e extrator genérico de provas de concurso estruturadas.

Detecta qualquer PDF de prova que contenha:
  - Numeração de questões  (1., 2., QUESTÃO 1, etc.)
  - Alternativas           ((A), (B), (C), (D), (E))
  - Gabarito no final      (tabela ou lista de respostas)

Vantagem: extrai questões via regex, sem nenhuma chamada de IA.
A IA entra apenas para justificativas e peguinhas.

Funciona com PDFs de qualquer banca ou plataforma — a detecção é
baseada em estrutura, não em marcas.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ExamMeta:
    ano: str = ""
    banca: str = ""
    orgao: str = ""
    prova: str = ""
    concurso_name: str = ""   # nome de exibição montado a partir dos metadados


@dataclass
class ParsedQuestion:
    number: int
    statement: str
    alternatives: list[dict] = field(default_factory=list)  # [{"letter":"A","text":"..."}]
    correct_answer: str = ""
    topic: str = "Geral"
    difficulty: str = "medium"


# ── Detecção de estrutura ─────────────────────────────────────────────────────

# Padrões que indicam presença de alternativas — sinal mais confiável
_ALT_SIGNATURE  = re.compile(r"\(A\)|\(B\)|\(C\)|\(D\)", re.IGNORECASE)  # (A) parênteses
_ALT_SIG_PLAIN  = re.compile(r"(?m)^[ \t]*[A-E]\)\s")                    # A) início de linha
_ALT_SIG_BARE   = re.compile(r"(?m)^[A-E](?:[ \t]|$)", re.MULTILINE)     # "A texto" ou "A\n" (letra solta)

# Padrões de numeração de questão
_QUESTION_NUM = re.compile(
    r"(?m)(?:^[ \t]*(?:Quest[aã]o|QUEST[AÃ]O)[ \t]+\d"   # "Questão 1"
    r"|^[ \t]*(?:Quest[aã]o|QUEST[AÃ]O)\s*$"              # "Questão" sózinha (null-byte PDF)
    r"|^[ \t]*\d{1,3}[.)]\s"                               # "1. " ou "1) "
    r"|^\(\d{1,3}\)\s"                                     # "(01) "
    r"|^[ \t]*\d{1,3}[ \t]+Q\d{5,}"                       # "1 Q3942431" (export)
    r"|^[ \t]*Enunciado\s*:)"                              # "Enunciado:" (apostilas)
)

# Gabarito ao final do documento
_GABARITO_SIGNATURE = re.compile(r"(?i)\b(?:gabarito|respostas)\b")


def is_structured_exam(text: str) -> bool:
    """
    Retorna True se o texto possui estrutura típica de prova de concurso.
    Detecta três formatos de alternativas:
      - (A) com parênteses    → padrão CESPE/FCC
      - A) início de linha    → padrão Vunesp/FGV
      - A texto (letra solta) → padrão de exportação (ex.: Qconcursos, AOCP)
    Funciona independentemente da origem (banca, plataforma, site).
    """
    # Remove null bytes antes de detectar (artefato de fontes customizadas)
    text = text.replace("\x00", " ")
    sample = text[:10000]   # 10 000 chars para cobrir PDFs com intro longa
    paren_hits = len(_ALT_SIGNATURE.findall(sample))
    plain_hits = len(_ALT_SIG_PLAIN.findall(sample))
    bare_hits  = len(_ALT_SIG_BARE.findall(sample))
    has_alternatives = max(paren_hits, plain_hits, bare_hits) >= 4
    has_numbering = len(_QUESTION_NUM.findall(sample)) >= 2
    return has_alternatives and has_numbering


# ── Extração de metadados ─────────────────────────────────────────────────────

def _extract_meta(text: str) -> ExamMeta:
    """
    Extrai Ano, Banca, Órgão e Prova de cabeçalhos padronizados.
    Formato reconhecido: "Ano: XXXX  Banca: XXX  Órgão: XXX  Prova: XXX"
    (campos na mesma linha ou em linhas separadas).
    """
    meta = ExamMeta()

    ano = re.search(r"Ano:\s*(\d{4})", text)
    if ano:
        meta.ano = ano.group(1).strip()

    banca = re.search(
        r"Banca:\s*(.+?)(?=\s+[OÓ]rg[ãa]o:|\s+Prova:|\n|$)", text
    )
    if banca:
        meta.banca = banca.group(1).strip()

    orgao = re.search(
        r"[OÓ]rg[ãa]o:\s*(.+?)(?=\s+Prova:|\s+Leia\b|\s+Assinale\b|\s+Julgue\b|\n|$)",
        text,
        re.IGNORECASE,
    )
    if orgao:
        meta.orgao = orgao.group(1).strip()

    prova = re.search(
        r"Prova:\s*(.+?)(?=\s+Leia\b|\s+Assinale\b|\s+Julgue\b|\n|$)",
        text,
    )
    if prova:
        meta.prova = prova.group(1).strip()

    meta.concurso_name = _build_display_name(meta)
    logger.info(
        f"[SmartParser] Metadados — banca='{meta.banca}' orgao='{meta.orgao}' "
        f"ano='{meta.ano}' → concurso_name='{meta.concurso_name}'"
    )
    return meta


def _build_display_name(meta: ExamMeta) -> str:
    """
    Monta o nome de exibição do concurso a partir dos metadados disponíveis.

    Formato preferencial (se campos disponíveis):
      "{Órgão} — {Cargo} ({Ano})"
      Ex: "Câmara de Goiânia - GO — Revisor de Texto (2026)"

    Quando vem do campo Prova com prefixo de banca:
      Remove o prefixo "Banca - Ano - " para exibir só o essencial.
    """
    name = meta.prova or meta.orgao
    if not name:
        return "Concurso Importado"

    # Remove prefixo da banca: ex. "IV - UFG - "
    if meta.banca:
        name = re.sub(
            r"^" + re.escape(meta.banca) + r"\s*[-–]\s*", "", name
        ).strip(" -–")

    # Remove prefixo do ano: ex. "2026 - "
    if meta.ano:
        name = re.sub(
            r"^" + re.escape(meta.ano) + r"\s*[-–]\s*", "", name
        ).strip(" -–")

    # Adiciona "(Ano)" ao final se não estiver presente
    if meta.ano and meta.ano not in name:
        name = f"{name} ({meta.ano})"

    return name.strip(" -–") or "Concurso Importado"


# ── Extração do Gabarito ──────────────────────────────────────────────────────

# Linha de gabarito: "01 - B", "01 B", "1. B", "01-B", "01) B", "01:B"
_GAB_LINE = re.compile(r"(?<!\d)(\d{1,3})\s*[-–.:)\s]\s*([A-Ea-e])\b")


def _extract_gabarito(text: str) -> dict[int, str]:
    """
    Localiza a seção de gabarito e retorna {numero_questao: letra_correta}.
    Busca no final do documento onde o gabarito normalmente aparece.
    """
    # Armazena a posição como int simples (elimina o hack _Pos)
    gab_start: int | None = None

    # Remove null bytes antes de detectar
    text = text.replace("\x00", " ")

    # Estratégia 1: "gabarito" ou "respostas" sozinho numa linha
    m = re.search(r"(?im)^\s*(?:gabarito|respostas)\s*$", text)
    if m:
        gab_start = m.start()

    # Estratégia 2: "Gabarito:" ou "Respostas:" inline
    if gab_start is None:
        m = re.search(r"(?i)\b(?:gabarito|respostas)\s*:", text)
        if m:
            gab_start = m.start()

    # Estratégia 3: qualquer "gabarito" / "respostas" nos últimos 4 000 chars
    if gab_start is None:
        tail = text[-4000:]
        m = re.search(r"(?i)\b(?:gabarito|respostas)\b", tail)
        if m:
            gab_start = max(0, len(text) - 4000) + m.start()

    if gab_start is None:
        logger.warning("[SmartParser] Gabarito não encontrado no documento")

    # ── Gabarito em bloco (seção ao final) ───────────────────────────────
    results: dict[int, str] = {}
    if gab_start is not None:
        gab_text = text[gab_start:]
        for m in _GAB_LINE.finditer(gab_text):
            num = int(m.group(1))
            if 1 <= num <= 500:
                results[num] = m.group(2).upper()

    # ── Gabarito inline: "RESPOSTA CORRETA: C" (apostilas/materiais) ─────
    # Extrai linha por linha para montar o gabarito sequencialmente
    inline_pat = re.compile(r"(?i)RESPOSTA[S]?\s+CORRETA[S]?\s*:\s*([A-Ea-e])")
    inline_matches = inline_pat.findall(text)
    if inline_matches and not results:
        for i, letter in enumerate(inline_matches, start=1):
            results[i] = letter.upper()
        logger.info(f"[SmartParser] Gabarito inline: {len(results)} respostas extraídas")

    logger.info(
        f"[SmartParser] Gabarito: {len(results)} entradas → "
        f"{dict(list(results.items())[:6])}..."
    )
    return results


# ── Extração de Questões ──────────────────────────────────────────────────────

# Formato "(A) texto" — parênteses ao redor da letra
_ALT_BLOCK_PAREN = re.compile(
    r"\(([A-Ea-e])\)\s*(.+?)(?=\s*\([A-Ea-e]\)|\s*[Gg]abarito|\Z)",
    re.DOTALL,
)

# Formato "A) texto" — sem parênteses, início de linha (precedido de espaço/tab opcional)
_ALT_BLOCK_PLAIN = re.compile(
    r"(?m)^[ \t]*([A-Ea-e])\)\s*(.+?)(?=^[ \t]*[A-Ea-e]\)|\s*[Gg]abarito|\Z)",
    re.DOTALL | re.MULTILINE,
)

# Formato "A texto" ou "A\n" — letra solta (com espaço, tab OU sozinha na linha)
_ALT_BLOCK_BARE = re.compile(
    r"(?m)^([A-E])(?:[ \t]+|\n)(.+?)(?=^[A-E](?:[ \t\n]|$)|\s*(?:[Gg]abarito|[Rr]espostas)|\Z)",
    re.DOTALL | re.MULTILINE,
)

# Q-ID genérico: sequências alfanuméricas que podem ser IDs de questão
_QID = re.compile(r"\b[Qq]\d{5,9}\b")

# Número de questão no formato "1 Q3942431" (exportação)
_QNUM_QID = re.compile(r"(?m)^[ \t]*(\d{1,3})[ \t]+Q\d{5,}")

# Número de questão no início: "1 ", "01 ", "1.", "01.", "1)"
_QNUM = re.compile(r"(?m)(?:^|\s)(\d{1,3})[.)]\s")

# Cabeçalhos que se repetem por página (remover antes de parsear)
_PAGE_HEADERS = re.compile(
    r"(?m)^.*(?:Ano:|Banca:|[OÓ]rg[ãa]o:|Prova:).*$", re.IGNORECASE
)


def _clean_text(text: str) -> str:
    """Remove cabeçalhos repetidos, IDs de questão e espaços excessivos."""
    # Remove null bytes (artefato de PDFs com fontes customizadas)
    text = text.replace("\x00", " ")
    # Remove linhas de cabeçalho de metadados (Ano:, Banca:, Órgão:, Prova:)
    text = _PAGE_HEADERS.sub("", text)
    # Remove IDs de questão (ex: Q3942431)
    text = _QID.sub("", text)
    # Remove linhas de categoria de questão (ex: "1  Português>Interpretação de Textos")
    text = re.sub(r"(?m)^\d{1,3}[ \t]+[^\n]*>[^\n]*$", "", text)
    # Remove URLs e rodapés de site (ex: "www.qconcursos.com")
    text = re.sub(r"(?m)^www\.\S+\s*$", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _infer_topic(statement: str) -> str:
    """Inferência básica de disciplina pelo enunciado."""
    stmt = statement.lower()
    if any(w in stmt for w in ["texto", "leitura", "trecho", "gramática", "morfologia",
                                "sintaxe", "ortografia", "pontuação", "redação", "língua",
                                "oração", "parágrafo", "vocábulo"]):
        return "Português"
    if any(w in stmt for w in ["constituição", "direito", "jurídico", "artigo",
                                "inciso", "parágrafo único", "lei nº", "decreto"]):
        return "Direito"
    if any(w in stmt for w in ["administração", "gestão", "planejamento", "orçamento",
                                "servidor", "cargo", "função pública"]):
        return "Administração"
    if any(w in stmt for w in ["informática", "computador", "software", "hardware",
                                "excel", "word", "internet", "rede", "sistema operacional"]):
        return "Informática"
    if any(w in stmt for w in ["calcule", "equação", "porcentagem", "razão", "proporção",
                                "probabilidade", "geometria", "álgebra"]):
        return "Matemática"
    return "Geral"


def _extract_questions(
    text: str, gabarito: dict[int, str]
) -> list[ParsedQuestion]:
    """
    Algoritmo principal de extração.

    Detecta automaticamente o formato das alternativas:
      - "(A) texto"  → formato com parênteses  (_ALT_BLOCK_PAREN)
      - "A) texto"   → formato sem parênteses, início de linha  (_ALT_BLOCK_PLAIN)

    Etapas:
      1. Limpa cabeçalhos e IDs
      2. Trunca na seção de gabarito
      3. Detecta formato predominante e escolhe regex correspondente
      4. Para cada "(A)" / "A)": extrai alternativas, recupera enunciado,
         determina número da questão e cruza com gabarito
    """
    clean = _clean_text(text)

    # Trunca antes do gabarito / respostas
    gab_idx = re.search(r"(?im)^\s*(?:gabarito|respostas)\s*", clean)
    if gab_idx:
        clean = clean[: gab_idx.start()]

    # ── Detecta formato predominante de alternativas ──────────────────────
    paren_hits = len(re.findall(r"\(A\)", clean))
    plain_hits = len(re.findall(r"(?m)^[ \t]*A\)\s", clean))
    # Bare: conta tanto "^A " quanto "^A\n" (letra sozinha na linha)
    bare_hits  = len(re.findall(r"(?m)^A(?:[ \t]|$)", clean))

    logger.info(
        f"[SmartParser] Detecção de formato — "
        f"(A): {paren_hits}×  |  A): {plain_hits}×  |  A bare: {bare_hits}× | "
        f"amostra: {repr(clean[:200])}"
    )

    if paren_hits >= plain_hits and paren_hits >= bare_hits and paren_hits >= 1:
        a_positions = [m.start() for m in re.finditer(r"\(A\)", clean)]
        alt_block_re = _ALT_BLOCK_PAREN
        last_pat = r"\(D\)|\(E\)"
        fmt = "paren"
    elif plain_hits >= bare_hits and plain_hits >= 1:
        a_positions = [m.start() for m in re.finditer(r"(?m)^[ \t]*A\)\s", clean)]
        alt_block_re = _ALT_BLOCK_PLAIN
        last_pat = r"(?m)^[ \t]*[DE]\)\s"
        fmt = "plain"
    elif bare_hits >= 1:
        # Captura "^A " e "^A\n" (letra sozinha na linha)
        a_positions = [m.start() for m in re.finditer(r"(?m)^A(?:[ \t]|$)", clean)]
        alt_block_re = _ALT_BLOCK_BARE
        last_pat = r"(?m)^[DE](?:[ \t]|$)"
        fmt = "bare"
    else:
        logger.warning(
            f"[SmartParser] Nenhum padrão de alternativa reconhecido — "
            f"amostra: {repr(clean[:400])}"
        )
        return []

    logger.info(
        f"[SmartParser] Formato '{fmt}' selecionado | "
        f"{len(a_positions)} blocos de alternativas encontrados"
    )

    questions: list[ParsedQuestion] = []
    q_counter = 0

    for i, a_pos in enumerate(a_positions):
        block_end = a_positions[i + 1] if i + 1 < len(a_positions) else len(clean)
        alt_block = clean[a_pos:block_end]

        # ── Extrai alternativas ───────────────────────────────────────────
        alternatives: list[dict] = []
        for m in alt_block_re.finditer(alt_block):
            letter = m.group(1).upper()
            alt_text = re.sub(r"\s+", " ", m.group(2)).strip().rstrip(".")
            if alt_text:
                alternatives.append({"letter": letter, "text": alt_text})

        if len(alternatives) < 2:
            continue

        # ── Recupera enunciado (busca para trás até o D/E anterior) ───────
        if i == 0:
            stmt_start = 0
        else:
            prev_end = -1
            for mf in re.finditer(last_pat, clean[:a_pos]):
                prev_end = mf.start()
            stmt_start = (prev_end + 3) if prev_end >= 0 else 0

        stmt_raw = clean[stmt_start:a_pos].strip()

        # ── Determina número da questão ───────────────────────────────────
        q_number = 0
        # Tenta formato "1 Q3942431" (exportação)
        qid_match = _QNUM_QID.search(stmt_raw)
        if qid_match:
            q_number = int(qid_match.group(1))
        else:
            num_matches = list(_QNUM.finditer(stmt_raw))
            if num_matches:
                q_number = int(num_matches[-1].group(1))

        if q_number == 0:
            q_counter += 1
            q_number = q_counter
        else:
            q_counter = q_number

        # ── Limpa enunciado ───────────────────────────────────────────────
        # Remove linha "N Q3942431 Disciplina>Categoria" (export format)
        stmt = re.sub(r"(?m)^\d{1,3}[ \t]+Q\d{5,}[^\n]*\n?", "", stmt_raw).strip()
        stmt = re.sub(r"^\d{1,3}[.)]\s*", "", stmt, count=1).strip()
        stmt = re.sub(r"[ \t]{2,}", " ", stmt).strip()
        stmt = re.sub(r"\n{3,}", "\n\n", stmt).strip()

        if len(stmt) < 15:
            continue

        questions.append(
            ParsedQuestion(
                number=q_number,
                statement=stmt,
                alternatives=alternatives,
                correct_answer=gabarito.get(q_number, ""),
                topic=_infer_topic(stmt),
                difficulty="medium",
            )
        )

    q_with_answer = sum(1 for q in questions if q.correct_answer)
    logger.info(
        f"[SmartParser] {len(questions)} questões extraídas "
        f"({q_with_answer} com gabarito, {len(questions) - q_with_answer} sem)"
    )
    return questions


# ── Extração dedicada para formato de exportação (com Q-ID) ──────────────────

# Linha de cabeçalho de questão: "1 Q3942431 Disciplina>..." ou "1 Q3942431"
_QID_HEADER = re.compile(r"(?m)^[ \t]*(\d{1,3})[ \t]+Q\d{5,}[^\n]*\n?")


def _extract_bare_questions(
    raw_text: str, gabarito: dict[int, str]
) -> list[ParsedQuestion]:
    """
    Extração dedicada para PDFs exportados com marcadores 'N Q12345'.

    Vantagem sobre o algoritmo genérico: usa os separadores de questão como
    fronteiras exatas, evitando contaminação entre enunciado e alternativas de
    questões adjacentes.
    """
    questions: list[ParsedQuestion] = []
    boundaries = list(_QID_HEADER.finditer(raw_text))
    if not boundaries:
        return questions

    for i, m in enumerate(boundaries):
        q_number = int(m.group(1))
        start = m.end()  # após a linha "N Q..."
        end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(raw_text)
        block = raw_text[start:end]

        # Remove cabeçalhos de metadados (Ano:, Banca:, Órgão:, Prova:)
        block = _PAGE_HEADERS.sub("", block)
        # Remove URLs
        block = re.sub(r"(?m)^https?://\S+\s*$", "", block)
        block = re.sub(r"(?m)^www\.\S+\s*$", "", block)
        # Remove seção de gabarito/respostas se aparecer dentro do bloco
        gab_m = re.search(r"(?im)^\s*(?:gabarito|respostas)\s*", block)
        if gab_m:
            block = block[: gab_m.start()]

        # Encontra o marcador real de alternativa A:
        # → o ÚLTIMO "^A" antes do PRIMEIRO "^B" (evita confundir com enunciado que começa por "A ")
        first_b = re.search(r"(?m)^B(?:[ \t]|$)", block)
        if not first_b:
            continue  # sem alternativa B → bloco inválido
        all_a = list(re.finditer(r"(?m)^A(?:[ \t]|$)", block))
        valid_a = [m for m in all_a if m.start() < first_b.start()]
        if not valid_a:
            continue
        a_match = valid_a[-1]

        stmt_raw = block[: a_match.start()].strip()
        alt_block = block[a_match.start():]

        # ── Extrai alternativas ───────────────────────────────────────────
        alternatives: list[dict] = []
        for am in _ALT_BLOCK_BARE.finditer(alt_block):
            letter = am.group(1).upper()
            alt_text = re.sub(r"\s+", " ", am.group(2)).strip().rstrip(".")
            if alt_text:
                alternatives.append({"letter": letter, "text": alt_text})

        if len(alternatives) < 2:
            continue

        # ── Limpa enunciado ───────────────────────────────────────────────
        stmt = re.sub(r"[ \t]{2,}", " ", stmt_raw)
        stmt = re.sub(r"\n{3,}", "\n\n", stmt).strip()
        if len(stmt) < 10:
            continue

        questions.append(
            ParsedQuestion(
                number=q_number,
                statement=stmt,
                alternatives=alternatives,
                correct_answer=gabarito.get(q_number, ""),
                topic=_infer_topic(stmt),
                difficulty="medium",
            )
        )

    q_with_answer = sum(1 for q in questions if q.correct_answer)
    logger.info(
        f"[SmartParser/bare] {len(questions)} questões extraídas "
        f"({q_with_answer} com gabarito)"
    )
    return questions


# ── API pública ───────────────────────────────────────────────────────────────

def parse_structured_exam(text: str) -> tuple[ExamMeta, list[ParsedQuestion]]:
    """
    Ponto de entrada principal do parser.

    Args:
        text: texto completo extraído do PDF

    Returns:
        (ExamMeta, list[ParsedQuestion])

    Notas:
        - ExamMeta.concurso_name é montado sem prefixos de plataformas externas
        - ParsedQuestion.correct_answer pode estar vazia se não houver gabarito
        - topic é inferência básica; a IA refina durante a análise
    """
    meta = _extract_meta(text)
    gabarito = _extract_gabarito(text)

    # Se o PDF tem marcadores "N Q12345" usa extração dedicada (mais precisa)
    if _QID_HEADER.search(text):
        logger.info("[SmartParser] Formato de exportação (Q-ID) detectado — usando _extract_bare_questions")
        questions = _extract_bare_questions(text, gabarito)
        # Fallback para extração genérica se não extraiu nenhuma questão
        if not questions:
            logger.warning("[SmartParser] _extract_bare_questions falhou — fallback para extração genérica")
            questions = _extract_questions(text, gabarito)
    else:
        questions = _extract_questions(text, gabarito)

    # ── Gate de qualidade ─────────────────────────────────────────────────
    # Se < 60% das questões têm 2+ alternativas válidas, a extração ficou ruim
    # (ex: PDF com alternativas em linha corrida ou layout incomum).
    # Retornar lista vazia força o pipeline a usar IA, que lida melhor.
    if questions:
        # Questões de concurso têm 4 ou 5 alternativas — exige pelo menos 4
        valid_ratio = sum(1 for q in questions if len(q.alternatives) >= 4) / len(questions)
        if valid_ratio < 0.60:
            logger.warning(
                f"[SmartParser] Qualidade baixa: apenas {valid_ratio:.0%} das questões têm "
                f"alternativas válidas — descartando para usar IA"
            )
            questions = []

    logger.info(
        f"[SmartParser] parse_structured_exam() → "
        f"concurso='{meta.concurso_name}' "
        f"questões={len(questions)} gabarito={len(gabarito)}"
    )
    return meta, questions
