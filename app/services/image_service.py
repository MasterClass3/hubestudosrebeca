"""
image_service.py — Extração de imagens de PDFs de concurso.

Fluxo:
  1. Abre o PDF com PyMuPDF (fitz) — mais preciso que pdfplumber para imagens
  2. Detecta posições de questões por busca de padrões de numeração no texto
  3. Extrai imagens embutidas, descartando ícones/elementos decorativos (<80×60px)
  4. Associa cada imagem à questão mais próxima acima (mesma página, por posição Y)
  5. Faz upload para o bucket 'pdfs' do Supabase via signed URL da Edge Function
  6. Retorna {numero_questao: [{"url": caminho, "order": idx, "description": ""}]}

Tudo é não-bloqueante — falhas de extração ou upload são logadas e ignoradas,
permitindo que o pipeline continue sem imagens.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from app.services.callback_service import SupabaseCallbackClient

logger = logging.getLogger(__name__)

# Descarta imagens menores que isso (ícones, marcas d'água, elementos decorativos)
_MIN_W = 80
_MIN_H = 60

# Palavras que indicam que o enunciado referencia um elemento visual
_IMAGE_REF_WORDS = [
    "figura", "tabela", "quadro", "imagem", "gráfico", "grafico",
    "ilustração", "ilustracao", "diagrama", "mapa", "charge",
    "observe", "conforme", "veja a", "analise a", "com base na",
    "de acordo com o gráfico", "de acordo com a tabela",
    "de acordo com a figura", "segundo a tabela", "segundo o gráfico",
]

# Padrões de numeração de questão (limpos de null bytes antes de testar)
_Q_NUM_PATTERNS = [
    re.compile(r"(?:Quest[aã]o|QUEST[AÃ]O)\s+(\d{1,3})"),  # "Questão 12"
    re.compile(r"^[ \t]*(\d{1,3})[.)]\s"),                   # "12. " ou "12) " início de linha
    re.compile(r"^\((\d{1,3})\)\s"),                         # "(12) " início de linha
    re.compile(r"^[ \t]*(\d{1,3})[ \t]+Q\d{5,}"),           # "1 Q3942431" (Qconcursos)
]

# Tipos MIME por extensão
_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class _RawImage:
    xref: int
    page_num: int
    y0: float       # posição Y do topo da imagem na página (pontos PDF)
    image_bytes: bytes
    ext: str        # png, jpg, etc.
    width: int
    height: int


# ── API pública ───────────────────────────────────────────────────────────────

def extract_question_images(
    pdf_bytes: bytes,
    study_plan_id: str,
    pdf_upload_id: str,
    user_id: str,
    cb: "SupabaseCallbackClient",
    parallelism: int = 4,
) -> dict[int, list[dict]]:
    """
    Extrai imagens do PDF e faz upload para o Supabase Storage.

    Args:
        pdf_bytes:      conteúdo bruto do PDF
        study_plan_id:  ID do plano de estudo (usado no caminho do arquivo)
        pdf_upload_id:  ID do upload (para logs)
        user_id:        ID do usuário (primeiro segmento do file_path)
        cb:             cliente da Edge Function (para signed upload URLs)
        parallelism:    threads de upload em paralelo

    Returns:
        {numero_questao: [{"url": caminho, "order": idx, "description": ""}]}
        Retorna {} silenciosamente em caso de erro (não-bloqueante).
    """
    tag = f"[ImageService:{pdf_upload_id}]"

    if not pdf_bytes:
        logger.info(f"{tag} pdf_bytes vazio — extração de imagens ignorada")
        return {}

    if not user_id:
        logger.info(f"{tag} user_id vazio — extração de imagens ignorada")
        return {}

    logger.info(f"{tag} iniciando — {len(pdf_bytes)} bytes | study_plan={study_plan_id}")

    try:
        import fitz  # PyMuPDF
        logger.info(f"{tag} PyMuPDF versão {fitz.version}")
    except ImportError:
        logger.warning(
            f"{tag} pymupdf não instalado — extração de imagens desabilitada. "
            "Instale com: pip install pymupdf"
        )
        return {}

    # ── Passo 1: abrir PDF e extrair posições das questões ────────────────
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        logger.info(f"{tag} PDF aberto — {len(doc)} páginas")
    except Exception as e:
        logger.error(f"{tag} Falha ao abrir PDF com fitz: {e}")
        return {}

    try:
        # ── Passo 1: posições das questões ────────────────────────────────
        q_positions = _find_question_positions(doc)
        total_markers = sum(len(v) for v in q_positions.values())
        logger.info(
            f"{tag} Passo 1 — {total_markers} marcadores de questão "
            f"em {len(q_positions)} páginas: "
            f"{dict(list(q_positions.items())[:3])}"
        )

        # ── Passo 2: extrair imagens com posições ─────────────────────────
        raw_images = _extract_raw_images(doc)
        logger.info(
            f"{tag} Passo 2 — {len(raw_images)} imagens extraídas "
            f"(filtro ≥{_MIN_W}×{_MIN_H}px)"
        )

        if not raw_images:
            # Conta o total bruto antes do filtro para diagnóstico
            total_raw = sum(len(doc[p].get_images(full=True)) for p in range(len(doc)))
            logger.info(
                f"{tag} Nenhuma imagem após filtro de tamanho. "
                f"Total bruto no PDF (sem filtro): {total_raw}. "
                f"Provável causa: tabelas são texto formatado, não imagens embutidas."
            )
            doc.close()
            return {}

        # ── Guarda-chuva: sem posições de questão → não desperdiçar uploads ─
        if not q_positions:
            logger.warning(
                f"{tag} Posições de questão não detectadas — as {len(raw_images)} imagens "
                f"encontradas são provavelmente logos/cabeçalhos decorativos. "
                f"Imagens não serão associadas a questões."
            )
            doc.close()
            return {}

        # ── Passo 3: associar imagens às questões ─────────────────────────
        q_images: dict[int, list[_RawImage]] = {}
        for img in raw_images:
            q_num = _assign_to_question(img.page_num, img.y0, q_positions)
            if q_num == 0:
                logger.debug(
                    f"{tag} Imagem xref={img.xref} p{img.page_num} y0={img.y0:.0f} "
                    f"→ q_num=0 (não associada a nenhuma questão) — ignorada"
                )
                continue   # descarta imagens que não foram associadas a nenhuma questão
            q_images.setdefault(q_num, []).append(img)
            logger.debug(
                f"{tag} Imagem xref={img.xref} p{img.page_num} "
                f"y0={img.y0:.0f} → questão {q_num} ({img.width}×{img.height}px .{img.ext})"
            )

        logger.info(
            f"{tag} Passo 3 — {len(raw_images)} imagens, "
            f"{sum(len(v) for v in q_images.values())} associadas a "
            f"{len(q_images)} questões: {list(q_images.keys())}"
        )

        doc.close()
    except Exception as e:
        logger.error(f"{tag} Erro durante extração: {e}", exc_info=True)
        try:
            doc.close()
        except Exception:
            pass
        return {}

    # ── Passo 4: upload em paralelo ───────────────────────────────────────
    # Monta lista de tarefas: (q_num, order_idx, image, storage_path)
    upload_tasks: list[tuple[int, int, _RawImage, str]] = []
    for q_num, images in q_images.items():
        for idx, img in enumerate(images):
            path = (
                f"{user_id}/{study_plan_id}/images/"
                f"q{q_num}_{idx}.{img.ext}"
            )
            upload_tasks.append((q_num, idx, img, path))

    if not upload_tasks:
        return {}

    result: dict[int, list[dict]] = {}
    result_lock = __import__("threading").Lock()

    def _upload_one(task: tuple) -> None:
        q_num, idx, img, path = task
        try:
            _upload_image(cb, path, img.image_bytes, img.ext)
            logger.info(f"{tag} Upload OK → q={q_num} idx={idx} path={path} ({len(img.image_bytes)} bytes)")
            with result_lock:
                result.setdefault(q_num, []).append({
                    "url": path,
                    "order": idx,
                    "description": "",
                })
        except Exception as up_err:
            logger.warning(
                f"{tag} Upload FALHOU q={q_num} idx={idx} path={path}: {up_err}"
            )

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [executor.submit(_upload_one, task) for task in upload_tasks]
        for f in as_completed(futures):
            f.result()  # exceções já capturadas internamente

    # Ordena por 'order' dentro de cada questão
    for q_num in result:
        result[q_num].sort(key=lambda x: x["order"])

    total_uploaded = sum(len(v) for v in result.values())
    logger.info(
        f"[ImageService:{pdf_upload_id}] "
        f"{total_uploaded} imagens enviadas para {len(result)} questões"
    )
    return result


def has_image_reference(statement: str) -> bool:
    """
    Retorna True se o enunciado menciona figura, tabela, gráfico etc.
    Usado para marcar questões sem imagem com difficulty='image_missing'.
    """
    stmt = statement.lower()
    return any(word in stmt for word in _IMAGE_REF_WORDS)


# ── Detecção de posições das questões ─────────────────────────────────────────

def _find_question_positions(doc) -> dict[int, list[tuple[float, int]]]:
    """
    Percorre cada página e localiza onde cada questão começa (pelo número).

    Limpa null bytes (\x00) antes de testar padrões — comum em PDFs do Qconcursos
    onde o número da questão é renderizado com fonte especial (ex: "Questão\x001").

    Returns:
        {page_num: [(y0, q_num), ...]}  — ordenado por y0 dentro de cada página
    """
    positions: dict[int, list[tuple[float, int]]] = {}
    sample_lines: list[str] = []   # primeiras linhas para diagnóstico

    for page_num in range(len(doc)):
        page = doc[page_num]
        try:
            text_dict = page.get_text("dict")
        except Exception:
            continue

        seen_q_on_page: set[int] = set()

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:     # 0 = bloco de texto
                continue
            for line in block.get("lines", []):
                raw_text = " ".join(
                    span.get("text", "") for span in line.get("spans", [])
                )
                # Limpa null bytes — artefato de fontes customizadas em PDFs de concurso
                line_text = raw_text.replace("\x00", " ").replace("\ufffd", "").strip()

                if len(sample_lines) < 20 and line_text:
                    sample_lines.append(f"p{page_num}:{line_text[:80]}")

                for pat in _Q_NUM_PATTERNS:
                    m = pat.search(line_text)
                    if m:
                        try:
                            q_num = int(m.group(1))
                        except ValueError:
                            continue
                        if not (1 <= q_num <= 500):
                            continue
                        if q_num in seen_q_on_page:
                            continue          # já registrado nesta página
                        seen_q_on_page.add(q_num)
                        y0 = line["bbox"][1]  # topo da linha
                        positions.setdefault(page_num, []).append((y0, q_num))
                        break

    # Ordena por posição Y dentro de cada página
    for page_num in positions:
        positions[page_num].sort(key=lambda t: t[0])

    if not positions:
        logger.info(
            f"[ImageService] _find_question_positions: nenhum marcador encontrado. "
            f"Amostra das primeiras linhas do PDF: {sample_lines[:10]}"
        )

    return positions


# ── Extração de imagens ───────────────────────────────────────────────────────

def _extract_raw_images(doc) -> list[_RawImage]:
    """
    Extrai todas as imagens do documento, descartando as pequenas.
    Evita duplicatas pelo xref (mesma imagem que aparece em múltiplas páginas).
    """
    images: list[_RawImage] = []
    seen_xrefs: set[int] = set()

    for page_num in range(len(doc)):
        page = doc[page_num]
        try:
            image_list = page.get_images(full=True)
        except Exception:
            continue

        for img_info in image_list:
            xref   = img_info[0]
            width  = img_info[2]
            height = img_info[3]

            if xref in seen_xrefs:
                continue
            if width < _MIN_W or height < _MIN_H:
                continue

            try:
                base_image   = doc.extract_image(xref)
                image_bytes  = base_image["image"]
                ext          = base_image.get("ext", "png").lower()

                # Posição Y na página (usa o primeiro rect encontrado)
                rects = page.get_image_rects(xref)
                y0    = rects[0].y0 if rects else 0.0

                seen_xrefs.add(xref)
                images.append(_RawImage(
                    xref=xref,
                    page_num=page_num,
                    y0=y0,
                    image_bytes=image_bytes,
                    ext=ext,
                    width=width,
                    height=height,
                ))
            except Exception as e:
                logger.debug(
                    f"[ImageService] Não foi possível extrair xref={xref}: {e}"
                )

    return images


# ── Associação imagem → questão ───────────────────────────────────────────────

def _assign_to_question(
    img_page: int,
    img_y: float,
    q_positions: dict[int, list[tuple[float, int]]],
) -> int:
    """
    Determina a qual questão a imagem pertence.

    Estratégia:
      1. Na mesma página: última questão que começa ANTES da imagem (±20pt)
      2. Fallback: primeira questão da página (imagem aparece antes das questões)
      3. Fallback: última questão da página anterior

    Retorna 0 se nenhuma questão puder ser associada.
    """
    page_qs = q_positions.get(img_page, [])

    if page_qs:
        candidate: int | None = None
        for y0, q_num in page_qs:
            if y0 <= img_y + 20:    # tolerância de 20pt
                candidate = q_num
            else:
                break
        if candidate is not None:
            return candidate
        # Imagem aparece antes de todas as questões na página → primeira questão
        return page_qs[0][1]

    # Sem questões nesta página → última questão da página anterior
    for search_page in range(img_page - 1, -1, -1):
        prev_qs = q_positions.get(search_page, [])
        if prev_qs:
            return prev_qs[-1][1]

    return 0  # não associado


# ── Upload ────────────────────────────────────────────────────────────────────

def _upload_image(
    cb: "SupabaseCallbackClient",
    file_path: str,
    image_bytes: bytes,
    ext: str,
) -> None:
    """
    Faz upload de uma imagem para o Supabase Storage.

    Fluxo:
      1. POST process-callback action='get_signed_upload_url' → signed_url
      2. PUT image_bytes para signed_url com Content-Type correto

    Lança RuntimeError em caso de falha.
    """
    signed_url  = cb.get_signed_upload_url(file_path, bucket="pdfs")
    content_type = _MIME.get(ext.lower(), "application/octet-stream")

    response = httpx.put(
        signed_url,
        content=image_bytes,
        headers={"Content-Type": content_type},
        timeout=60.0,
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"Upload falhou HTTP {response.status_code}: {response.text[:200]}"
        )
