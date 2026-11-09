"""Assistant RGPH-5 : RAG français, audio et historique persistant dans Supabase."""

import logging
import mimetypes
import json
import html
import os
import re
import math
import time
from collections import Counter
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np
import tiktoken
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from supabase import create_client

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "folder_rag"
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("assistant_rgph5")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()

if not GEMINI_API_KEY:
    raise RuntimeError("Secret GEMINI_API_KEY manquant.")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Secrets SUPABASE_URL et SUPABASE_KEY manquants.")

client = genai.Client(api_key=GEMINI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

LLM_MODEL = "gemini-3.5-flash-lite"
TRANSCRIPTION_MODEL = "gemini-3.8-flash"
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50
TOP_K = 3
MAX_AUDIO_SIZE_MB = 25

TRANSCRIPTION_PROMPT = (
    "Enregistrement en français sur les Recensement de la population "
    "et de l'habitat du Sénégal (RGPH), les ménages, la population, "
    "les régions, les départements, les communes et les statistiques de l'ANSD."
)

# ---------------------------------------------------------------------
# SUPABASE : ENREGISTREMENT ET LECTURE DE L'HISTORIQUE
# ---------------------------------------------------------------------

TABLE_NAME = "historique_interactions"


def save_interaction(
    question, answer, mode, sources="", status="success",
    duration_traitment=0.0, llm_modele_name=LLM_MODEL,
    transcript_modele_name=""
):
    """Enregistre une interaction durablement dans Supabase."""
    payload = {
        "date_question": datetime.now(timezone.utc).isoformat(),
        "type_question": mode,
        "question": question or "",
        "reponse": answer or "",
        "sources": sources or "",
        "statut": status,
        "duration_traitment": round(float(duration_traitment), 3),
        "llm_modele_name": llm_modele_name or "",
        "transcript_modele_name": transcript_modele_name or "",
    }
    result = supabase.table(TABLE_NAME).insert(payload).execute()
    return result


def get_history_rows(limit=5000):
    """Récupère les dernières interactions depuis Supabase."""
    result = (
        supabase.table(TABLE_NAME)
        .select("date_question,type_question,question,reponse,sources,statut,"
                "duration_traitment,llm_modele_name,transcript_modele_name")
        .order("date_question", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def create_excel_export(rows):
    """Génère un fichier Excel à partir des interactions."""
    headers = [
        "date_question", "type_question", "question",
        "reponse", "sources", "statut", "duration_traitment",
        "llm_modele_name", "transcript_modele_name"
    ]
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Historique RGPH5"
    sheet.append(headers)

    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")

    for row in reversed(rows):
        sheet.append([row.get(column, "") or "" for column in headers])

    widths = {"A": 25, "B": 16, "C": 55, "D": 75, "E": 55,
              "F": 14, "G": 20, "H": 28, "I": 30}
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions

    temp = tempfile.NamedTemporaryFile(
        prefix="historique_rgph5_", suffix=".xlsx", delete=False
    )
    temp.close()
    workbook.save(temp.name)
    return temp.name


# ---------------------------------------------------------------------
# INDEXATION PDF ET RECHERCHE BM25 LOCALE
# ---------------------------------------------------------------------

def build_chunks():
    pdfs = sorted(PDF_DIR.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"Aucun PDF trouvé dans {PDF_DIR}. Placez les documents dans folder_rag."
        )

    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    chunks = []

    for pdf in pdfs:
        try:
            reader = PdfReader(str(pdf))
            for page_no, page in enumerate(reader.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if not page_text:
                    continue

                tokens = encoding.encode(page_text)
                start = 0
                while start < len(tokens):
                    end = min(start + CHUNK_SIZE, len(tokens))
                    chunk_text = encoding.decode(tokens[start:end]).strip()

                    if chunk_text:
                        chunks.append({
                            "text": chunk_text,
                            "source": pdf.name,
                            "page": page_no,
                        })

                    if end >= len(tokens):
                        break
                    start = max(end - CHUNK_OVERLAP, start + 1)

        except Exception:
            logger.exception("Erreur de lecture du PDF %s", pdf.name)

    if not chunks:
        raise ValueError(
            "Aucun texte exploitable extrait des PDF. "
            "Les documents scannés nécessitent un OCR préalable."
        )
    return chunks


def tokenize(text):
    """Tokenisation locale simple, adaptée aux questions en français."""
    return re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ0-9]+", (text or "").lower())


def build_bm25_index(documents):
    """Construit un index BM25 local sans appel réseau ni embedding distant."""
    tokenized = [tokenize(item["text"]) for item in documents]
    doc_lengths = [len(tokens) for tokens in tokenized]
    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)

    document_frequency = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    total_docs = len(tokenized)
    idf = {
        term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in document_frequency.items()
    }
    term_frequencies = [Counter(tokens) for tokens in tokenized]

    return {
        "lengths": doc_lengths,
        "avgdl": avgdl,
        "idf": idf,
        "tf": term_frequencies,
    }


logger.info("Extraction et indexation locale des documents PDF...")
index_start = time.perf_counter()
corpus = build_chunks()
bm25_index = build_bm25_index(corpus)
logger.info(
    "Index BM25 prêt : %d segments en %.2f secondes",
    len(corpus),
    time.perf_counter() - index_start,
)


def retrieve(question):
    """Recherche lexicale locale BM25, sans appel Gemini pour les embeddings."""
    query_terms = tokenize(question)
    if not query_terms:
        return []

    k1 = 1.5
    b = 0.75
    avgdl = bm25_index["avgdl"] or 1.0
    scores = []

    for idx, term_counts in enumerate(bm25_index["tf"]):
        doc_len = bm25_index["lengths"][idx]
        score = 0.0

        for term in query_terms:
            frequency = term_counts.get(term, 0)
            if frequency == 0:
                continue

            idf_value = bm25_index["idf"].get(term, 0.0)
            denominator = frequency + k1 * (
                1 - b + b * doc_len / avgdl
            )
            score += idf_value * (
                frequency * (k1 + 1) / denominator
            )

        scores.append(score)

    best_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:TOP_K]

    return [
        corpus[i]
        for i in best_indices
        if scores[i] > 0
    ]


def format_answer_for_display(answer, status):
    """Affiche les réponses valides en bleu et les absences/erreurs en rouge."""
    safe_answer = html.escape(str(answer or "")).replace("\\n", "<br>")
    color = "#1565C0" if status == "success" else "#D32F2F"
    return (
        f'<div style="color:{color}; font-weight:700;">'
        f'<strong>Réponses:</strong><br>{safe_answer.replace(chr(10), "<br>")}'
        '</div>'
    )


def answer_question(question):
    """Recherche locale puis génération de réponse avec Gemini Flash-Lite."""
    total_start = time.perf_counter()

    retrieval_start = time.perf_counter()
    passages = retrieve(question)
    retrieval_seconds = time.perf_counter() - retrieval_start

    if not passages:
        answer = (
            "Je n'ai pas trouvé de passage correspondant à cette question "
            "dans les documents RGPH-5 disponibles."
        )
        logger.info(
            "PERFORMANCE question | recherche_locale=%.2fs | Gemini=0.00s "
            "| total=%.2fs | passages=0",
            retrieval_seconds,
            time.perf_counter() - total_start,
        )
        return answer, "", "not_found"

    context = "\n\n".join(
        f"[Source : {item['source']}, page {item['page']}]\n{item['text']}"
        for item in passages
    )

    prompt = f"""
Tu es un assistant statistique spécialisé dans les recensements du Sénégal.
Réponds en français, clairement et de manière concise.
Utilise exclusivement le contexte documentaire fourni.
N'invente aucun chiffre, aucune date ni aucun résultat.

Si le contexte ne permet pas de répondre de façon suffisamment fiable,
indique que l'information n'a pas été trouvée et choisis le statut "not_found".
Si le contexte permet de répondre, choisis le statut "success".
Une absence d'information n'est jamais une erreur technique.
Cite les sources sous la forme [document, page X].
Signale les contradictions éventuelles entre les sources.

Retourne uniquement un objet JSON valide avec :
- "status" : "success" ou "not_found"
- "answer" : la réponse destinée à l'utilisateur

CONTEXTE DOCUMENTAIRE :
{context}

QUESTION :
{question}
"""

    generation_start = time.perf_counter()
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=700,
            response_mime_type="application/json",
        ),
    )
    generation_seconds = time.perf_counter() - generation_start

    raw_content = response.text or "{}"
    try:
        result = json.loads(raw_content)
        answer = str(result.get("answer", "")).strip()
        status = result.get("status", "not_found")

        if status not in ("success", "not_found"):
            status = "not_found"

        if not answer:
            answer = (
                "Je n'ai pas trouvé d'information suffisamment fiable "
                "pour répondre à cette question dans les documents disponibles."
            )
            status = "not_found"

    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.exception("Gemini n'a pas retourné le JSON attendu.")
        answer = (
            "Je n'ai pas pu déterminer une réponse fiable à partir "
            "des documents disponibles."
        )
        status = "error"

    sources = "; ".join(
        sorted({
            f"{item['source']} - page {item['page']}"
            for item in passages
        })
    )

    logger.info(
        "PERFORMANCE question | recherche_locale=%.2fs | Gemini=%.2fs "
        "| total=%.2fs | passages=%d",
        retrieval_seconds,
        generation_seconds,
        time.perf_counter() - total_start,
        len(passages),
    )
    return answer, sources, status


# ---------------------------------------------------------------------
# TRANSCRIPTION AUDIO
# ---------------------------------------------------------------------

def transcribe_audio(audio_path):
    """Transcrit un audio français avec Gemini."""
    if not audio_path:
        raise ValueError("Aucun enregistrement audio fourni.")

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError("Fichier audio introuvable.")

    if path.stat().st_size > MAX_AUDIO_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"Le fichier audio dépasse {MAX_AUDIO_SIZE_MB} Mo. "
            "Veuillez le raccourcir ou le compresser."
        )

    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    audio_bytes = path.read_bytes()

    response = client.models.generate_content(
        model=TRANSCRIPTION_MODEL,
        contents=[
            TRANSCRIPTION_PROMPT
            + " Transcris fidèlement les paroles en français. "
              "Retourne uniquement le texte prononcé, sans commentaire.",
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(temperature=0),
    )

    transcription = (response.text or "").strip()
    if not transcription:
        raise ValueError("Aucune parole n'a pu être transcrite.")
    return transcription


def process_question(text_question=None, audio_path=None):
    text_question = (text_question or "").strip()

    if bool(text_question) == bool(audio_path):
        return "", format_answer_for_display("Choisissez un seul mode : texte OU audio.", "error")

    mode = "audio" if audio_path else "texte"
    question = ""
    processing_start = time.perf_counter()

    try:
        if audio_path:
            transcription_start = time.perf_counter()
            question = transcribe_audio(audio_path)
            logger.info(
                "PERFORMANCE transcription Gemini=%.2fs",
                time.perf_counter() - transcription_start,
            )
        else:
            question = text_question
        answer, sources, status = answer_question(question)
        duration_traitment = time.perf_counter() - processing_start

        try:
            save_interaction(
                question, answer, mode, sources, status,
                duration_traitment=duration_traitment,
                llm_modele_name=LLM_MODEL,
                transcript_modele_name=TRANSCRIPTION_MODEL if mode == "audio" else "",
            )
        except Exception:
            logger.exception("La réponse a été générée, mais son enregistrement Supabase a échoué.")

        logger.info(
            "PERFORMANCE traitement complet=%.2fs",
            time.perf_counter() - processing_start,
        )
        return question, format_answer_for_display(answer, status)

    except ValueError as error:
        duration_traitment = time.perf_counter() - processing_start
        try:
            save_interaction(
                question or text_question, str(error), mode, "", "error",
                duration_traitment=duration_traitment,
                llm_modele_name=LLM_MODEL,
                transcript_modele_name=TRANSCRIPTION_MODEL if mode == "audio" else "",
            )
        except Exception:
            logger.exception("Impossible d'enregistrer l'erreur dans Supabase.")
        return question or "", format_answer_for_display(str(error), "error")

    except Exception:
        logger.exception("Erreur pendant le traitement de la question.")
        message = (
            "Une erreur technique est survenue. "
            "Veuillez réessayer dans quelques instants."
        )
        duration_traitment = time.perf_counter() - processing_start
        try:
            save_interaction(
                question or text_question, message, mode, "", "error",
                duration_traitment=duration_traitment,
                llm_modele_name=LLM_MODEL,
                transcript_modele_name=TRANSCRIPTION_MODEL if mode == "audio" else "",
            )
        except Exception:
            logger.exception("Impossible d'enregistrer l'erreur dans Supabase.")
        return question or "", format_answer_for_display(message, "error")


# ---------------------------------------------------------------------
# ADMINISTRATION : HISTORIQUE ET EXPORT EXCEL
# ---------------------------------------------------------------------

def load_admin_history(password):
    if not ADMIN_PASSWORD:
        return "Configurez ADMIN_PASSWORD dans les variables d'environnement Render.", []

    if not password or password != ADMIN_PASSWORD:
        return "Mot de passe incorrect.", []

    try:
        rows = get_history_rows()
        table = [
            [
                row.get("date_question", ""),
                row.get("type_question", ""),
                row.get("question", ""),
                row.get("reponse", ""),
                row.get("sources", ""),
                row.get("statut", ""),
                str(row.get("duration_traitment", "")),
                row.get("llm_modele_name", ""),
                row.get("transcript_modele_name", ""),
            ]
            for row in rows
        ]
        return f"{len(rows)} interaction(s) trouvée(s) dans Supabase.", table
    except Exception:
        logger.exception("Erreur de lecture de l'historique Supabase.")
        return "Impossible de lire l'historique Supabase. Vérifiez la configuration et les droits.", []


def export_admin_history(password):
    if not ADMIN_PASSWORD:
        raise gr.Error("Configurez ADMIN_PASSWORD dans les variables Render.")
    if not password or password != ADMIN_PASSWORD:
        raise gr.Error("Mot de passe administrateur incorrect.")

    try:
        rows = get_history_rows()
        if not rows:
            raise gr.Error("Aucune interaction à exporter.")
        return create_excel_export(rows)
    except gr.Error:
        raise
    except Exception:
        logger.exception("Erreur pendant l'export Excel.")
        raise gr.Error("L'export a échoué. Vérifiez la connexion à Supabase.")


# ---------------------------------------------------------------------
# INTERFACE GRADIO
# ---------------------------------------------------------------------

CSS = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
}
footer { visibility: hidden; }
"""

with gr.Blocks(
    title="STATSENEGAL - GPT",
    css=CSS,
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # Assistant statistique(STATSENEGAL - GPT) – RGPH

        Posez vos questions sur les résultats des Recensements généraux
        de la population et de l'habitat du Sénégal.

        Vous pouvez écrire votre question ou l'enregistrer en audio ou charger l'audio.
        """
    )

    with gr.Tab("Question écrite"):
        text_input = gr.Textbox(
            label="Votre question",
            placeholder="Ex. Quelle est la population du Sénégal en 2023 ?",
            lines=3,
        )
        text_button = gr.Button("Envoyer", variant="primary")
        text_question_output = gr.Textbox(label="Question traitée", interactive=False)
        text_answer_output = gr.HTML()
        text_button.click(
            fn=lambda question: process_question(text_question=question),
            inputs=text_input,
            outputs=[text_question_output, text_answer_output],
        )

    with gr.Tab("Question audio"):
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Enregistrez votre question",
        )
        audio_button = gr.Button("Transcrire et envoyer", variant="primary")
        audio_question_output = gr.Textbox(
            label="Transcription de votre question",
            lines=3,
            interactive=False,
        )
        audio_answer_output = gr.HTML()
        audio_button.click(
            fn=lambda audio: process_question(audio_path=audio),
            inputs=audio_input,
            outputs=[audio_question_output, audio_answer_output],
        )

    with gr.Tab("Administration – Historique"):
        gr.Markdown(
            """
            ### Consultation des interactions
            Cette rubrique est réservée à l'administrateur.
            Définissez `ADMIN_PASSWORD` dans les variables d'environnement Render.
            Les questions, réponses et sources sont stockées dans Supabase.
            """
        )
        admin_password = gr.Textbox(
            label="Mot de passe administrateur",
            type="password",
        )
        with gr.Row():
            history_button = gr.Button("Consulter l'historique", variant="primary")
            export_button = gr.Button("Exporter en Excel")

        admin_status = gr.Markdown()
        history_table = gr.Dataframe(
            headers=[
                "Date/heure", "Type", "Question", "Réponse", "Sources",
                "Statut", "Durée (s)", "Modèle LLM", "Modèle transcription"
            ],
            datatype=["str", "str", "str", "str", "str", "str", "number", "str", "str"],
            interactive=False,
            wrap=True,
            label="Historique des questions et réponses",
        )
        excel_file = gr.File(label="Fichier Excel")

        history_button.click(
            fn=load_admin_history,
            inputs=admin_password,
            outputs=[admin_status, history_table],
        )
        export_button.click(
            fn=export_admin_history,
            inputs=admin_password,
            outputs=excel_file,
        )

    gr.Markdown(
        """
        ---
        <small>Les réponses sont produites à partir des documents intégrés.
        Consultez les références pour retrouver les passages sources.</small>
        """
    )


if __name__ == "__main__":
    demo.queue(max_size=64, default_concurrency_limit=8)
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
