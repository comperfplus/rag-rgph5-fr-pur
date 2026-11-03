"""Assistant RGPH-5 : RAG français, audio et historique persistant dans Supabase."""

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
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

API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()

if not API_KEY:
    raise RuntimeError("Secret OPENAI_API_KEY manquant.")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Secrets SUPABASE_URL et SUPABASE_KEY manquants.")

client = OpenAI(api_key=API_KEY, timeout=120.0)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 70
TOP_K = 5
MAX_AUDIO_SIZE_MB = 25

TRANSCRIPTION_PROMPT = (
    "Enregistrement en français sur le Recensement général de la population "
    "et de l'habitat du Sénégal (RGPH-5), les ménages, la population, "
    "les régions, les départements, les communes et les statistiques de l'ANSD."
)

# ---------------------------------------------------------------------
# SUPABASE : ENREGISTREMENT ET LECTURE DE L'HISTORIQUE
# ---------------------------------------------------------------------

TABLE_NAME = "historique_interactions"


def save_interaction(question, answer, mode, sources="", status="success"):
    """Enregistre une interaction durablement dans Supabase."""
    payload = {
        "date_question": datetime.now(timezone.utc).isoformat(),
        "type_question": mode,
        "question": question or "",
        "reponse": answer or "",
        "sources": sources or "",
        "statut": status,
    }
    result = supabase.table(TABLE_NAME).insert(payload).execute()
    return result


def get_history_rows(limit=5000):
    """Récupère les dernières interactions depuis Supabase."""
    result = (
        supabase.table(TABLE_NAME)
        .select("date_question,type_question,question,reponse,sources,statut")
        .order("date_question", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def create_excel_export(rows):
    """Génère un fichier Excel à partir des interactions."""
    headers = [
        "date_question", "type_question", "question",
        "reponse", "sources", "statut"
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

    widths = {"A": 25, "B": 16, "C": 55, "D": 75, "E": 55, "F": 14}
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
# INDEXATION PDF ET RAG
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


def embed(texts):
    vectors = []
    for start in range(0, len(texts), 64):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts[start:start + 64],
        )
        vectors.extend(item.embedding for item in response.data)
    return np.asarray(vectors, dtype=np.float32)


logger.info("Indexation des documents PDF...")
corpus = build_chunks()
corpus_vectors = embed([item["text"] for item in corpus])
logger.info("Index RAG prêt : %d segments", len(corpus))


def retrieve(question):
    query_vector = embed([question])[0]
    norms = (
        np.linalg.norm(corpus_vectors, axis=1)
        * np.linalg.norm(query_vector)
    )
    scores = np.dot(corpus_vectors, query_vector) / np.maximum(norms, 1e-12)
    indices = np.argsort(scores)[::-1][:TOP_K]
    return [corpus[int(index)] for index in indices]


def answer_question(question):
    passages = retrieve(question)
    context = "\n\n".join(
        f"[Source : {item['source']}, page {item['page']}]\n{item['text']}"
        for item in passages
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant statistique spécialisé dans le RGPH-5 "
                    "du Sénégal. Réponds en français, clairement et de manière "
                    "pédagogique. Utilise uniquement le contexte documentaire. "
                    "N'invente aucun chiffre, aucune date ni aucun résultat. "
                    "Si l'information manque, indique-le explicitement. "
                    "Cite les sources sous la forme [document, page X]. "
                    "Signale les éventuelles contradictions entre les sources."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Contexte documentaire :\n{context}\n\n"
                    f"Question : {question}"
                ),
            },
        ],
    )

    answer = (response.choices[0].message.content or "").strip()
    sources = "; ".join(
        sorted({
            f"{item['source']} - page {item['page']}"
            for item in passages
        })
    )
    return answer, sources


# ---------------------------------------------------------------------
# TRANSCRIPTION AUDIO
# ---------------------------------------------------------------------

def transcribe_audio(audio_path):
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

    with path.open("rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=audio_file,
            prompt=TRANSCRIPTION_PROMPT,
        )

    transcription = (getattr(result, "text", "") or "").strip()
    if not transcription:
        raise ValueError("Aucune parole n'a pu être transcrite.")
    return transcription


def process_question(text_question=None, audio_path=None):
    text_question = (text_question or "").strip()

    if bool(text_question) == bool(audio_path):
        return "", "Choisissez un seul mode : texte OU audio."

    mode = "audio" if audio_path else "texte"
    question = ""

    try:
        question = transcribe_audio(audio_path) if audio_path else text_question
        answer, sources = answer_question(question)

        try:
            save_interaction(question, answer, mode, sources, "success")
        except Exception:
            logger.exception("La réponse a été générée, mais son enregistrement Supabase a échoué.")

        return question, answer

    except ValueError as error:
        try:
            save_interaction(question or text_question, str(error), mode, "", "error")
        except Exception:
            logger.exception("Impossible d'enregistrer l'erreur dans Supabase.")
        return question or "", str(error)

    except Exception:
        logger.exception("Erreur pendant le traitement de la question.")
        message = (
            "Une erreur technique est survenue. "
            "Veuillez réessayer dans quelques instants."
        )
        try:
            save_interaction(question or text_question, message, mode, "", "error")
        except Exception:
            logger.exception("Impossible d'enregistrer l'erreur dans Supabase.")
        return question or "", message


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
    title="Assistant statistique RGPH-5",
    css=CSS,
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # Assistant statistique – RGPH-5

        Posez vos questions sur les résultats du Recensement général
        de la population et de l'habitat du Sénégal.

        Vous pouvez écrire votre question ou l'enregistrer en français.
        """
    )

    with gr.Tab("Question écrite"):
        text_input = gr.Textbox(
            label="Votre question",
            placeholder="Ex. Quelle est la population du Sénégal selon le RGPH-5 ?",
            lines=3,
        )
        text_button = gr.Button("Envoyer", variant="primary")
        text_question_output = gr.Textbox(label="Question traitée", interactive=False)
        text_answer_output = gr.Markdown()
        text_button.click(
            fn=lambda question: process_question(text_question=question),
            inputs=text_input,
            outputs=[text_question_output, text_answer_output],
        )

    with gr.Tab("Question audio"):
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Enregistrez votre question en français",
        )
        audio_button = gr.Button("Transcrire et envoyer", variant="primary")
        audio_question_output = gr.Textbox(
            label="Transcription de votre question",
            lines=3,
            interactive=False,
        )
        audio_answer_output = gr.Markdown()
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
                "Date/heure", "Type", "Question", "Réponse", "Sources", "Statut"
            ],
            datatype=["str", "str", "str", "str", "str", "str"],
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
