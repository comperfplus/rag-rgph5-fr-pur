"""Assistant RGPH-5 : questions textuelles ou audio en français."""
import logging
import os
from pathlib import Path

import gradio as gr
import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "folder_rag"
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("assistant_rgph5")

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Ajoutez OPENAI_API_KEY dans les Secrets du Space Hugging Face."
    )

client = OpenAI(api_key=API_KEY, timeout=120.0)

# Modèles et paramètres
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


def build_chunks():
    """Extrait les PDF et conserve le nom du fichier et le numéro de page."""
    pdfs = sorted(PDF_DIR.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"Aucun PDF trouvé dans {PDF_DIR}. "
            "Placez les documents dans le dossier folder_rag."
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
                    text = encoding.decode(tokens[start:end]).strip()

                    if text:
                        chunks.append({
                            "text": text,
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
    """Calcule les embeddings par lots."""
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
    """Retourne les segments les plus proches de la question."""
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
    return (response.choices[0].message.content or "").strip()


def transcribe_audio(audio_path):
    """Transcrit un fichier audio français via l'API OpenAI."""
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
    """Traite une question provenant d'un seul des deux modes."""
    text_question = (text_question or "").strip()

    if bool(text_question) == bool(audio_path):
        return "", "Choisissez un seul mode : texte OU audio."

    try:
        question = (
            transcribe_audio(audio_path)
            if audio_path
            else text_question
        )
        return question, answer_question(question)

    except ValueError as error:
        return "", str(error)
    except Exception:
        logger.exception("Erreur pendant le traitement de la question.")
        return "", (
            "Une erreur technique est survenue. "
            "Veuillez réessayer dans quelques instants."
        )


# Interface utilisateur
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
        text_question_output = gr.Textbox(
            label="Question traitée", interactive=False
        )
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

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
