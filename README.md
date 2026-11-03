# Assistant RGPH-5 – Supabase

## Structure
- `app.py` : assistant Gradio, RAG, transcription audio, historique Supabase et export Excel.
- `requirements.txt` : dépendances Python.
- `supabase_schema.sql` : script de création de la table.
- `folder_rag/` : déposer ici les PDF RGPH-5.

## 1. Configurer Supabase
1. Créer un projet sur https://supabase.com/
2. Ouvrir SQL Editor.
3. Exécuter le contenu de `supabase_schema.sql`.
4. Dans Project Settings > API, récupérer l'URL du projet et une clé secrète (service_role/secret).
5. Ne jamais publier cette clé dans GitHub ou dans le navigateur.

## 2. Variables d'environnement Render
Ajouter dans Environment :
- `OPENAI_API_KEY` : clé API OpenAI.
- `SUPABASE_URL` : URL du projet Supabase.
- `SUPABASE_KEY` : clé secrète Supabase côté serveur (service_role/secret).
- `ADMIN_PASSWORD` : mot de passe fort pour l'onglet Administration.

## 3. Déploiement Render
Build Command : `pip install -r requirements.txt`
Start Command : `python app.py`

Le port est lu depuis la variable `PORT` fournie par Render.

## 4. Fonctionnalités
- Questions écrites et audio en français.
- Réponses fondées sur les PDF et références documentaires.
- Enregistrement durable des questions, réponses, sources, dates et statuts dans Supabase.
- Consultation de l'historique avec mot de passe administrateur.
- Export Excel des interactions.

## Sécurité et confidentialité
- La clé Supabase secrète ne doit jamais être intégrée au code côté navigateur ni publiée.
- L'accès à l'historique est protégé dans l'interface par `ADMIN_PASSWORD`.
- Informer les utilisateurs de la collecte des questions et réponses et définir une durée de conservation adaptée.
- Pour un usage institutionnel, renforcer l'administration avec une authentification dédiée plutôt qu'un simple mot de passe partagé.
