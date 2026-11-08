# Assistant RGPH-5 — version Gemini

## Modifications
- Génération des réponses : Gemini (`gemini-3.8-flash`)
- Embeddings : Gemini (`gemini-embedding-2`)
- Transcription audio : Gemini (`gemini-3.8-flash`)
- Historique Supabase conservé
- Statuts `success`, `not_found` et `error` conservés
- Interface Gradio et lecture des PDF conservées

## Variables Render
Conserver les variables :
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `ADMIN_PASSWORD`

Remplacer `OPENAI_API_KEY` par :
- `GEMINI_API_KEY` : clé API obtenue depuis Google AI Studio

## Déploiement
1. Remplacer `app.py` et `requirements.txt` dans le dépôt GitHub.
2. Dans Render > Environment, ajouter `GEMINI_API_KEY`.
3. Vérifier les trois variables Supabase et `ADMIN_PASSWORD`.
4. Supprimer `OPENAI_API_KEY` si elle n'est plus utilisée.
5. Commit/push puis attendre le redéploiement Render.

Les PDF doivent rester dans le dossier `folder_rag/` du dépôt.
La clé Gemini ne doit jamais être enregistrée dans GitHub ni partagée.
Les quotas et éventuels coûts dépendent du compte Google AI Studio et du modèle accessible.
