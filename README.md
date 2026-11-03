# Assistant RGPH-5 simplifié

## Contenu
- `app.py` : application Gradio.
- `requirements.txt` : dépendances Python.
- `folder_rag/` : dossier à remplir avec les PDF du RGPH-5.

## Déploiement sur Hugging Face Spaces
1. Remplacer l'ancien `app.py` par ce nouveau fichier.
2. Remplacer `requirements.txt`.
3. Ajouter les documents PDF dans `folder_rag/`.
4. Dans Settings > Secrets du Space, ajouter `OPENAI_API_KEY`.
5. Redémarrer le Space.

## Fonctionnement
- Question écrite en français : envoyée directement au RAG.
- Question audio en français : transcrite par l'API OpenAI `gpt-4o-mini-transcribe`.
- Recherche documentaire : `text-embedding-3-small`.
- Génération de réponse : `gpt-4o-mini`.

## À savoir
- Aucun GPU RunPod n'est nécessaire avec cette architecture.
- Les PDF scannés sans couche texte nécessitent un OCR.
- Les embeddings sont recalculés à chaque démarrage du Space. Pour un service
  à fort trafic, prévoir une indexation persistante afin de limiter les coûts
  et le temps de démarrage.
- La limite audio configurée dans le code est de 25 Mo.
- Tester la qualité de transcription et les références sur un jeu de questions
  avant l'ouverture au public.
- Ne jamais publier la clé API OpenAI dans le code ou dans un dépôt public.
