<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1FqeNKsRJ7SV-iHiCS98y1nXHChPirUcW

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

## Render RAG Bootstrap

To make RAG retrieval available on Render without committing `chroma_db`, the app can
download your index from Hugging Face on startup.

Set these environment variables in Render:
- `RAG_PERSIST_DIRECTORY=/var/data/chroma_db` (recommended if using Render Disk)
- `RAG_HF_BOOTSTRAP=1`
- `RAG_HF_REPO_ID=Agnes999/legalbot8`
- `RAG_HF_REPO_TYPE=dataset`
- `HF_TOKEN=<your_huggingface_token>` (required only if dataset is private)

# legal-doc
 
# legal-bot

# legalbot8
