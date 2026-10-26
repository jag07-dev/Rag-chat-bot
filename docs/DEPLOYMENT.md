# Deployment Guide

Two ways to put this app online. Start with Option A: it is free, needs no credit card, and takes about 10 minutes. Use Option B if Option A runs out of memory or you want a cloud deployment to talk about.

Platform limits and prices change, so check the current Streamlit and Google Cloud documentation before relying on any number in this file.

## Before you deploy (both options)

1. **Keep secrets out of Git.** `.env` must be listed in `.gitignore` (it is, in this repo). Never commit your Groq key.
2. **Check `requirements.txt`** lists: `streamlit`, `pypdf`, `sentence-transformers`, `faiss-cpu`, `groq`, `python-dotenv`.
3. **Use Python 3.11 or 3.12 in the cloud.** Your local machine may run a newer version, and some packages (PyTorch, FAISS) publish wheels for newer Python versions later than for the established ones.
4. **Test locally from a clean environment** (new virtual environment, `pip install -r requirements.txt`, `streamlit run app.py`). If it works there, the cloud install will usually work too.
5. **Remember a public app uses your Groq quota.** Anyone with the link can ask questions. Share the link in applications, not publicly.

## Option A: Streamlit Community Cloud (free)

1. **Push the project to GitHub.**
   ```bash
   git init
   git add .
   git status        # confirm .env is NOT listed
   git commit -m "RAG document chatbot"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
   The repository must be public to use the free tier's public deployment, or you can use your one private app slot.
2. **Sign in** at share.streamlit.io with your GitHub account and authorize access to the repository.
3. **Click "Create app"** (or "New app") and choose the repository, the `main` branch, and `app.py` as the main file.
4. **Open "Advanced settings"** and:
   - choose **Python 3.12** (or 3.11);
   - paste your secrets in TOML format:
     ```toml
     GROQ_API_KEY = "your_groq_api_key_here"
     GROQ_MODEL = "openai/gpt-oss-120b"
     ```
5. **Click Deploy.** The first build installs PyTorch and the embedding model, so it can take several minutes. Watch the logs.
6. **Open the app URL**, upload `data/rag_demo_company_handbook.pdf`, and ask: "How many days per week can eligible employees work remotely?" You should get "up to two days per week" with a page caption.
7. **Put the URL in your README** (replace `<your-app-url>`), commit, and push. The app redeploys automatically on every push.

### If the app cannot read the API key

Streamlit exposes top-level secrets to the app, and the code reads them with `os.getenv`. If your key still shows as missing, add this near the top of `app.py` instead of reading only from the environment:

```python
def get_setting(name, default=None):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return default

GROQ_API_KEY = get_setting("GROQ_API_KEY")
GROQ_MODEL = get_setting("GROQ_MODEL", "openai/gpt-oss-120b")
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "This app has gone over its resource limits" | PyTorch plus the model exceed the memory limit | Try Option B, or a hosting service with more memory |
| Install is very slow or fails on `torch` | The default Linux wheel is the large GPU build | Add `--extra-index-url https://download.pytorch.org/whl/cpu` as the first line of `requirements.txt` and `torch` as a line below it |
| `ModuleNotFoundError` | Package missing from `requirements.txt` | Add it, commit, push |
| Model not found (404) | The model was retired | Set `GROQ_MODEL` to a current model from Groq's model list |
| App is slow on the first visit | Free apps sleep after a period without traffic | Open the link once before an interview or demo |

## Option B: Google Cloud Run (free allowance, billing account required)

Cloud Run has a monthly free allowance, but a billing account must be enabled to use it, so set a budget alert and a low instance limit before you deploy.

1. **Set up Google Cloud.** Create a project, enable billing, install the `gcloud` CLI, and run `gcloud auth login` and `gcloud config set project <PROJECT_ID>`.
2. **Enable the services:**
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
   ```
3. **Add a `Dockerfile`** to the project root:
   ```dockerfile
   FROM python:3.12-slim
   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Download the embedding model at build time so the first request is fast
   RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

   COPY app.py .

   ENV PORT=8080
   CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
   ```
4. **Store the API key in Secret Manager** (never in the image):
   ```bash
   echo -n "your_groq_api_key_here" | gcloud secrets create groq-api-key --data-file=-
   ```
   Then allow the Cloud Run service account to read it (Console: Secret Manager, the secret, Permissions, grant "Secret Manager Secret Accessor" to the default compute service account).
5. **Deploy from source:**
   ```bash
   gcloud run deploy rag-chatbot \
     --source . \
     --region <your-region> \
     --memory 2Gi --cpu 2 \
     --max-instances 2 \
     --session-affinity \
     --allow-unauthenticated \
     --set-env-vars GROQ_MODEL=openai/gpt-oss-120b \
     --set-secrets GROQ_API_KEY=groq-api-key:latest
   ```
   `--max-instances 2` caps cost, `--memory 2Gi` gives PyTorch room, and `--session-affinity` keeps one visitor's Streamlit session on one instance.
6. **Open the URL** the command prints and run the same test question as in Option A.
7. **Set a budget alert** in Billing, Budgets and alerts, so an unexpected cost emails you.
8. **Clean up when the job search is over:** `gcloud run services delete rag-chatbot --region <your-region>`.

## What to say in an interview

> "I deployed it on Streamlit Community Cloud, connected to the GitHub repo, so every push redeploys. The Groq key lives in the platform's secrets, not in the repository. I pinned a Python version that has wheels for FAISS and PyTorch. I also know the limits: free apps sleep after a period without traffic and have a memory cap, and a public link uses my API quota. For something production-like I'd containerize it and run it on Cloud Run with the key in Secret Manager, a capped number of instances, and a budget alert, and I'd add authentication and rate limiting."

Only say you used Cloud Run if you actually deployed it there.
