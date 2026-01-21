# LTX-2 PyWorker for Vast.ai Serverless

This directory contains the PyWorker configuration for the LTX-2 video generation API.

## Repository Structure

When you create your PYWORKER_REPO, push the contents of this directory to the **root** of your repo:

```
your-pyworker-repo/
├── worker.py        # <-- Must be at root!
├── requirements.txt
└── README.md
```

**NOT** in a subdirectory:
```
your-pyworker-repo/
└── pyworker/        # ❌ Wrong!
    └── worker.py
```

## Deployment

1. Create a new GitHub repo (e.g., `ltx2-pyworker`)

2. Push this directory's contents to it:
   ```bash
   cd pyworker
   git init
   git add .
   git commit -m "Initial PyWorker configuration"
   git remote add origin https://github.com/YOUR_USER/ltx2-pyworker.git
   git push -u origin main
   ```

3. In your Vast.ai template, set:
   - `PYWORKER_REPO`: `https://github.com/YOUR_USER/ltx2-pyworker`
   - `PYWORKER_REF`: `main` (optional)

## How It Works

The Vast.ai `start_server.sh` will:
1. Clone your repo to `/workspace/vast-pyworker`
2. Create a venv and install `requirements.txt`
3. Install `vastai-sdk`
4. Run `python3 -m worker`

Your `worker.py` configures the PyWorker proxy that routes requests to the backend API server.

