# Setup Guide

## Prerequisites
- Python 3.9+
- An Anthropic API key (get one at console.anthropic.com)

## Live App
No setup required — the app is deployed at: https://scotus-vote-predictor-az.streamlit.app/

## Run Locally

1. Clone the repository:
```bash
git clone https://github.com/pkotapati57/scotus-vote-predictor
cd scotus-vote-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your Anthropic API key — create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "your_key_here"
```

4. Run the app:
```bash
streamlit run app.py
```

## Model Files
All model files are included in the `models/` directory. No additional downloads required.

## Reproducing Training:
To retrain the model from scratch:
1. Download the Supreme Court Database (SCDB) justice-centered CSV from scdb.wustl.edu
2. Upload to Google Colab
3. Run `SCOTUS_Project_Data_Pipeline_+_Define_Models_+_Training.ipynb` top to bottom
4. Run `SCOTUS_Project_RAG.ipynb` to rebuild justice profiles
5. Download model files and place in `models/`

## Requirements
All dependencies are listed in `requirements.txt` and install automatically via pip.
