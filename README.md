# scotus-vote-predictor

# ⚖️ U.S. Supreme Court Vote Predictor
A machine learning web application that predicts how each of the nine current U.S. Supreme Court Justices will vote on any legal case, with retrieval-augmented generated explanations grounded in each justice's judicial philosophy and past.

## What it Does
SCOTUS Vote Predictor takes a case description, plaintiff, and defendant as input and runs it through a multi-stage ML pipeline: a two-tower neural network predicts each justice's vote based on case embeddings and justice-specific features, a RAG system retrieves relevant judicial philosophy from curated justice profiles, and Claude (Anthropic) generates 2-3 sentence explanations for each prediction. The app displays the overall court decision, vote count, individual justice predictions with confidence scores, and expandable AI-generated reasoning for each justice.

## Quick Start
1. Visit the live web app: https://scotus-vote-predictor-az.streamlit.app/
2. Enter the plaintiff name, defendant name, and a description of the case
3. Click "Predict All 9 Votes 🗳️"
4. View the overall decision, vote count, and individual justice predictions with explanations

To run locally:
```bash
git clone https://github.com/pkotapati57/scotus-vote-predictor
cd scotus-vote-predictor
pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY = "your_key_here"' > .streamlit/secrets.toml
streamlit run app.py
```
*Note that an Anthropic API Key is required to run the model locally. 
The web-deployed app already has one in use, and runs on a finite number of tokens.


## Video Links
- Demo video: https://youtu.be/JOoeuajyhIE
- Technical walkthrough: https://youtu.be/7g3VODJ80F4

## Evaluation
The model achieves the following on the held-out test set (StratifiedKFold, 20% test):

| Metric | Score |
|--------|-------|
| Accuracy | 91.9% |
| F1 Macro | 88.0% |
| ROC AUC | 96.7% |

Per-justice accuracy:

| Justice | Accuracy | Test Cases |
|---------|----------|------------|
| Chief Justice John Roberts | 92.4% | 278 |
| Justice Clarence Thomas | 93.0% | 458 |
| Justice Samuel Alito | 91.5% | 246 |
| Justice Sonia Sotomayor | 94.1% | 202 |
| Justice Elena Kagan | 93.8% | 192 |
| Justice Neil Gorsuch | 69.8% | 86 |
| Justice Brett Kavanaugh | 93.4% | 76 |
| Justice Amy Coney Barrett | 96.2% | 53 |
| Justice Ketanji Brown Jackson | 96.4% | 28 |

The model performs best on cases with clear ideological valence — gun rights, abortion, religious liberty, and affirmative action. It struggles on administrative law and separation of powers cases where voting patterns are less ideologically consistent. Justice Gorsuch is the most difficult to predict (69.8%) due to his independent voting streak — he occasionally joins liberals on Fourth Amendment, criminal procedure, and Native American rights cases. Justice Jackson's high test accuracy (96.4%) should be interpreted cautiously given she only appears in 28 test cases, the fewest of any justice — as one of the Court's newest members, limited historical data means her predictions may be less reliable in practice.

While individual justice leanings are occasionally misclassified, the model is generally accurate in its overall vote outcome — correctly predicting which side wins and by what margin on most ideologically clear cases. The model does not account for recusals; if a justice recuses themselves from a case due to a conflict of interest, the model will still generate a prediction for them.

  **Ablation Study Results:**
  
  | Version | Split Strategy | Features | Oyez Coverage | F1 Macro | ROC AUC |
  |---------|---------------|----------|---------------|----------|---------|
  | v1 | Temporal | 7 (no direction) | 10% | 0.585 | 0.751 |
  | v3 | Temporal | 7 + direction | 10% | 0.783 | 0.916 |
  | v4 | StratifiedKFold | 8 + direction + bias | 10% | 0.874 | 0.965 |
  | v5 | StratifiedKFold | 8 + direction + bias | 69% | 0.880 | 0.967 |

  **Comparison of Architectures:**
  
  The most impactful change across all versions was switching from a temporal split to StratifiedKFold. The temporal split systematically disadvantaged newer justices — Barrett and Jackson joined the court in 2020 and 2022 respectively, meaning almost all their cases fell in the test set with virtually no training examples. StratifiedKFold fixed this by ensuring proportional justice representation across all folds, which is why Barrett improved from 58% to 96% and Jackson from 29% to 96%.
    
  Adding direction features (v1→v3) was the second most impactful change. It gave the model explicit signal about whether a case historically produced a conservative or liberal outcome, rather than inferring ideology purely from case text embeddings. I further refined this with the justice_issue_bias feature (v3→v4), which aims to teach the model how each individual justice deviates from the court average on each issue area: e.g. capturing Gorsuch's tendency to vote more liberally than other conservatives on criminal procedure. Finally, improving Oyez API coverage from 10% to 69% (v4→v5) gave the model richer semantic case representations, replacing sparse fallback text with actual case descriptions and legal questions for the majority of cases.

## Individual Contributions
Solo project — all data collection, model training, RAG pipeline, and deployment by Prithvi Kotapati.
