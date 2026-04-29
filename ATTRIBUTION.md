# Attribution

## Data Sources

**Supreme Court Database (SCDB)**
- Source: scdb.wustl.edu
- Spaeth, Harold J., Lee Epstein, et al. 2025 Supreme Court Database, Version 2025 Release 01
- Used for all justice vote data, case metadata, issue areas, and decision directions
- License: Public domain

**Oyez Project**
- Source: oyez.org / api.oyez.org
- Case descriptions and legal questions fetched via public Oyez API
- Used to enrich case text representations for embedding
- License: Public domain

**Martin-Quinn Ideology Scores**
- Source: mqscores.lsa.umich.edu
- Martin, Andrew D. and Kevin M. Quinn. 2002. "Dynamic Ideal Point Estimation via Markov Chain Monte Carlo for the U.S. Supreme Court,
  1953-1999." Political Analysis 10(2):134-153
- Used as justice ideology features

**Justia Supreme Court Biographies**
- Source: supreme.justia.com/justices/
- Used as source material for justice biographical profiles in the RAG system

## Justice Profiles
Justice biographical profiles in the RAG system were compiled with assistance from Claude (Anthropic). 
Source material drawn from Justia Supreme Court Biographies (supreme.justia.com/justices/), 
Oyez justice profiles (oyez.org), and publicly available legal scholarship. 
Claude assisted in synthesizing and writing the profile text based on these sources. 
Initial judicial philosophy summaries were drafted by Claude based on its training knowledge of Supreme Court jurisprudence, 
including publicly known Martin-Quinn ideology scores, landmark opinions, and each justice's documented judicial philosophy.

## AI Tools Used
Claude (Anthropic) was used for a few purposes in this project:
1. **Debugging assistance** — Claude was occasionally used for debugging and suggesting possible frameworks for API usage
   throughout development.
3. **Justice profiles** — biographical and judicial philosophy profiles were drafted with Claude's assistance using public source material
4. **Runtime inference and RAG generation** — Claude (claude-sonnet-4-5) serves two roles at runtime:
   (1) extracting structured legal features from case descriptions (issue area, ideological direction, plaintiff alignment)
   (2) acting as the generator in the RAG pipeline — receiving retrieved justice profiles as context and generating 2-3 sentence explanations
       for each predicted vote.

## Libraries and Frameworks
- PyTorch — neural network training and inference
- Sentence Transformers (all-mpnet-base-v2) — case text embeddings
- Scikit-learn — preprocessing, evaluation metrics, StratifiedKFold
- Streamlit — web application framework
- Anthropic Python SDK — Claude API integration
- Pandas, NumPy — data processing
- Joblib — model serialization
- Requests — Oyez API data collection
