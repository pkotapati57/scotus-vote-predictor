# SCOTUS VOTE PREDICTOR -- Streamlit App Deplyoyment (Includes Predictor)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import anthropic

# ── PAGE CONFIG ───────────────────────────────────────────────

st.set_page_config(
    page_title="SCOTUS Vote Predictor",
    page_icon="⚖️",
    layout="wide"
)

# ── CONSTANTS ─────────────────────────────────────────────────

CURRENT_JUSTICES = [
    'JGRoberts', 'CThomas', 'SAAlito', 'SSotomayor',
    'EKagan', 'NMGorsuch', 'BMKavanaugh', 'ACBarrett', 'KBJackson'
]

JUSTICE_NAMES = {
    'JGRoberts':  'Chief Justice John Roberts',
    'CThomas':    'Justice Clarence Thomas',
    'SAAlito':    'Justice Samuel Alito',
    'SSotomayor': 'Justice Sonia Sotomayor',
    'EKagan':     'Justice Elena Kagan',
    'NMGorsuch':  'Justice Neil Gorsuch',
    'BMKavanaugh':'Justice Brett Kavanaugh',
    'ACBarrett':  'Justice Amy Coney Barrett',
    'KBJackson':  'Justice Ketanji Brown Jackson'
}

JUSTICE_PHOTOS = {
    'JGRoberts':  'https://fedsoc.org/cdn-cgi/image/fit=scale-down,width=560,format=jpeg/https://fedsoc-cms-public.s3.amazonaws.com/headshots/XJRy5KtszNRQ8A2PDtKgP82zSzzSrpMhckFHBcuZ.jpeg',
    'CThomas':    'https://ca-times.brightspotcdn.com/dims4/default/eeb55bb/2147483647/strip/true/crop/2598x2629+0+0/resize/1200x1214!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2F92%2Fc9%2Feb0c32094790be86b889eeafaef7%2Fap101008146934.jpg',
    'SAAlito':    'https://hips.hearstapps.com/hmg-prod/images/united-states-supreme-court-associate-justice-samuel-alito-news-photo-1718112586.jpg?crop=0.642xw:0.963xh;0.260xw,0.0366xh&resize=1200:*',
    'SSotomayor': 'https://constitutionallawreporter.com/wp-content/uploads/2016/10/sonia-sotomayor.jpg',
    'EKagan':     'https://www.womenshistory.org/sites/default/files/images/2022-10/Elena-Kagan-square.jpg',
    'NMGorsuch':  'https://npr.brightspotcdn.com/dims4/default/59db74f/2147483647/strip/true/crop/300x400+0+0/resize/880x1173!/quality/90/?url=http%3A%2F%2Fnpr-brightspot.s3.amazonaws.com%2Flegacy%2Fwp-content%2Fuploads%2F2019%2F03%2FNeil-Gorsuch-Chip-Somodevilla-Getty-Images.jpg',
    'BMKavanaugh':'https://www.rollingstone.com/wp-content/uploads/2018/09/angry-kavanaugh.jpg?w=1581&h=1054&crop=1',
    'ACBarrett':  'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Official_Amy_Barrett_photo.jpg/960px-Official_Amy_Barrett_photo.jpg',
    'KBJackson':  'https://upload.wikimedia.org/wikipedia/commons/d/d6/Ketanji_Brown_Jackson_official_SCOTUS_portrait.jpg'
}

ISSUE_AREAS = {
    1: 'Criminal Procedure', 2: 'Civil Rights', 3: 'First Amendment',
    4: 'Due Process', 5: 'Privacy', 6: 'Attorneys', 7: 'Unions',
    8: 'Economic Activity', 9: 'Judicial Power', 10: 'Federalism',
    11: 'Interstate Relations', 12: 'Federal Taxation',
    13: 'Miscellaneous', 14: 'Private Action'
}

# ── MODEL DEFINITIONS ─────────────────────────────────────────

class SCOTUSDataset(torch.utils.data.Dataset):
    def __init__(self, jf, cf, ji, labels):
        self.jf = torch.FloatTensor(jf)
        self.cf = torch.FloatTensor(cf)
        self.ji = torch.LongTensor(ji)
        self.labels = torch.FloatTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.jf[idx], self.cf[idx], self.ji[idx], self.labels[idx]

class SCOTUSModel(nn.Module):
    def __init__(self, justice_input_dim, case_input_dim,
                 n_justices=9, embedding_dim=64):
        super().__init__()
        self.case_branch = nn.Sequential(
            nn.Linear(case_input_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, embedding_dim), nn.ReLU()
        )
        self.justice_branch = nn.Sequential(
            nn.Linear(justice_input_dim, 32), nn.BatchNorm1d(32),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, embedding_dim), nn.ReLU()
        )
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2)
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid()
            ) for _ in range(n_justices)
        ])

    def forward(self, jf, cf, ji):
        case_emb    = self.case_branch(cf)
        justice_emb = self.justice_branch(jf)
        combined    = torch.cat([case_emb, justice_emb], dim=1)
        shared      = self.shared(combined)
        outputs = torch.zeros(len(ji))
        for j in range(9):
            mask = (ji == j)
            if mask.sum() > 0:
                outputs[mask] = self.heads[j](shared[mask]).squeeze(1)
        return outputs

# ── LOAD RESOURCES ────────────────────────────────────────────

@st.cache_resource
def load_all():
    pca            = joblib.load('models/pca.pkl')
    justice_scaler = joblib.load('models/justice_scaler_v5.pkl')
    embed_model    = SentenceTransformer('all-mpnet-base-v2')

    with open('models/justice_feature_cols_v5.pkl', 'rb') as f:
        JUSTICE_FEATURE_COLS = pickle.load(f)

    model = SCOTUSModel(
        justice_input_dim=len(JUSTICE_FEATURE_COLS),
        case_input_dim=67,
        n_justices=9,
        embedding_dim=64
    )
    model.load_state_dict(
        torch.load('models/scotus_model_v5.pt', map_location='cpu'))
    model.eval()

    df = pd.read_parquet('models/scotus_df_clean_v5.parquet')

    with open('models/combined_profiles.pkl', 'rb') as f:
        profiles = pickle.load(f)

    return model, pca, justice_scaler, embed_model, df, profiles, JUSTICE_FEATURE_COLS

# ── GROQ CLIENT ───────────────────────────────────────────────

def get_client():
    return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ── LLM FUNCTIONS ─────────────────────────────────────────────

# ── UPDATED EXTRACT CASE FEATURES ────────────────────────────
# Add plaintiff/defendant to extraction

def extract_case_features(case_description, plaintiff, defendant, client):
    prompt = f"""You are an expert in U.S. Supreme Court jurisprudence with deep knowledge of constitutional law, SCOTUS voting patterns, 
                and the ideological leanings of the current justices.
                
                The Supreme Court has 6 conservative justices (Roberts, Thomas, Alito, Gorsuch, Kavanaugh, Barrett) and 3 liberal justices (Sotomayor, Kagan, Jackson).

                CRITICAL DISTINCTION: decision_direction refers to the ideological direction of the COURT'S RULING, not the politics of the law being challenged.
                - Conservative outcome (1): court expands gun rights, supports religion in public life, limits abortion, restricts voting rights expansions, 
                limits federal regulatory power, supports law enforcement, upholds executive power for Republican presidents
                - Liberal outcome (2): court expands civil rights, protects abortion access, expands voting rights, upholds federal regulatory power, 
                restricts gun laws, separates church and state

                EXAMPLES:
                - City bans guns → court strikes down ban → decision_direction=1 (CONSERVATIVE, expanding gun rights)
                - State mandates school prayer → court strikes it down → decision_direction=2 (LIBERAL, separating church and state)
                - Christian baker refuses gay wedding cake → court sides with baker → decision_direction=1 (CONSERVATIVE, religious liberty)
                - EPA regulates carbon emissions → court limits EPA power → decision_direction=1 (CONSERVATIVE, limiting regulatory power)

                Plaintiff: "{plaintiff}"
                Defendant: "{defendant}"
                Case: "{case_description}"

                Carefully analyze who wins and what the ideological direction of that outcome is.

                Return ONLY valid JSON:
                {{
                    "issue_area_code": integer 1-14 where 1=Criminal Procedure, 2=Civil Rights,
                        3=First Amendment, 4=Due Process, 5=Privacy, 6=Attorneys, 7=Unions,
                        8=Economic Activity, 9=Judicial Power, 10=Federalism,
                        11=Interstate Relations, 12=Federal Taxation, 13=Miscellaneous, 14=Private Action,
                    "decision_direction": 1 for conservative outcome, 2 for liberal outcome,
                    "estimated_controversy": float 0.0 to 1.0,
                    "plaintiff_wins_if_liberal": true if plaintiff wins when liberals prevail,
                                                false if plaintiff wins when conservatives prevail,
                    "reasoning": "one sentence explaining your determination"
                }}"""
    try:
        r = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = r.content[0].text.strip()
        raw = raw.replace('```json', '').replace('```', '').strip()
        data = json.loads(raw)
        print(f"LLM reasoning: {data.get('reasoning', 'none')}")
        data['plaintiff_wins_if_conservative'] = not data.get(
            'plaintiff_wins_if_liberal', True)
        return data
    except Exception as e:
        print(f"Extraction error: {e}")
        return {"issue_area_code": 13, "decision_direction": 3,
                "estimated_controversy": 0.5,
                "plaintiff_wins_if_conservative": False}

def get_vote_label(vote, plaintiff, defendant, plaintiff_wins_if_conservative):
    """Convert model vote (1=majority/conservative, 0=dissent/liberal) to plain English."""
    # vote=1 means justice voted with historical majority (tends conservative)
    # vote=0 means justice voted against majority (tends liberal)
    if plaintiff_wins_if_conservative:
        winner = plaintiff if vote == 1 else defendant
    else:
        winner = defendant if vote == 1 else plaintiff
    return f"For {winner}"

def generate_justice_explanation(justice_code, justice_name,
                                  case_description, vote_label,
                                  confidence, profile, client):
    prompt = f"""You are a legal analyst. The Supreme Court prediction model has determined that 
                {justice_name} will vote {vote_label} in this case. Your job is to explain WHY 
                this makes sense given their judicial philosophy.

                Case: {case_description}
                Confidence: {confidence:.1%}

                Justice's background and judicial philosophy:
                {profile[:800]}

                Write 2-3 sentences explaining why {justice_name} would vote {vote_label}, 
                drawing specifically on their ideology, key past opinions, and judicial philosophy. 
                Be confident and direct. Never express doubt about the prediction or suggest it 
                might be wrong. Never say the prediction appears incorrect or inconsistent. Never 
                say the prediction 'appears inconsistent' or 'appears incorrect'. Never say
                I must respectfully note'. Present the reasoning as if it is correct, even if 
                it feels wrong to."""
    try:
        r = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return r.content[0].text.strip()
    except Exception as e:
        print(f"Explanation error: {e}")
        return f"Explanation unavailable: {e}"

# PREDICTION PIPELINE

def predict_all_justices(case_description, case_features, model, pca, 
                          justice_scaler, embed_model, df, JUSTICE_FEATURE_COLS):
    
    # Embed case description
    case_emb_raw     = embed_model.encode([case_description])[0].reshape(1, -1)
    case_emb_reduced = pca.transform(case_emb_raw)  # (1, 64)
    
    # Add direction features from LLM extraction
    direction = case_features.get('decision_direction', 3)
    extra = np.array([[
        1.0 if direction == 1 else 0.0,  # conservative
        1.0 if direction == 2 else 0.0,  # liberal
        0.5                               # unknown margin
    ]])

    case_emb_final = np.hstack([case_emb_reduced, extra])  # (1, 67)

    results = {}

    for j_idx, j_code in enumerate(CURRENT_JUSTICES):
        jdf = df[df['justiceName'] == j_code].sort_values(
            'term', ascending=False)
        if len(jdf) == 0:
            continue
        row = jdf.iloc[0]

        feats = []
        for col in JUSTICE_FEATURE_COLS:
            try:
                feats.append(float(row[col]))
            except:
                feats.append(0.0)

        X_justice        = np.array(feats).reshape(1, -1)
        X_justice_scaled = justice_scaler.transform(X_justice)

        with torch.no_grad():
            jf    = torch.FloatTensor(X_justice_scaled)
            cf    = torch.FloatTensor(case_emb_final)
            ji    = torch.LongTensor([j_idx])
            proba = model(jf, cf, ji).item()

        results[j_code] = {
            'proba':      proba,
            'vote':       1 if proba > 0.5 else 0,
            'confidence': proba if proba > 0.5 else 1 - proba
        }

    return results

# User Interface

st.title("⚖️ SCOTUS Vote Predictor")
st.markdown("Predict how each Supreme Court Justice will vote on a case "
            "using a neural multi-output model and AI-powered explanations.")

st.divider()

col1, col2 = st.columns(2)
with col1:
    plaintiff = st.text_input("Plaintiff", placeholder="e.g. Group of parents")
with col2:
    defendant = st.text_input("Defendant", placeholder="e.g. State of Texas")

case_description = st.text_area(
    "Describe the case",
    placeholder="e.g. A state law banning all abortions after 6 weeks...",
    height=150
)

predict_btn = st.button(
    "Predict All 9 Votes 🗳️", type="primary", use_container_width=True)

if predict_btn:
    if not case_description or not plaintiff or not defendant:
        st.warning("Please fill in plaintiff, defendant, and case description.")
    else:
        model, pca, justice_scaler, embed_model, df, profiles, JUSTICE_FEATURE_COLS = load_all()
        client = get_client()

        with st.spinner("Analyzing case and predicting votes..."):
            case_features = extract_case_features(
                case_description, plaintiff, defendant, client)
            predictions   = predict_all_justices(
                case_description, case_features, model, pca, justice_scaler,
                embed_model, df, JUSTICE_FEATURE_COLS)

        # Add vote labels
        plaintiff_wins_if_conservative = case_features.get(
            'plaintiff_wins_if_conservative', True)
        
        for j_code in predictions:
            vote = predictions[j_code]['vote']
            predictions[j_code]['vote_label'] = get_vote_label(
                vote, plaintiff, defendant, plaintiff_wins_if_conservative)

        # Overall result
        plaintiff_votes = sum(
            1 for r in predictions.values() 
            if plaintiff in r['vote_label'])
        defendant_votes = 9 - plaintiff_votes
        
        overall_winner = plaintiff if plaintiff_votes >= 5 else defendant
        overall = f"✅ {overall_winner} WINS"

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Decision", overall)
        col2.metric("Vote", f"{max(plaintiff_votes, defendant_votes)}-{min(plaintiff_votes, defendant_votes)}")
        col3.metric("Issue Area",
                    ISSUE_AREAS.get(case_features.get('issue_area_code', 13),
                                   'Miscellaneous'))

        st.divider()
        st.subheader("Individual Justice Predictions")

        with st.spinner("Generating explanations..."):
            explanations = {}
            for j_code in CURRENT_JUSTICES:
                if j_code in predictions:
                    pred = predictions[j_code]
                    explanations[j_code] = generate_justice_explanation(
                        j_code,
                        JUSTICE_NAMES[j_code],
                        case_description,
                        pred['vote_label'],
                        pred['confidence'],
                        profiles.get(j_code, ''),
                        client
                    )

        # Justice cards
        cols = st.columns(3)
        for i, j_code in enumerate(CURRENT_JUSTICES):
            if j_code not in predictions:
                continue
            pred = predictions[j_code]
            is_winner = pred['vote_label'].startswith(f"For {overall_winner}")
            vote_color = "green" if is_winner else "red"

            with cols[i % 3]:
                st.image(JUSTICE_PHOTOS[j_code], width=120)
                st.markdown(f"**{JUSTICE_NAMES[j_code]}**")
                st.markdown(f":{vote_color}[**{pred['vote_label']}**]")
                st.caption(f"Confidence: {pred['confidence']:.1%}")
                with st.expander("Why?"):
                    st.write(explanations.get(j_code, 'N/A'))
                st.divider()

st.caption("Built with Multi-Output Neural Network + RAG + Llama 3.3 70B | "
           "Data: Supreme Court Database (1991-2024)")

# The End :(