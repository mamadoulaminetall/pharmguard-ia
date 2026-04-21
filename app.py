"""
PharmGuard IA — Agent de détection d'ordonnances à risque
MedFlow AI · Dr. Mamadou Lamine TALL
"""

import streamlit as st
import anthropic
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmGuard IA — Détection Ordonnances",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", st.secrets.get("ANTHROPIC_API_KEY", "")))

# ─────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_drugs_db():
    return pd.read_csv("data/drugs_db.csv")

@st.cache_data
def load_history():
    df = pd.read_csv("data/dispensing_history.csv")
    df["date_dispensation"] = pd.to_datetime(df["date_dispensation"])
    return df

DRUGS_DB  = load_drugs_db()
HISTORY   = load_history()

# ─────────────────────────────────────────────────────────────────
# CSS GLOBAL
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
[data-testid="stAppViewContainer"] { background: #020c1e; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
body { font-family: 'Inter', sans-serif; }
iframe[height="0"] { display: none !important; }

/* ── HEADER ── */
.pg-header {
    background: linear-gradient(160deg, #020c1e 0%, #040f1e 100%);
    border-bottom: 1px solid rgba(239,68,68,0.2);
    padding: 20px 48px;
    display: flex; align-items: center; justify-content: space-between;
}
.pg-logo { display: flex; align-items: center; gap: 14px; }
.pg-logo-icon {
    width: 48px; height: 48px; border-radius: 12px;
    background: linear-gradient(135deg, #ef4444, #dc2626);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 24px rgba(239,68,68,0.4);
}
.pg-logo-title { font-size: 1.4rem; font-weight: 800; color: #f1f5f9; letter-spacing: -0.5px; }
.pg-logo-sub { font-size: 0.72rem; color: #ef4444; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; }
.pg-badge {
    background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3);
    color: #ef4444; border-radius: 100px; padding: 6px 16px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 1.5px;
}

/* ── MAIN LAYOUT ── */
.pg-main { padding: 32px 48px; }
.pg-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.pg-card {
    background: #0a1628; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 28px;
}
.pg-card-title {
    font-size: 0.78rem; font-weight: 700; color: #64748b;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 20px;
    display: flex; align-items: center; gap: 8px;
}

/* ── FORM ── */
.pg-input-label {
    font-size: 0.8rem; font-weight: 600; color: #94a3b8; margin-bottom: 6px;
}
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stDateInput > div > div > input {
    background: #060e1e !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stSelectbox > div > div > div:focus {
    border-color: rgba(239,68,68,0.4) !important;
    box-shadow: 0 0 0 3px rgba(239,68,68,0.1) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 14px 32px !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    width: 100% !important; cursor: pointer !important;
    box-shadow: 0 4px 24px rgba(239,68,68,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(239,68,68,0.45) !important;
}

/* ── SCORE GAUGE ── */
.score-gauge {
    text-align: center; padding: 32px 24px;
    background: #060e1e; border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 20px;
}
.score-number { font-size: 4rem; font-weight: 900; line-height: 1; }
.score-label { font-size: 0.85rem; color: #64748b; margin-top: 8px; }
.score-verdict {
    font-size: 1.1rem; font-weight: 700; margin-top: 16px;
    padding: 10px 24px; border-radius: 100px; display: inline-block;
}

/* ── RISK FLAGS ── */
.flag-item {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 16px; border-radius: 10px; margin-bottom: 8px;
}
.flag-icon { font-size: 16px; flex-shrink: 0; margin-top: 2px; }
.flag-text { font-size: 0.82rem; color: #cbd5e1; line-height: 1.5; }
.flag-title { font-weight: 700; margin-bottom: 2px; }

/* ── HISTORY TABLE ── */
.hist-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; border-radius: 8px; margin-bottom: 6px;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04);
    font-size: 0.78rem;
}
.hist-drug { color: #f1f5f9; font-weight: 600; }
.hist-date { color: #64748b; }
.hist-dose { color: #94a3b8; }

/* ── THINKING ── */
.thinking-box {
    background: rgba(6,182,212,0.05); border: 1px solid rgba(6,182,212,0.15);
    border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;
}
.thinking-step {
    font-size: 0.78rem; color: #64748b; font-family: monospace;
    padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.thinking-step:last-child { border-bottom: none; }
.thinking-step.active { color: #06b6d4; }

/* ── AGENT REPORT ── */
.report-box {
    background: #060e1e; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 24px; font-size: 0.85rem;
    color: #94a3b8; line-height: 1.8;
}
.report-box h3 { color: #f1f5f9; font-size: 1rem; margin: 16px 0 8px; }
.report-box strong { color: #f1f5f9; }

/* ── DRUG PILLS ── */
.drug-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 100px; margin: 3px;
    font-size: 0.75rem; font-weight: 600;
}
.drug-high  { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); color: #f87171; }
.drug-med   { background: rgba(249,115,22,0.15); border: 1px solid rgba(249,115,22,0.4); color: #fb923c; }
.drug-low   { background: rgba(16,185,129,0.1);  border: 1px solid rgba(16,185,129,0.3); color: #34d399; }

/* ── DEMO BANNER ── */
.demo-banner {
    background: linear-gradient(90deg, rgba(249,115,22,0.1), rgba(239,68,68,0.1));
    border: 1px solid rgba(249,115,22,0.3);
    border-radius: 10px; padding: 12px 20px;
    font-size: 0.8rem; color: #fb923c;
    margin-bottom: 24px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# AGENT TOOLS (fonctions Python)
# ─────────────────────────────────────────────────────────────────

def classifier_medicament(nom_molecule: str) -> dict:
    """Cherche le médicament dans la base et retourne sa classification."""
    nom_lower = nom_molecule.lower().strip()
    row = DRUGS_DB[
        DRUGS_DB["nom_molecule"].str.lower().str.contains(nom_lower, na=False) |
        DRUGS_DB["dci"].str.lower().str.contains(nom_lower, na=False)
    ]
    if row.empty:
        return {"trouve": False, "molecule": nom_molecule, "message": "Médicament non trouvé dans la base"}
    r = row.iloc[0]
    return {
        "trouve": True,
        "molecule": r["nom_molecule"],
        "dci": r["dci"],
        "classe": r["classe"],
        "sous_classe": r["sous_classe"],
        "niveau_controle": int(r["niveau_controle"]),
        "potentiel_abus": r["potentiel_abus"],
        "dose_max_jour_mg": float(r["dose_max_jour_mg"]),
        "dose_alerte_mg": float(r["dose_alerte_mg"]),
        "ordonnance_securisee": r["ordonnance_securisee"]
    }


def verifier_historique_patient(nom_patient: str, prenom_patient: str = "") -> dict:
    """Vérifie l'historique de dispensation des 90 derniers jours."""
    cutoff = datetime.now() - timedelta(days=90)
    mask = HISTORY["nom"].str.upper().str.contains(nom_patient.upper(), na=False)
    if prenom_patient:
        mask &= HISTORY["prenom"].str.upper().str.contains(prenom_patient.upper(), na=False)
    df = HISTORY[mask & (HISTORY["date_dispensation"] >= cutoff)].copy()

    if df.empty:
        return {"patient_trouve": False, "message": "Aucun historique dans les 90 derniers jours"}

    prescripteurs_uniques = df["prescripteur_nom"].nunique()
    pharmacies_uniques = df["pharmacie_id"].nunique()
    molecules_controlees = df[df["molecule"].isin(
        DRUGS_DB[DRUGS_DB["potentiel_abus"] == "oui"]["nom_molecule"].tolist()
    )]["molecule"].tolist()

    records = []
    for _, row in df.sort_values("date_dispensation", ascending=False).iterrows():
        records.append({
            "date": row["date_dispensation"].strftime("%d/%m/%Y"),
            "molecule": row["molecule"],
            "dose_mg": row["dose_mg"],
            "boites": row["quantite_boites"],
            "prescripteur": row["prescripteur_nom"],
            "specialite": row["prescripteur_specialite"],
            "pharmacie": row["pharmacie_id"],
        })

    return {
        "patient_trouve": True,
        "nom": f"{df.iloc[0]['prenom']} {df.iloc[0]['nom']}",
        "nb_dispensations_90j": len(df),
        "prescripteurs_differents": prescripteurs_uniques,
        "pharmacies_differentes": pharmacies_uniques,
        "molecules_controlees_recentes": list(set(molecules_controlees)),
        "historique": records[:10],
        "alerte_doctor_shopping": prescripteurs_uniques >= 3 or pharmacies_uniques >= 3,
    }


def detecter_interactions(molecules: list) -> dict:
    """Détecte les interactions médicamenteuses dangereuses connues."""
    INTERACTIONS = {
        frozenset(["morphine", "diazepam"]): {"gravite": "MAJEURE", "description": "Association opioïde + BZD : risque de dépression respiratoire fatale"},
        frozenset(["morphine", "alprazolam"]): {"gravite": "MAJEURE", "description": "Association opioïde + BZD : dépression respiratoire"},
        frozenset(["morphine", "zolpidem"]): {"gravite": "MAJEURE", "description": "Association opioïde + hypnotique : sédation excessive"},
        frozenset(["oxycodone", "diazepam"]): {"gravite": "MAJEURE", "description": "Association opioïde fort + BZD : risque vital"},
        frozenset(["oxycodone", "alprazolam"]): {"gravite": "MAJEURE", "description": "Association opioïde fort + BZD : risque vital"},
        frozenset(["tramadol", "sertraline"]): {"gravite": "MODÉRÉE", "description": "Syndrome sérotoninergique possible"},
        frozenset(["tramadol", "escitalopram"]): {"gravite": "MODÉRÉE", "description": "Syndrome sérotoninergique possible"},
        frozenset(["tramadol", "venlafaxine"]): {"gravite": "MODÉRÉE", "description": "Syndrome sérotoninergique — risque élevé"},
        frozenset(["warfarine", "ibuprofene"]): {"gravite": "MAJEURE", "description": "AINS + AVK : risque hémorragique majeur"},
        frozenset(["warfarine", "aspirine"]): {"gravite": "MAJEURE", "description": "Double antiagrégant/anticoagulant : hémorragie"},
        frozenset(["lithium", "ibuprofene"]): {"gravite": "MODÉRÉE", "description": "AINS augmente la lithiémie : toxicité"},
        frozenset(["metformine", "alcool"]): {"gravite": "MODÉRÉE", "description": "Risque d'acidose lactique"},
        frozenset(["diazepam", "alprazolam"]): {"gravite": "MODÉRÉE", "description": "Double prescription BZD : sédation additive"},
        frozenset(["diazepam", "lorazepam"]): {"gravite": "MODÉRÉE", "description": "Double prescription BZD : risque de surdosage"},
        frozenset(["alprazolam", "lorazepam"]): {"gravite": "MODÉRÉE", "description": "Double prescription BZD"},
        frozenset(["codeine", "tramadol"]): {"gravite": "MODÉRÉE", "description": "Association d'opioïdes : accumulation"},
        frozenset(["morphine", "tramadol"]): {"gravite": "MODÉRÉE", "description": "Association d'opioïdes : risque de surdosage"},
    }
    mols_lower = [m.lower() for m in molecules]
    detected = []
    for pair, info in INTERACTIONS.items():
        pair_list = list(pair)
        if any(p in mols_lower for p in [pair_list[0]]) and any(p in mols_lower for p in [pair_list[1]]):
            detected.append({
                "molecules": pair_list,
                "gravite": info["gravite"],
                "description": info["description"]
            })
    return {
        "nb_interactions": len(detected),
        "interactions": detected,
        "alerte_majeure": any(i["gravite"] == "MAJEURE" for i in detected)
    }


def analyser_prescripteur(specialite: str, molecules: list) -> dict:
    """Vérifie la cohérence entre spécialité du prescripteur et médicaments prescrits."""
    HIGH_RISK_DRUGS = ["morphine", "oxycodone", "fentanyl", "methylphenidate", "methadone", "buprenorphine", "ketamine"]
    SPECIALITES_AUTORISEES_OPIOIDES_FORTS = ["oncologie", "douleur", "soins palliatifs", "anesthésie", "neurologie"]

    flags = []
    mols_lower = [m.lower() for m in molecules]
    spec_lower = specialite.lower()

    opioïdes_forts = [m for m in mols_lower if m in ["morphine", "oxycodone", "fentanyl"]]
    if opioïdes_forts and not any(s in spec_lower for s in SPECIALITES_AUTORISEES_OPIOIDES_FORTS):
        flags.append(f"Opioïde(s) fort(s) ({', '.join(opioïdes_forts)}) prescrit(s) par un {specialite} — spécialité inhabituelle")

    if "methylphenidate" in mols_lower and "pédiatrie" not in spec_lower and "psychiatrie" not in spec_lower and "neurologie" not in spec_lower:
        flags.append(f"Méthylphénidate prescrit par un {specialite} — habituellement pédiatre/psychiatre/neurologue")

    if "methadone" in mols_lower and "addictologie" not in spec_lower and "médecine générale" not in spec_lower:
        flags.append(f"Méthadone prescrite par un {specialite} — prescription limitée en France")

    bzd_count = sum(1 for m in mols_lower if m in ["diazepam", "alprazolam", "lorazepam", "clonazepam", "oxazepam", "bromazepam", "zolpidem", "zopiclone"])
    if bzd_count >= 2:
        flags.append(f"Prescription simultanée de {bzd_count} BZD/hypnotiques — association déconseillée")

    return {
        "specialite": specialite,
        "flags_prescripteur": flags,
        "nb_flags": len(flags),
        "prescripteur_coherent": len(flags) == 0
    }


# ─────────────────────────────────────────────────────────────────
# OUTILS ANTHROPIC (schéma)
# ─────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "classifier_medicament",
        "description": "Classe un médicament : classe thérapeutique, niveau de contrôle (1-4), potentiel d'abus, dose maximale journalière. Utiliser pour chaque médicament de l'ordonnance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "nom_molecule": {"type": "string", "description": "Nom du médicament ou DCI"}
            },
            "required": ["nom_molecule"]
        }
    },
    {
        "name": "verifier_historique_patient",
        "description": "Consulte l'historique de dispensation du patient sur les 90 derniers jours. Détecte le doctor shopping, les multi-pharmacies, la fréquence de dispensation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "nom_patient": {"type": "string", "description": "Nom de famille du patient"},
                "prenom_patient": {"type": "string", "description": "Prénom du patient (optionnel)"}
            },
            "required": ["nom_patient"]
        }
    },
    {
        "name": "detecter_interactions",
        "description": "Détecte les interactions médicamenteuses dangereuses entre les médicaments de l'ordonnance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "molecules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Liste des médicaments de l'ordonnance"
                }
            },
            "required": ["molecules"]
        }
    },
    {
        "name": "analyser_prescripteur",
        "description": "Vérifie la cohérence entre la spécialité du prescripteur et les médicaments prescrits. Détecte les prescriptions hors spécialité.",
        "input_schema": {
            "type": "object",
            "properties": {
                "specialite": {"type": "string", "description": "Spécialité médicale du prescripteur"},
                "molecules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Liste des médicaments prescrits"
                }
            },
            "required": ["specialite", "molecules"]
        }
    }
]


def run_tool(name: str, inputs: dict) -> str:
    if name == "classifier_medicament":
        return json.dumps(classifier_medicament(**inputs), ensure_ascii=False)
    if name == "verifier_historique_patient":
        return json.dumps(verifier_historique_patient(**inputs), ensure_ascii=False)
    if name == "detecter_interactions":
        return json.dumps(detecter_interactions(**inputs), ensure_ascii=False)
    if name == "analyser_prescripteur":
        return json.dumps(analyser_prescripteur(**inputs), ensure_ascii=False)
    return json.dumps({"erreur": f"Outil inconnu: {name}"})


# ─────────────────────────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es PharmGuard IA, un agent expert en pharmacovigilance pour les pharmacies de ville françaises.

Ton rôle : analyser une ordonnance et détecter les risques en utilisant les outils disponibles.

Processus obligatoire :
1. Classer CHAQUE médicament avec classifier_medicament()
2. Vérifier l'historique du patient avec verifier_historique_patient()
3. Détecter les interactions avec detecter_interactions()
4. Analyser le prescripteur avec analyser_prescripteur()
5. Synthétiser et calculer un score de risque

Score de risque (0-100) :
- 0-25 : Ordonnance normale ✅
- 26-50 : Vigilance recommandée ⚠️
- 51-75 : Risque élevé — Vérification approfondie requise 🔶
- 76-100 : Très suspect — Refus de dispensation recommandé 🔴

Format de réponse final (JSON strict) :
{
  "score": <0-100>,
  "verdict": "NORMAL|VIGILANCE|RISQUE_ÉLEVÉ|TRÈS_SUSPECT",
  "decision": "DISPENSER|VIGILANCE|VÉRIFIER_AVANT_DISPENSATION|REFUS_RECOMMANDÉ",
  "flags": [
    {"niveau": "CRITIQUE|ÉLEVÉ|MODÉRÉ|INFO", "message": "...", "detail": "..."}
  ],
  "synthese": "Paragraphe de synthèse clinique en français, professionnel, 3-5 phrases",
  "actions_recommandees": ["action 1", "action 2"]
}

Sois précis, professionnel, et factuel. Ne juge pas, tu fournis une aide à la décision."""


def analyze_prescription(patient_info: dict, medicaments: list, prescripteur: dict, thinking_placeholder) -> dict:
    """Lance l'agent et retourne le résultat final."""

    meds_str = "\n".join([
        f"- {m['nom']} {m['dose']}mg — {m['quantite']} boîte(s)"
        for m in medicaments
    ])

    user_message = f"""Analyse cette ordonnance :

PATIENT : {patient_info['nom']} {patient_info['prenom']}, né(e) le {patient_info['ddn']}

PRESCRIPTEUR : {prescripteur['nom']} — {prescripteur['specialite']}

MÉDICAMENTS :
{meds_str}

Date de l'ordonnance : {datetime.now().strftime('%d/%m/%Y')}

Analyse complète requise avec tous les outils disponibles."""

    messages = [{"role": "user", "content": user_message}]
    thinking_steps = []

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        # Afficher les tool calls en cours
        tool_calls_in_turn = [b for b in response.content if b.type == "tool_use"]
        for tc in tool_calls_in_turn:
            step = f"🔧 {tc.name}({json.dumps(tc.input, ensure_ascii=False)[:80]}...)"
            thinking_steps.append(step)
            thinking_placeholder.markdown(
                '<div class="thinking-box">' +
                "".join(f'<div class="thinking-step {"active" if i == len(thinking_steps)-1 else ""}">{s}</div>'
                        for i, s in enumerate(thinking_steps)) +
                '</div>',
                unsafe_allow_html=True
            )

        if response.stop_reason == "end_turn":
            # Extraire le JSON final
            for block in response.content:
                if hasattr(block, "text"):
                    text = block.text.strip()
                    try:
                        start = text.find("{")
                        end = text.rfind("}") + 1
                        if start >= 0:
                            return json.loads(text[start:end])
                    except:
                        pass
            return {"score": 0, "verdict": "ERREUR", "decision": "VÉRIFIER MANUELLEMENT", "flags": [], "synthese": "Erreur d'analyse", "actions_recommandees": []}

        # Exécuter les outils
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                result_parsed = json.loads(result)
                step_done = f"✅ {block.name} → {str(result_parsed)[:100]}"
                thinking_steps[-len([b for b in response.content if b.type == "tool_use"])+
                              [b for b in response.content if b.type == "tool_use"].index(block)] = step_done
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


# ─────────────────────────────────────────────────────────────────
# HELPERS UI
# ─────────────────────────────────────────────────────────────────
def score_color(score: int) -> str:
    if score <= 25: return "#10b981"
    if score <= 50: return "#f59e0b"
    if score <= 75: return "#f97316"
    return "#ef4444"

def flag_style(niveau: str) -> tuple:
    styles = {
        "CRITIQUE": ("🔴", "rgba(239,68,68,0.1)",  "rgba(239,68,68,0.2)"),
        "ÉLEVÉ":    ("🟠", "rgba(249,115,22,0.1)", "rgba(249,115,22,0.2)"),
        "MODÉRÉ":   ("🟡", "rgba(245,158,11,0.1)", "rgba(245,158,11,0.2)"),
        "INFO":     ("🔵", "rgba(59,130,246,0.08)", "rgba(59,130,246,0.15)"),
    }
    return styles.get(niveau, ("⚪", "rgba(255,255,255,0.04)", "rgba(255,255,255,0.08)"))


# ─────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="pg-header">
  <div class="pg-logo">
    <div class="pg-logo-icon">🛡️</div>
    <div>
      <div class="pg-logo-title">PharmGuard IA</div>
      <div class="pg-logo-sub">Détection ordonnances · Agent IA · MedFlow AI</div>
    </div>
  </div>
  <div class="pg-badge">⚡ DÉMO · v1.0</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="pg-main">', unsafe_allow_html=True)

st.markdown("""
<div class="demo-banner">
  ⚠️ <strong>Mode démonstration</strong> — Données patients fictives · Ne pas utiliser pour des décisions médicales réelles
</div>
""", unsafe_allow_html=True)

# Layout 2 colonnes
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.markdown('<div class="pg-card">', unsafe_allow_html=True)
    st.markdown('<div class="pg-card-title">📋 Saisie de l\'ordonnance</div>', unsafe_allow_html=True)

    # Patient
    st.markdown("**Patient**")
    c1, c2 = st.columns(2)
    with c1:
        patient_nom = st.text_input("Nom", value="ROUX", key="p_nom", label_visibility="collapsed", placeholder="Nom de famille")
    with c2:
        patient_prenom = st.text_input("Prénom", value="Sophie", key="p_prenom", label_visibility="collapsed", placeholder="Prénom")
    patient_ddn = st.text_input("Date de naissance", value="15/05/1990", key="p_ddn", placeholder="JJ/MM/AAAA")

    st.divider()

    # Prescripteur
    st.markdown("**Prescripteur**")
    c3, c4 = st.columns(2)
    with c3:
        presc_nom = st.text_input("Nom prescripteur", value="Dr. Dupont", key="pr_nom", label_visibility="collapsed", placeholder="Dr. Nom")
    with c4:
        presc_spec = st.selectbox("Spécialité", [
            "médecine générale", "psychiatrie", "neurologie", "oncologie",
            "addictologie", "anesthésie", "soins palliatifs", "douleur",
            "pédiatrie", "cardiologie", "endocrinologie", "autre"
        ], key="pr_spec", label_visibility="collapsed")

    st.divider()

    # Médicaments
    st.markdown("**Médicaments prescrits**")

    if "meds" not in st.session_state:
        st.session_state.meds = [
            {"nom": "alprazolam", "dose": "4", "quantite": "3"},
            {"nom": "zolpidem", "dose": "10", "quantite": "1"},
        ]

    for i, med in enumerate(st.session_state.meds):
        cm1, cm2, cm3, cm4 = st.columns([3, 1.5, 1.5, 0.8])
        with cm1:
            st.session_state.meds[i]["nom"] = st.text_input(f"med_{i}_nom", value=med["nom"], label_visibility="collapsed", placeholder="Médicament / DCI", key=f"mn{i}")
        with cm2:
            st.session_state.meds[i]["dose"] = st.text_input(f"med_{i}_dose", value=med["dose"], label_visibility="collapsed", placeholder="Dose (mg)", key=f"md{i}")
        with cm3:
            st.session_state.meds[i]["quantite"] = st.text_input(f"med_{i}_qt", value=med["quantite"], label_visibility="collapsed", placeholder="Boîtes", key=f"mq{i}")
        with cm4:
            if st.button("✕", key=f"del{i}") and len(st.session_state.meds) > 1:
                st.session_state.meds.pop(i)
                st.rerun()

    if st.button("＋ Ajouter un médicament", key="add_med"):
        st.session_state.meds.append({"nom": "", "dose": "", "quantite": "1"})
        st.rerun()

    st.markdown("")
    analyze_btn = st.button("🔍 Analyser l'ordonnance", key="analyze")
    st.markdown('</div>', unsafe_allow_html=True)

    # Cas démo rapides
    st.markdown('<div class="pg-card" style="margin-top:16px">', unsafe_allow_html=True)
    st.markdown('<div class="pg-card-title">⚡ Cas démonstration rapides</div>', unsafe_allow_html=True)

    demo_cases = {
        "🔴 Doctor shopping BZD": {
            "nom": "ROUX", "prenom": "Sophie", "ddn": "15/05/1990",
            "presc_nom": "Dr. Dupont", "presc_spec": "médecine générale",
            "meds": [{"nom": "alprazolam", "dose": "4", "quantite": "3"},
                     {"nom": "zolpidem", "dose": "10", "quantite": "1"}]
        },
        "🔴 Opioïdes hors spécialité": {
            "nom": "SIMON", "prenom": "Claire", "ddn": "03/12/1995",
            "presc_nom": "Dr. Lefebvre", "presc_spec": "médecine générale",
            "meds": [{"nom": "oxycodone", "dose": "40", "quantite": "2"},
                     {"nom": "diazepam", "dose": "20", "quantite": "1"}]
        },
        "✅ Ordonnance normale": {
            "nom": "LEROY", "prenom": "Anne", "ddn": "19/01/1972",
            "presc_nom": "Dr. Dupont", "presc_spec": "médecine générale",
            "meds": [{"nom": "metformine", "dose": "2000", "quantite": "3"},
                     {"nom": "amlodipine", "dose": "10", "quantite": "1"}]
        },
        "🟠 Interaction AVK + AINS": {
            "nom": "NGUYEN", "prenom": "Van", "ddn": "17/08/1960",
            "presc_nom": "Dr. Simon", "presc_spec": "médecine générale",
            "meds": [{"nom": "warfarine", "dose": "7.5", "quantite": "2"},
                     {"nom": "ibuprofene", "dose": "1200", "quantite": "1"}]
        }
    }

    for case_name, case_data in demo_cases.items():
        if st.button(case_name, key=f"demo_{case_name}"):
            st.session_state["demo_load"] = case_data
            st.rerun()

    if "demo_load" in st.session_state:
        d = st.session_state.pop("demo_load")
        st.session_state.meds = d["meds"]
        # Update patient/presc via query params workaround — just show message
        st.success(f"Cas chargé : modifiez les champs si besoin puis cliquez Analyser")

    st.markdown('</div>', unsafe_allow_html=True)


with col_result:
    st.markdown('<div class="pg-card" style="min-height:600px">', unsafe_allow_html=True)
    st.markdown('<div class="pg-card-title">🤖 Analyse Agent IA</div>', unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#334155">
          <div style="font-size:3rem;margin-bottom:16px">🛡️</div>
          <div style="font-size:1rem;font-weight:600;color:#475569">Prêt à analyser</div>
          <div style="font-size:0.82rem;margin-top:8px;color:#334155">
            Remplissez le formulaire et cliquez sur<br><strong style="color:#ef4444">Analyser l'ordonnance</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if analyze_btn:
        thinking_ph = st.empty()

        meds_clean = [m for m in st.session_state.meds if m["nom"].strip()]
        if not meds_clean:
            st.error("Ajoutez au moins un médicament.")
        else:
            with st.spinner(""):
                thinking_ph.markdown('<div class="thinking-box"><div class="thinking-step active">🤖 Agent PharmGuard IA démarré...</div></div>', unsafe_allow_html=True)

                result = analyze_prescription(
                    patient_info={"nom": patient_nom, "prenom": patient_prenom, "ddn": patient_ddn},
                    medicaments=meds_clean,
                    prescripteur={"nom": presc_nom, "specialite": presc_spec},
                    thinking_placeholder=thinking_ph
                )
                st.session_state.result = result

            thinking_ph.empty()
            st.rerun()

    if "result" in st.session_state:
        r = st.session_state.result
        score = r.get("score", 0)
        color = score_color(score)

        # Score gauge
        verdict_labels = {
            "NORMAL": ("✅ Ordonnance normale", "rgba(16,185,129,0.15)", "#10b981"),
            "VIGILANCE": ("⚠️ Vigilance recommandée", "rgba(245,158,11,0.15)", "#f59e0b"),
            "RISQUE_ÉLEVÉ": ("🔶 Risque élevé", "rgba(249,115,22,0.15)", "#f97316"),
            "TRÈS_SUSPECT": ("🔴 Très suspect", "rgba(239,68,68,0.15)", "#ef4444"),
        }
        verdict = r.get("verdict", "NORMAL")
        vl, vbg, vc = verdict_labels.get(verdict, verdict_labels["NORMAL"])

        st.markdown(f"""
        <div class="score-gauge">
          <div class="score-number" style="color:{color}">{score}<span style="font-size:1.5rem;color:#475569">/100</span></div>
          <div class="score-label">Score de risque PharmGuard</div>
          <div class="score-verdict" style="background:{vbg};color:{vc};border:1px solid {vc}44">{vl}</div>
          <div style="font-size:0.82rem;color:#64748b;margin-top:12px;font-weight:600">{r.get('decision','')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Flags
        flags = r.get("flags", [])
        if flags:
            st.markdown(f'<div style="font-size:0.78rem;font-weight:700;color:#64748b;letter-spacing:1.5px;margin-bottom:10px">⚠️ SIGNAUX DÉTECTÉS ({len(flags)})</div>', unsafe_allow_html=True)
            for flag in flags:
                icon, bg, border = flag_style(flag.get("niveau", "INFO"))
                st.markdown(f"""
                <div class="flag-item" style="background:{bg};border:1px solid {border}">
                  <div class="flag-icon">{icon}</div>
                  <div class="flag-text">
                    <div class="flag-title">{flag.get('message','')}</div>
                    <div style="color:#64748b;font-size:0.78rem">{flag.get('detail','')}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Synthèse
        synthese = r.get("synthese", "")
        if synthese:
            st.markdown(f"""
            <div class="report-box" style="margin-top:16px">
              <div style="font-size:0.72rem;font-weight:700;color:#64748b;letter-spacing:1.5px;margin-bottom:10px">📝 SYNTHÈSE CLINIQUE</div>
              {synthese}
            </div>
            """, unsafe_allow_html=True)

        # Actions
        actions = r.get("actions_recommandees", [])
        if actions:
            st.markdown(f"""
            <div class="report-box" style="margin-top:12px">
              <div style="font-size:0.72rem;font-weight:700;color:#64748b;letter-spacing:1.5px;margin-bottom:10px">✅ ACTIONS RECOMMANDÉES</div>
              {''.join(f'<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);color:#94a3b8">→ {a}</div>' for a in actions)}
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔄 Nouvelle analyse", key="reset"):
            del st.session_state.result
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;padding:24px;border-top:1px solid rgba(255,255,255,0.05);
     margin-top:16px;font-size:0.72rem;color:#334155">
  PharmGuard IA · MedFlow AI · Dr. M.L. TALL · Outil d'aide à la décision — Ne remplace pas le jugement professionnel
</div>
""", unsafe_allow_html=True)
