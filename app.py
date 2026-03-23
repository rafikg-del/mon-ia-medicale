"""
MedFiche AI — Triple-Agent Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent 1 : Gemini 1.5 Pro   — Extraction OCR + analyse des schémas
Agent 2 : DeepSeek-R1      — Physiopathologie causale (Chain-of-Thought)
Agent 3 : Grok-3 (xAI)    — Synthèse finale 10 points + LaTeX + Chat
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from typing import Callable, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
XAI_API_KEY      = os.getenv("XAI_API_KEY")

GEMINI_MODEL   = "gemini-1.5-pro-latest"
DEEPSEEK_MODEL = "deepseek-reasoner"          # DeepSeek-R1
GROK_MODEL     = "grok-3"                     # xAI Grok-3

MAX_DOC_CHARS  = 150_000
CHAT_CTX_CHARS =  60_000

# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """\
Tu es un agent d'extraction médicale de haut niveau (Gemini 1.5 Pro).
Exploite ta grande fenêtre de contexte pour analyser ce document de façon EXHAUSTIVE.

Extrais et retourne un JSON structuré avec :
1. "titre"             : titre ou thème principal du cours
2. "donnees_brutes"    : tous les faits, chiffres, valeurs normales/pathologiques, statistiques
3. "schemas_causaux"   : cascades mécanistiques complètes (A → B → C → conséquence clinique)
4. "entites_medicales" : pathologies, médicaments (DCI + posologie si mentionnée), examens, traitements
5. "sequences"         : évolution temporelle, stades, délais diagnostiques
6. "points_critiques"  : complications, urgences, contre-indications absolues/relatives
7. "mnémotechniques"   : acronymes, règles mémo, classifications
8. "algorithmes"       : arbres décisionnels, procédures étape par étape

DOCUMENT :
---
{document_text}
---

Retourne UNIQUEMENT le JSON. Sois exhaustif — chaque détail alimentera deux autres agents IA.
"""

PHYSIOPATH_PROMPT = """\
Tu es DeepSeek-R1, un modèle de raisonnement avancé avec Chain-of-Thought profond.
Ta mission EXCLUSIVE : produire la section "Physiopathologie CAUSALE" la plus rigoureuse possible.

DONNÉES EXTRAITES PAR GEMINI :
```json
{extracted_data}
```

Raisonne étape par étape (tu peux montrer ton processus de réflexion).
Puis génère la section Physiopathologie en respectant cette structure :

## ⚡ PHYSIOPATHOLOGIE CAUSALE

### 1. Mécanisme déclencheur initial
[Explique le trigger primaire avec précision moléculaire/cellulaire]

### 2. Cascade physiopathologique complète
Utilise des flèches causales pour chaque maillon :
- **Étape 1 :** [Trigger] → [Mécanisme moléculaire] → [Effet cellulaire]
- **Étape 2 :** [Effet cellulaire] → [Réponse tissulaire] → [Conséquence organique]
- **Étape N :** [...] → [...] → [Manifestation clinique finale]

### 3. Schéma logique
```
[Cause primaire]
      ↓
[Mécanisme A] ──────────────→ [Conséquence latérale A1]
      ↓                                    ↓
[Mécanisme B]                    [Amplification A2]
      ↓
[Manifestation clinique terminale]
```

### 4. Bottlenecks mécanistiques clés
| Bottleneck | Mécanisme d'amplification | Cible thérapeutique |
|------------|--------------------------|---------------------|
| ...        | ...                      | ...                 |

### 5. Corrélations anatomo-cliniques
[Lier chaque mécanisme à son expression clinique observable]

Utilise LaTeX pour les constantes : $K_a$, $\Delta G$, $Ca^{2+}$, $HCO_3^-$, etc.
Pour les équations chimiques : $$CO_2 + H_2O \rightleftharpoons H^+ + HCO_3^-$$
"""

SYNTHESIS_PROMPT = """\
Tu es Grok-3 (xAI), expert en synthèse médicale et rédaction scientifique de précision.
Tu reçois le travail de deux agents IA spécialisés. Ta mission : produire la fiche médicale finale parfaite.

━━━ ENTRÉE AGENT 1 — Extraction Gemini ━━━
```json
{extracted_data}
```

━━━ ENTRÉE AGENT 2 — Physiopathologie DeepSeek-R1 ━━━
{physiopath_analysis}

━━━ TA MISSION ━━━
Synthétise ces deux sources en une fiche médicale complète, rigoureuse et pédagogique.
La section Physiopathologie est DÉJÀ rédigée par DeepSeek — intègre-la telle quelle (section 1).
Complète les 9 autres sections avec précision.

RÈGLES LATEX OBLIGATOIRES :
- Constantes et valeurs → inline : $K_a = 10^{{-7}}$, $T_{{1/2}}$, $\Delta G < 0$
- Ions dans le texte : $Ca^{{2+}}$, $Na^+$, $Cl^-$, $HCO_3^-$, $O_2$, $CO_2$
- Équations chimiques → display : $$CO_2 + H_2O \\rightleftharpoons H^+ + HCO_3^-$$
- Formules de calcul : $$\\text{{Clairance}} = \\frac{{U \\times V}}{{P}}$$

━━━ STRUCTURE OBLIGATOIRE EN 10 POINTS ━━━

# {title}

> *Résumé essentiel en 2-3 phrases — ce qu'il faut retenir absolument.*

---

## 1. ⚡ PHYSIOPATHOLOGIE CAUSALE
[INTÈGRE ICI LE CONTENU COMPLET DE DEEPSEEK-R1 SANS MODIFICATION]

---

## 2. 🔬 BOTTLENECKS MÉCANISTIQUES
*(Si déjà dans la section 1, enrichis avec les implications thérapeutiques)*

| Bottleneck | Mécanisme | Conséquence si non bloqué | Intervention |
|------------|-----------|--------------------------|-------------|
| ... | ... | ... | ... |

---

## 3. 🩺 MANIFESTATIONS CLINIQUES
**Très fréquents (> 50%) :**
- ...

**Fréquents (10–50%) :**
- ...

**Rares mais critiques (< 10%) :**
- ...

---

## 4. ⚖️ DIAGNOSTIC DIFFÉRENTIEL

| Pathologie | Arguments POUR | Arguments CONTRE | Examen discriminant |
|------------|----------------|-----------------|---------------------|
| ... | ... | ... | ... |

---

## 5. 🔍 EXAMENS COMPLÉMENTAIRES

**En urgence :**
- ...

**Confirmation diagnostique :**
- ...

**Bilan de sévérité / suivi :**
- ...

---

## 6. 💊 TRAITEMENT ÉTIOLOGIQUE
*(Traiter la cause — DCI, posologie, durée)*
- ...

---

## 7. 🩹 TRAITEMENT SYMPTOMATIQUE
*(Traiter les symptômes)*
- ...

---

## 8. ⚠️ COMPLICATIONS
> 🚨 POINTS DE VIGILANCE CRITIQUES

| Complication | Facteurs de risque | Signes d'alarme | Conduite à tenir |
|-------------|-------------------|-----------------|-----------------|
| ... | ... | ... | ... |

**Contre-indications absolues :** ...

---

## 9. 📊 PRONOSTIC
*(Données chiffrées : survie, récidive, guérison — cite les sources si disponibles)*
- ...

---

## 10. 🧠 ALGORITHME DÉCISIONNEL

```
PRÉSENTATION CLINIQUE INITIALE
            ↓
    [Critère diagnostic A ?]
    ├── OUI ──→ [Examen B]
    │              ├── Positif ──→ Diagnostic confirmé → Traitement X
    │              └── Négatif ──→ Diagnostic différentiel Y
    └── NON ──→ [Critère C ?]
                   ├── OUI ──→ Bilan complémentaire → Traitement Z
                   └── NON ──→ Surveillance / Réévaluation à J7
```

---

## 💡 POINTS CLÉS / MNÉMOTECHNIQUES

**À retenir absolument :**
- ...

**Mnémotechnique :**
- ...

---

Rédige la fiche COMPLÈTE. Chaque "..." doit être remplacé par du vrai contenu médical.
"""

GROK_CHAT_SYSTEM = """\
Tu es Grok-3 (xAI), assistant médical intégré à MedFiche AI.
Tu as accès au contenu suivant pour répondre aux questions de l'utilisateur.

{fiche_block}

{context_block}

Règles :
- Si l'utilisateur demande un point précis de la fiche (ex: "explique le point 3"), \
cite directement la section correspondante.
- Complète avec le cours source si la fiche ne suffit pas.
- Si une information est absente des deux sources, dis-le clairement.
- Réponds en français, de façon précise et pédagogique.
- Utilise LaTeX quand tu cites des constantes ou des équations.
"""

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf(path: str) -> str:
    import fitz
    doc = fitz.open(path)
    out = ""
    for i, page in enumerate(doc):
        out += f"\n--- Page {i+1} ---\n{page.get_text()}"
    doc.close()
    return out

def extract_pptx(path: str) -> str:
    from pptx import Presentation
    prs = Presentation(path)
    out = ""
    for i, slide in enumerate(prs.slides):
        out += f"\n--- Slide {i+1} ---\n"
        if slide.shapes.title:
            out += f"TITRE: {slide.shapes.title.text}\n"
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip() and shape != slide.shapes.title:
                out += shape.text + "\n"
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                out += f"[NOTES]: {notes}\n"
    return out

def extract_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    out = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    for table in doc.tables:
        out += "\n[TABLEAU]\n"
        for row in table.rows:
            out += " | ".join(c.text.strip() for c in row.cells) + "\n"
    return out

def transcribe_video(uploaded_file, model_size: str = "base") -> str:
    import whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    try:
        model = whisper.load_model(model_size)
        return model.transcribe(path, language="fr", verbose=False)["text"]
    finally:
        os.unlink(path)

def extract_document(uploaded_file, file_type: str, whisper_size: str = "base") -> str:
    if file_type == "mp4":
        return transcribe_video(uploaded_file, model_size=whisper_size)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    try:
        return {"pdf": extract_pdf, "pptx": extract_pptx, "docx": extract_docx}[file_type](path)
    finally:
        os.unlink(path)

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — GEMINI 1.5 PRO (Extraction)
# ══════════════════════════════════════════════════════════════════════════════

class GeminiAgent:
    def __init__(self):
        import google.generativeai as genai
        genai.configure(
            api_key=GOOGLE_API_KEY,
            client_options={"api_endpoint": "generativelanguage.googleapis.com"},
        )
        # Passe le nom sans préfixe — la lib ajoute "models/" automatiquement
        self.model = genai.GenerativeModel(model_name=GEMINI_MODEL)

    def extract(self, document_text: str) -> str:
        prompt = EXTRACTION_PROMPT.format(document_text=document_text)
        response = self.model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 8192, "temperature": 0.2},
        )
        return response.text

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — DEEPSEEK-R1 (Physiopathologie — Chain-of-Thought)
# ══════════════════════════════════════════════════════════════════════════════

class DeepSeekAgent:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )

    def reason_physiopath(self, extracted_data: str) -> tuple[str, str]:
        """
        Returns (reasoning_trace, physiopath_content).
        reasoning_trace : le Chain-of-Thought interne de DeepSeek-R1 (si disponible)
        physiopath_content : la section rédigée
        """
        prompt = PHYSIOPATH_PROMPT.format(extracted_data=extracted_data)
        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
        )
        msg = response.choices[0].message
        # DeepSeek-R1 expose le raisonnement dans reasoning_content (si dispo)
        reasoning = getattr(msg, "reasoning_content", "") or ""
        content   = msg.content or ""
        return reasoning, content

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — GROK-3 / xAI (Synthèse finale + Chat)
# ══════════════════════════════════════════════════════════════════════════════

class GrokAgent:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )
        self._messages: list[dict] = []
        self._system: str = ""

    def synthesize(self, extracted_data: str, physiopath_analysis: str, title: str = "Cours médical") -> str:
        prompt = SYNTHESIS_PROMPT.format(
            extracted_data=extracted_data,
            physiopath_analysis=physiopath_analysis,
            title=title,
        )
        response = self.client.chat.completions.create(
            model=GROK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
        )
        return response.choices[0].message.content

    # ── Chat ────────────────────────────────────────────────────────────────

    def initialize_chat(self, context: str = "", fiche: str = "") -> None:
        self._messages = []
        fiche_block = (
            f"=== FICHE MÉDICALE FINALE (référence prioritaire) ===\n{fiche}\n=== FIN ==="
            if fiche else "Aucune fiche générée pour l'instant."
        )
        context_block = (
            f"=== COURS SOURCE (complément) ===\n{context[:CHAT_CTX_CHARS]}\n=== FIN ==="
            if context else ""
        )
        self._system = GROK_CHAT_SYSTEM.format(
            fiche_block=fiche_block,
            context_block=context_block,
        )

    def chat(self, user_message: str, context: str = "", fiche: str = "") -> str:
        if not self._system:
            self.initialize_chat(context, fiche)
        self._messages.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(
            model=GROK_MODEL,
            messages=[{"role": "system", "content": self._system}] + self._messages,
            max_tokens=4096,
        )
        reply = response.choices[0].message.content
        self._messages.append({"role": "assistant", "content": reply})
        return reply

    def reset_chat(self) -> None:
        self._messages = []
        self._system   = ""

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE TRIPLE-AGENT
# ══════════════════════════════════════════════════════════════════════════════

def run_triple_pipeline(
    document_text: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Séquence :
      1. Gemini  → extraction JSON exhaustive
      2. DeepSeek-R1 → physiopathologie causale (CoT)
      3. Grok-3  → synthèse fiche complète 10 points
    """
    def _p(v: float, msg: str) -> None:
        if progress_callback:
            progress_callback(v, msg)

    # ── Agent 1 : Gemini ──────────────────────────────────────────────────
    _p(0.05, "🔍 Agent 1 — Gemini 1.5 Pro extrait le corpus et analyse les schémas...")
    gemini = GeminiAgent()
    extracted_data = gemini.extract(document_text)
    _p(0.28, "✅ Agent 1 terminé — Données structurées prêtes.")

    # ── Agent 2 : DeepSeek-R1 ────────────────────────────────────────────
    _p(0.32, "🧠 Agent 2 — DeepSeek-R1 raisonne sur la physiopathologie causale...")
    deepseek = DeepSeekAgent()
    reasoning_trace, physiopath = deepseek.reason_physiopath(extracted_data)
    _p(0.62, "✅ Agent 2 terminé — Physiopathologie causale rédigée.")

    # ── Titre extrait pour Grok ──────────────────────────────────────────
    import json, re as _re
    title = "Cours médical"
    try:
        raw = _re.search(r'\{.*\}', extracted_data, _re.DOTALL)
        if raw:
            data = json.loads(raw.group())
            title = data.get("titre", title)
    except Exception:
        pass

    # ── Agent 3 : Grok-3 ─────────────────────────────────────────────────
    _p(0.66, "✨ Agent 3 — Grok-3 synthétise la fiche finale et optimise le LaTeX...")
    grok = GrokAgent()
    final_fiche = grok.synthesize(extracted_data, physiopath, title=title)
    _p(1.0, "🏁 Triplette terminée — Fiche validée par les 3 agents !")

    return {
        "extracted_data"  : extracted_data,
        "reasoning_trace" : reasoning_trace,
        "physiopath"      : physiopath,
        "final_fiche"     : final_fiche,
        "title"           : title,
    }

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_DISPLAY_MATH = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)

def render_with_latex(content: str) -> None:
    """$$...$$ → st.latex()  |  reste → st.markdown() (gère $...$ inline)."""
    for i, part in enumerate(_DISPLAY_MATH.split(content)):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            st.latex(part.strip())

def file_hash(f) -> str:
    f.seek(0); h = hashlib.md5(f.read()).hexdigest(); f.seek(0)
    return h

def ctx_hash(t: str) -> str:
    return str(hash(t[:500])) if t else ""

def fmt_size(n: int) -> str:
    for u in ["B","KB","MB","GB"]:
        if n < 1024: return f"{n:.1f} {u}"
        n //= 1024
    return f"{n:.1f} TB"

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
<style>
/* ── Header ── */
.hdr{background:linear-gradient(135deg,#0a0a23 0%,#1a0533 50%,#001a33 100%);
     padding:26px 32px;border-radius:16px;margin-bottom:24px;
     box-shadow:0 8px 32px rgba(0,0,0,0.5);}
.hdr h1{color:#fff;margin:0;font-size:28px;letter-spacing:-0.5px;}
.hdr .sub{color:rgba(255,255,255,0.6);margin:8px 0 0;font-size:13px;}
.hdr .agents{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;}
.ag{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;
    border-radius:20px;font-size:12px;font-weight:600;color:#fff;}
.ag1{background:linear-gradient(90deg,#1a73e8,#0d47a1);}
.ag2{background:linear-gradient(90deg,#00bcd4,#006064);}
.ag3{background:linear-gradient(90deg,#7c4dff,#311b92);}

/* ── Step box ── */
.step{background:#e8eaf6;border-left:4px solid #3f51b5;padding:10px 16px;
      border-radius:0 8px 8px 0;margin:12px 0;font-size:14px;
      color:#1a237e;font-weight:600;}

/* ── Agent progress cards ── */
.agent-card{border-radius:10px;padding:14px 18px;margin:8px 0;
            display:flex;align-items:center;gap:12px;font-size:14px;}
.agent-card.a1{background:#e3f2fd;border-left:4px solid #1565c0;}
.agent-card.a2{background:#e0f7fa;border-left:4px solid #00838f;}
.agent-card.a3{background:#ede7f6;border-left:4px solid #6a1b9a;}
.agent-card .ico{font-size:22px;}
.agent-card .lbl{font-weight:700;}
.agent-card .sub{font-size:12px;color:#666;margin-top:2px;}

/* ── Fiche badge ── */
.fbadge{display:inline-flex;align-items:center;gap:8px;
        background:linear-gradient(90deg,#1a73e8,#7c4dff);
        color:#fff;padding:6px 16px;border-radius:20px;
        font-size:13px;font-weight:600;margin-bottom:18px;
        box-shadow:0 2px 10px rgba(124,77,255,0.4);}

/* ── Compare labels ── */
.clbl{background:#e8eaf6;border-radius:8px;padding:8px 14px;
      font-weight:700;font-size:13px;color:#3f51b5;
      margin-bottom:10px;text-align:center;}
.clbl.fin{background:#e8f5e9;color:#2e7d32;}

/* ── API warning ── */
.api-warn{background:#fff3e0;border:2px solid #ff9800;border-radius:10px;
          padding:16px 20px;margin:10px 0;font-size:14px;}

/* ── Sidebar ── */
.sb-hdr{background:linear-gradient(135deg,#0a0a23,#1a0533);
        padding:16px 18px;border-radius:12px;margin-bottom:14px;}

#MainMenu,footer{visibility:hidden;}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# UI — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

def render_upload() -> tuple[Optional[str], Optional[str]]:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1565c0,#4a148c);
                padding:20px 25px;border-radius:12px;margin-bottom:20px;">
        <h3 style="color:#fff;margin:0;font-size:20px;">📁 Importer votre cours</h3>
        <p style="color:rgba(255,255,255,0.7);margin:6px 0 0;font-size:13px;">
            PDF · PPTX · DOCX · MP4</p>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Fichier", type=["pdf","pptx","docx","mp4"],
        label_visibility="collapsed",
    )

    if not uploaded:
        st.markdown("""
        <div style="border:2px dashed #90caf9;border-radius:10px;padding:30px;
                    text-align:center;color:#5c6bc0;background:#f8f9ff;">
            <div style="font-size:36px;">📂</div>
            <div style="font-size:15px;margin-top:8px;">Aucun fichier sélectionné</div>
        </div>""", unsafe_allow_html=True)
        return None, None

    ftype = uploaded.name.split(".")[-1].lower()
    st.markdown(f"""
    <div style="background:#e8f5e9;border-left:4px solid #43a047;
                border-radius:0 8px 8px 0;padding:10px 15px;margin:10px 0;
                font-size:14px;color:#1b5e20;">
        📄 <b>{uploaded.name}</b> — {fmt_size(uploaded.size)} — <code>.{ftype}</code>
    </div>""", unsafe_allow_html=True)

    # Whisper selector AVANT le spinner
    whisper_size = "base"
    if ftype == "mp4":
        whisper_size = st.selectbox(
            "Qualité Whisper",
            ["tiny","base","small","medium"], index=1,
            help="tiny=rapide, medium=précis mais lent",
        )

    # Cache par hash — évite toute ré-extraction au rerun
    fh        = file_hash(uploaded)
    cache_key = f"doc_{fh}_{whisper_size}"

    if cache_key in st.session_state:
        text = st.session_state[cache_key]
    else:
        if ftype == "mp4":
            with st.spinner("🎙️ Transcription Whisper en cours..."):
                text = transcribe_video(uploaded, model_size=whisper_size)
        else:
            with st.spinner("Lecture du fichier..."):
                text = extract_document(uploaded, ftype)
        st.session_state[cache_key] = text

    c1, c2, c3 = st.columns(3)
    c1.metric("Caractères", f"{len(text):,}")
    c2.metric("Mots",       f"{len(text.split()):,}")
    c3.metric("Tokens (~)", f"{len(text)//4:,}")

    with st.expander("👁️ Aperçu", expanded=False):
        st.text(text[:3000] + ("..." if len(text) > 3000 else ""))

    return text, uploaded.name

# ══════════════════════════════════════════════════════════════════════════════
# UI — FICHE DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def render_fiche(content: str, badge: bool = True) -> None:
    if badge:
        st.markdown(
            '<div class="fbadge">🤖 Gemini · DeepSeek-R1 · Grok-3 — Triple consensus</div>',
            unsafe_allow_html=True,
        )
    render_with_latex(content)

def render_comparison(draft: str, final: str) -> None:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="clbl">🧠 DeepSeek-R1 Physiopath + brouillon Grok</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            render_with_latex(draft)
    with c2:
        st.markdown('<div class="clbl fin">✅ Fiche Finale — Triple Agent</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            render_with_latex(final)

# ══════════════════════════════════════════════════════════════════════════════
# UI — PROGRESS PANEL
# ══════════════════════════════════════════════════════════════════════════════

def render_agent_progress(active: int) -> None:
    """Affiche 3 cartes agents avec l'agent actif mis en évidence."""
    agents = [
        ("a1", "🔍", "Agent 1 — Gemini 1.5 Pro",    "Extraction OCR + analyse des schémas"),
        ("a2", "🧠", "Agent 2 — DeepSeek-R1",         "Physiopathologie causale (Chain-of-Thought)"),
        ("a3", "✨", "Agent 3 — Grok-3",              "Synthèse finale 10 points + LaTeX"),
    ]
    for i, (cls, ico, lbl, sub) in enumerate(agents, 1):
        status = "⏳ En cours..." if i == active else ("✅ Terminé" if i < active else "⌛ En attente")
        opacity = "1" if i <= active else "0.4"
        st.markdown(f"""
        <div class="agent-card {cls}" style="opacity:{opacity};">
            <div class="ico">{ico}</div>
            <div>
                <div class="lbl">{lbl}</div>
                <div class="sub">{sub} — {status}</div>
            </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# UI — SIDEBAR CHAT (Grok-3)
# ══════════════════════════════════════════════════════════════════════════════

def _init_chat() -> None:
    defaults = {
        "grok_agent"      : None,
        "chat_messages"   : [],
        "chat_doc_hash"   : "",
        "chat_fiche_hash" : "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _get_grok() -> GrokAgent:
    if st.session_state.grok_agent is None:
        st.session_state.grok_agent = GrokAgent()
    return st.session_state.grok_agent

def render_sidebar(document_context: str = "", final_fiche: str = "") -> None:
    _init_chat()

    dh = ctx_hash(document_context)
    fh = ctx_hash(final_fiche)
    if dh != st.session_state.chat_doc_hash or fh != st.session_state.chat_fiche_hash:
        if document_context or final_fiche:
            _get_grok().initialize_chat(document_context, final_fiche)
            st.session_state.chat_doc_hash   = dh
            st.session_state.chat_fiche_hash = fh

    status = (
        "✅ Fiche + cours chargés" if final_fiche
        else ("📄 Cours chargé" if document_context else "Aucun contexte")
    )

    with st.sidebar:
        st.markdown(f"""
        <div class="sb-hdr">
            <div style="color:#fff;font-size:18px;font-weight:700;">
                ✨ Grok-3 — Assistant
            </div>
            <div style="color:rgba(255,255,255,0.6);font-size:12px;margin-top:4px;">
                xAI · Basé sur votre fiche médicale
            </div>
            <div style="color:rgba(255,255,255,0.4);font-size:11px;">{status}</div>
        </div>""", unsafe_allow_html=True)

        if not document_context and not final_fiche:
            st.warning("⚠️ Importez un document pour activer Grok-3.")
        elif not final_fiche:
            st.info("💡 Générez une fiche pour des réponses sur-mesure.")
        else:
            st.success("🎯 Grok-3 a accès à votre fiche médicale.")

        st.divider()

        # Historique
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="background:#e3f2fd;border-radius:10px 10px 2px 10px;
                            padding:10px 13px;margin:6px 0;font-size:13px;color:#0d47a1;">
                    👤 <b>Vous</b><br>{msg['content']}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#ede7f6;border-radius:10px 10px 10px 2px;
                            padding:10px 13px;margin:4px 0;font-size:13px;color:#311b92;">
                    ✨ <b>Grok-3</b>
                </div>""", unsafe_allow_html=True)
                st.markdown(msg["content"])

        # Input
        user_input = st.chat_input("Posez une question sur le cours...")
        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.spinner("Grok-3 réfléchit..."):
                reply = _get_grok().chat(user_input, document_context, final_fiche)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state.chat_messages:
            st.divider()
            if st.button("🗑️ Effacer le chat", use_container_width=True):
                st.session_state.chat_messages   = []
                st.session_state.grok_agent      = None
                st.session_state.chat_doc_hash   = ""
                st.session_state.chat_fiche_hash = ""
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MedFiche AI — Triple Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
    <h1>🏥 MedFiche AI</h1>
    <p class="sub">Fiches médicales par consensus de trois intelligences artificielles spécialisées</p>
    <div class="agents">
        <span class="ag ag1">🔍 Agent 1 — Gemini 1.5 Pro · Extraction</span>
        <span class="ag ag2">🧠 Agent 2 — DeepSeek-R1 · Raisonnement</span>
        <span class="ag ag3">✨ Agent 3 — Grok-3 · Synthèse & Chat</span>
    </div>
</div>""", unsafe_allow_html=True)

# ── API guard ─────────────────────────────────────────────────────────────────
missing = {
    "GOOGLE_API_KEY"  : GOOGLE_API_KEY,
    "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
    "XAI_API_KEY"     : XAI_API_KEY,
}
missing_keys = [k for k, v in missing.items() if not v]
if missing_keys:
    st.markdown(
        '<div class="api-warn">⚠️ <b>Clés API manquantes :</b> '
        + ", ".join(f"<code>{k}</code>" for k in missing_keys)
        + "<br><br>Ajoutez-les dans <b>Streamlit Cloud → Settings → Secrets</b> "
        "ou dans votre fichier <code>.env</code> local.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [
    ("document_text", None), ("file_name", None),
    ("results", None),       ("generating", False),
    ("active_agent", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
_fiche = (st.session_state.results or {}).get("final_fiche", "") or ""
render_sidebar(
    document_context=st.session_state.document_text or "",
    final_fiche=_fiche,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📁 Import & Génération",
    "📋 Fiche Finale",
    "🔬 Détails des Agents",
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════════════════════
with tab1:
    doc_text, file_name = render_upload()

    if doc_text:
        truncated = doc_text[:MAX_DOC_CHARS] + (
            "\n\n[... CONTENU TRONQUÉ ...]" if len(doc_text) > MAX_DOC_CHARS else ""
        )
        st.session_state.document_text = truncated
        st.session_state.file_name     = file_name
        st.markdown(
            f'<div class="step">✅ Document prêt — ~{len(truncated)//4:,} tokens. '
            'Lancez la génération par la Triplette.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.document_text:
        col_btn, _ = st.columns([2, 3])
        with col_btn:
            go = st.button(
                "🚀 Lancer la Triplette Médicale",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.generating,
            )

        if go:
            st.session_state.generating   = True
            st.session_state.active_agent = 1

            progress_bar      = st.progress(0.0)
            status_placeholder = st.empty()
            agents_placeholder = st.empty()

            def _upd(v: float, msg: str) -> None:
                progress_bar.progress(v)
                status_placeholder.markdown(
                    f'<div class="step">{msg}</div>', unsafe_allow_html=True
                )
                # Déduire l'agent actif depuis la progression
                active = 1 if v < 0.30 else (2 if v < 0.65 else 3)
                with agents_placeholder.container():
                    render_agent_progress(active)

            try:
                results = run_triple_pipeline(
                    st.session_state.document_text,
                    progress_callback=_upd,
                )
                st.session_state.results    = results
                st.session_state.generating = False
                status_placeholder.success(
                    "🎉 Triplette terminée ! Consultez l'onglet **Fiche Finale**."
                )
                with agents_placeholder.container():
                    render_agent_progress(4)   # tous terminés
            except Exception as e:
                st.session_state.generating = False
                st.error(f"❌ Erreur pipeline : {e}")

# ════════════════════════════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.results:
        st.info("👆 Importez un document et lancez la **Triplette Médicale**.")
    else:
        res = st.session_state.results
        ct, ctog, cdl = st.columns([3, 2, 1])
        with ct:
            st.markdown(f"## 📋 {res.get('title', 'Fiche Médicale')}")
        with ctog:
            show_cmp = st.toggle("🔀 Physiopath / Fiche Finale", value=False)
        with cdl:
            st.download_button(
                "⬇️ .md",
                data=res["final_fiche"],
                file_name=f"fiche_{st.session_state.file_name or 'medicale'}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        st.divider()
        if show_cmp:
            render_comparison(res["physiopath"], res["final_fiche"])
        else:
            render_fiche(res["final_fiche"], badge=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 3
# ════════════════════════════════════════════════════════════════════════
with tab3:
    if not st.session_state.results:
        st.info("Lancez d'abord la Triplette dans l'onglet 1.")
    else:
        res = st.session_state.results

        st.markdown("## 🔬 Détails des 3 Agents")
        st.markdown("""
        <div style="background:#e8eaf6;border-radius:10px;padding:14px 18px;
                    margin-bottom:20px;font-size:13px;color:#1a237e;">
            <b>Pipeline :</b>
            📄 Document
            → <b>🔍 Gemini 1.5 Pro (extraction JSON)</b>
            → <b>🧠 DeepSeek-R1 (physiopathologie CoT)</b>
            → <b>✨ Grok-3 (synthèse 10 points)</b>
            → ✅ Fiche finale
        </div>""", unsafe_allow_html=True)

        with st.expander("🔍 Agent 1 — Extraction Gemini 1.5 Pro", expanded=False):
            st.caption("Données structurées extraites depuis le document source.")
            st.markdown(res["extracted_data"])

        with st.expander("🧠 Agent 2 — Physiopathologie DeepSeek-R1", expanded=False):
            st.caption("Analyse physiopathologique avec Chain-of-Thought.")
            if res.get("reasoning_trace"):
                with st.expander("🔎 Trace de raisonnement interne (CoT)", expanded=False):
                    st.markdown(f"```\n{res['reasoning_trace'][:5000]}\n```")
            render_with_latex(res["physiopath"])

        with st.expander("✨ Agent 3 — Fiche Finale Grok-3", expanded=True):
            st.caption("Synthèse complète en 10 points avec LaTeX optimisé.")
            render_fiche(res["final_fiche"], badge=True)
