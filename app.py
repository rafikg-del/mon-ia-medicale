"""
MedFiche AI — Single-file Streamlit app for Streamlit Cloud deployment.
Architecture : Gemini 1.5 Pro (extraction + critique) × Claude Opus 4.6 (rédaction + révision)
Consensus en 2 tours. Sidebar chat hybride persistant.
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

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL      = "gemini-1.5-pro"
CLAUDE_MODEL      = "claude-opus-4-6"
MAX_DOC_CHARS     = 150_000
CHAT_CTX_CHARS    = 60_000

# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """
Tu es un agent d'extraction médicale expert. Analyse ce document de manière exhaustive.

Extrais et structure en JSON :
1. **donnees_brutes** : faits, chiffres, valeurs normales/pathologiques, statistiques
2. **schemas_causaux** : cascades physiopathologiques complètes (A → B → C → conséquence)
3. **entites_medicales** : pathologies, médicaments (DCI + posologie), examens, traitements
4. **sequences_temporelles** : évolution, délais, progression par stades
5. **points_critiques** : complications, urgences, contre-indications, vigilance
6. **mnémotechniques** : acronymes, règles mémo, classifications
7. **algorithmes** : arbres décisionnels, procédures étape par étape

Document :
---
{document_text}
---

Retourne un JSON structuré complet. Sois EXHAUSTIF.
"""

CRITIQUE_PROMPT = """
Tu es un agent de validation médicale. Audite cette fiche médicale par rapport aux données sources.

**DONNÉES SOURCES :**
```json
{extracted_data}
```

**FICHE À CRITIQUER :**
---
{fiche_content}
---

Produis une critique structurée :

## OMISSIONS CRITIQUES
(informations présentes dans les sources mais absentes de la fiche + section cible)

## IMPRÉCISIONS
(citation du passage imprécis + reformulation correcte)

## ERREURS FACTUELLES
(erreur exacte + correction sourcée)

## AMÉLIORATIONS SUGGÉRÉES
(schémas causaux manquants, mnémotechniques oubliés, etc.)
"""

FICHE_DRAFT_PROMPT = """
Tu es Claude, expert en rédaction médicale structurée.
Rédige une fiche médicale complète à partir des données extraites ci-dessous.

**RÈGLES DE NOTATION MATHÉMATIQUE ET CHIMIQUE (obligatoires) :**
- Constantes et valeurs → LaTeX inline `$...$` : $K_a = 10^{{-7}}$, $T_{{1/2}} = 6{{,}}5\\ \\text{{h}}$
- Équations chimiques → LaTeX display `$$...$$` :
  $$CO_2 + H_2O \\rightleftharpoons H^+ + HCO_3^-$$
- Ions dans le texte : $Ca^{{2+}}$, $Na^+$, $HCO_3^-$, $O_2$, $CO_2$
- Formules de calcul : $$\\text{{Clairance}} = \\frac{{U \\times V}}{{P}}$$

**DONNÉES SOURCES EXTRAITES PAR GEMINI :**
```json
{extracted_data}
```

---

## STRUCTURE OBLIGATOIRE EN 10 POINTS

⚠️ La section **Physiopathologie Causale** doit être la PLUS DENSE de toute la fiche.

---

# [TITRE : Pathologie / Thème du cours]

> *Résumé en 2-3 phrases de l'essentiel.*

---

## 1. ⚡ PHYSIOPATHOLOGIE CAUSALE
*(Section principale — développe TOUTE la cascade mécanistique)*

### Mécanisme déclencheur
- ...

### Cascade physiopathologique
- **Étape 1 :** [Trigger] → [Mécanisme moléculaire] → [Effet cellulaire]
- **Étape 2 :** [Effet cellulaire] → [Réponse tissulaire] → [Manifestation clinique]

### Schéma logique
```
[Cause primaire]
      ↓
[Mécanisme A] ──→ [Conséquence A1]
      ↓                    ↓
[Mécanisme B]       [Conséquence A2]
      ↓
[Manifestation clinique finale]
```

---

## 2. 🔬 BOTTLENECKS MÉCANISTIQUES

| Bottleneck | Mécanisme | Conséquence si non bloqué | Intervention thérapeutique |
|------------|-----------|--------------------------|---------------------------|
| ... | ... | ... | ... |

---

## 3. 🩺 MANIFESTATIONS CLINIQUES

**Très fréquents (> 50%) :** ...
**Fréquents (10–50%) :** ...
**Rares mais importants (< 10%) :** ...

---

## 4. ⚖️ DIAGNOSTIC DIFFÉRENTIEL

| Pathologie | Argument POUR | Argument CONTRE | Examen discriminant |
|------------|---------------|-----------------|---------------------|
| ... | ... | ... | ... |

---

## 5. 🔍 EXAMENS COMPLÉMENTAIRES

**En urgence :** ...
**Confirmation diagnostique :** ...
**Bilan de sévérité / suivi :** ...

---

## 6. 💊 TRAITEMENT ÉTIOLOGIQUE
...

---

## 7. 🩹 TRAITEMENT SYMPTOMATIQUE
...

---

## 8. ⚠️ COMPLICATIONS
> 🚨 **POINTS DE VIGILANCE CRITIQUES**

| Complication | Facteurs de risque | Signes d'alarme | Conduite à tenir |
|--------------|-------------------|-----------------|-----------------|
| ... | ... | ... | ... |

**Contre-indications absolues :** ...

---

## 9. 📊 PRONOSTIC
...

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

**À retenir absolument :** ...
**Mnémotechnique :** ...

---

Rédige la fiche COMPLÈTE en remplaçant tous les "..." par le vrai contenu.
"""

FICHE_REVISION_PROMPT = """
Tu es Claude, expert en rédaction médicale.
Intègre TOUTES les corrections de l'audit Gemini dans ta fiche.
Rappel notation : constantes → `$...$`, équations chimiques → `$$...$$`.

**TA FICHE ORIGINALE :**
---
{original_fiche}
---

**CRITIQUE GEMINI :**
---
{critique}
---

Règles :
1. Corrige toutes les erreurs factuelles
2. Ajoute toutes les informations manquantes
3. La Physiopathologie reste la section la plus dense
4. L'algorithme décisionnel reste en code block
5. Structure en 10 points inchangée

Retourne la fiche ENTIÈRE et CORRIGÉE.
"""


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_from_pdf(file_path: str) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(file_path)
    text = ""
    for i, page in enumerate(doc):
        text += f"\n--- Page {i + 1} ---\n{page.get_text()}"
    doc.close()
    return text


def extract_from_pptx(file_path: str) -> str:
    from pptx import Presentation
    prs = Presentation(file_path)
    text = ""
    for i, slide in enumerate(prs.slides):
        text += f"\n--- Slide {i + 1} ---\n"
        if slide.shapes.title:
            text += f"TITRE: {slide.shapes.title.text}\n"
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip() and shape != slide.shapes.title:
                text += shape.text + "\n"
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                text += f"[NOTES]: {notes}\n"
    return text


def extract_from_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    for table in doc.tables:
        text += "\n[TABLEAU]\n"
        for row in table.rows:
            text += " | ".join(c.text.strip() for c in row.cells) + "\n"
    return text


def transcribe_video(uploaded_file, model_size: str = "base") -> str:
    import whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(tmp_path, language="fr", verbose=False)
        return result["text"]
    finally:
        os.unlink(tmp_path)


def extract_document(uploaded_file, file_type: str, whisper_size: str = "base") -> str:
    """Extract text from any supported file type."""
    if file_type == "mp4":
        return transcribe_video(uploaded_file, model_size=whisper_size)

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        extractors = {"pdf": extract_from_pdf, "pptx": extract_from_pptx, "docx": extract_from_docx}
        return extractors[file_type](tmp_path)
    finally:
        os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI AGENT
# ══════════════════════════════════════════════════════════════════════════════

class GeminiAgent:
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        self._genai = genai
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self._chat_session = None
        self._context_loaded = False

    def extract_data(self, document_text: str) -> str:
        prompt = EXTRACTION_PROMPT.format(document_text=document_text)
        return self.model.generate_content(prompt).text

    def critique_fiche(self, extracted_data: str, fiche_content: str) -> str:
        prompt = CRITIQUE_PROMPT.format(
            extracted_data=extracted_data,
            fiche_content=fiche_content,
        )
        return self.model.generate_content(prompt).text

    def initialize_chat(self, context: str = "", fiche: str = "") -> None:
        parts = ["Tu es un assistant médical expert (Gemini)."]
        if fiche:
            parts.append(
                "Tu as accès à la FICHE MÉDICALE FINALE validée par consensus.\n"
                "C'est ta référence prioritaire.\n\n"
                f"=== FICHE MÉDICALE FINALE ===\n{fiche}\n=== FIN DE LA FICHE ==="
            )
        if context:
            parts.append(
                "Tu as également accès au cours source.\n\n"
                f"=== COURS SOURCE ===\n{context[:CHAT_CTX_CHARS]}\n=== FIN DU COURS ==="
            )
        if fiche or context:
            parts.append(
                "Quand l'utilisateur pose une question sur un point précis de la fiche, "
                "appuie-toi directement sur la section correspondante. "
                "Complète avec le cours si nécessaire. "
                "Si absent des deux, dis-le clairement."
            )
        else:
            parts.append("Réponds aux questions médicales de manière précise et pédagogique.")

        system_ctx = "\n\n".join(parts)
        ack = (
            "Compris. J'ai la fiche médicale finale et le cours. Je suis prêt."
            if (fiche or context) else
            "Compris. Je suis prêt à répondre."
        )
        self._chat_session = self.model.start_chat(history=[
            {"role": "user",  "parts": [system_ctx]},
            {"role": "model", "parts": [ack]},
        ])
        self._context_loaded = True

    def chat(self, user_message: str, context: str = "", fiche: str = "") -> str:
        if self._chat_session is None or (not self._context_loaded and (context or fiche)):
            self.initialize_chat(context, fiche)
        return self._chat_session.send_message(user_message).text

    def reset_chat(self) -> None:
        self._chat_session = None
        self._context_loaded = False


# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE AGENT
# ══════════════════════════════════════════════════════════════════════════════

class ClaudeAgent:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._messages: list[dict] = []
        self._system_prompt: str = ""

    def _call(self, prompt: str, max_tokens: int = 8192) -> str:
        msg = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def draft_fiche(self, extracted_data: str) -> str:
        return self._call(FICHE_DRAFT_PROMPT.format(extracted_data=extracted_data))

    def revise_fiche(self, original_fiche: str, critique: str) -> str:
        return self._call(FICHE_REVISION_PROMPT.format(
            original_fiche=original_fiche,
            critique=critique,
        ))

    def initialize_chat(self, context: str = "", fiche: str = "") -> None:
        self._messages = []
        parts = ["Tu es Claude, un expert médical pédagogue."]
        if fiche:
            parts.append(
                "Tu as accès à la FICHE MÉDICALE FINALE validée par consensus.\n"
                "C'est ta référence prioritaire.\n\n"
                f"=== FICHE MÉDICALE FINALE ===\n{fiche}\n=== FIN DE LA FICHE ==="
            )
        if context:
            parts.append(
                "Tu as également accès au cours source.\n\n"
                f"=== COURS SOURCE ===\n{context[:CHAT_CTX_CHARS]}\n=== FIN DU COURS ==="
            )
        if fiche or context:
            parts.append(
                "Quand l'utilisateur pose une question sur un point précis de la fiche, "
                "appuie-toi directement sur la section correspondante. "
                "Complète avec le cours si nécessaire. "
                "Si absent des deux, dis-le clairement."
            )
        else:
            parts.append("Réponds aux questions médicales de manière précise et structurée.")
        self._system_prompt = "\n\n".join(parts)

    def chat(self, user_message: str, context: str = "", fiche: str = "") -> str:
        if not self._system_prompt and (context or fiche):
            self.initialize_chat(context, fiche)
        self._messages.append({"role": "user", "content": user_message})
        msg = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=self._system_prompt or "Tu es un expert médical pédagogue.",
            messages=self._messages,
        )
        response = msg.content[0].text
        self._messages.append({"role": "assistant", "content": response})
        return response

    def reset_chat(self) -> None:
        self._messages = []
        self._system_prompt = ""


# ══════════════════════════════════════════════════════════════════════════════
# CONSENSUS WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

def run_consensus(
    document_text: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    2-turn consensus workflow:
      Tour 1 : Gemini extrait → Claude rédige
      Tour 2 : Gemini critique → Claude corrige
    """
    def _p(v, msg):
        if progress_callback:
            progress_callback(v, msg)

    gemini = GeminiAgent()
    claude = ClaudeAgent()

    _p(0.10, "🔍 Tour 1 — Gemini analyse et extrait les données brutes...")
    extracted_data = gemini.extract_data(document_text)

    _p(0.35, "✍️ Tour 1 — Claude rédige la fiche médicale en 10 points...")
    draft_fiche = claude.draft_fiche(extracted_data)

    _p(0.62, "🔬 Tour 2 — Gemini valide et critique la fiche...")
    critique = gemini.critique_fiche(extracted_data, draft_fiche)

    _p(0.85, "📝 Tour 2 — Claude intègre les corrections et finalise...")
    final_fiche = claude.revise_fiche(draft_fiche, critique)

    _p(1.0, "✅ Consensus terminé — Fiche médicale validée !")
    return {
        "extracted_data": extracted_data,
        "draft_fiche":    draft_fiche,
        "critique":       critique,
        "final_fiche":    final_fiche,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_DISPLAY_MATH_RE = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)


def render_with_latex(content: str) -> None:
    """
    Render markdown with full LaTeX support:
    - $$...$$ → st.latex() (KaTeX display, crisp)
    - Rest    → st.markdown() (handles $...$ inline via KaTeX)
    Splitting on $$ prevents Streamlit from swallowing display math
    when mixed with fenced code blocks (algorithme décisionnel).
    """
    parts = _DISPLAY_MATH_RE.split(content)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            st.latex(part.strip())


def file_hash(uploaded_file) -> str:
    uploaded_file.seek(0)
    h = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return h


def format_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"


def context_hash(text: str) -> str:
    return str(hash(text[:500])) if text else ""


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

GLOBAL_CSS = """
<style>
    .main-header {
        background: linear-gradient(135deg, #0D47A1 0%, #4A148C 100%);
        padding: 24px 30px; border-radius: 14px; margin-bottom: 24px;
        box-shadow: 0 8px 30px rgba(13,71,161,0.35);
    }
    .main-header h1 { color: white; margin: 0; font-size: 30px; }
    .main-header .sub { color: rgba(255,255,255,0.7); margin: 8px 0 0 0; font-size: 13px; }
    .main-header .badges { margin-top: 12px; display: flex; gap: 8px; }
    .badge { display:inline-block; padding:3px 10px; border-radius:12px;
             font-size:11px; font-weight:600; color:white; }
    .badge.g { background: rgba(52,168,83,0.85); }
    .badge.c { background: rgba(100,100,255,0.85); }
    .badge.x { background: rgba(255,255,255,0.2); }

    .step-box {
        background:#E8EAF6; border-left:4px solid #3F51B5;
        padding:10px 16px; border-radius:0 8px 8px 0;
        margin:12px 0; font-size:14px; color:#1A237E; font-weight:600;
    }
    .api-warn {
        background:#FFF3E0; border:2px solid #FF9800; border-radius:10px;
        padding:16px 20px; margin:10px 0; font-size:14px;
    }
    .consensus-badge {
        display:inline-flex; align-items:center; gap:8px;
        background:linear-gradient(90deg,#00897B,#00ACC1);
        color:white; padding:6px 14px; border-radius:20px;
        font-size:13px; font-weight:600; margin-bottom:18px;
        box-shadow:0 2px 8px rgba(0,137,123,0.35);
    }
    .compare-label {
        background:#E8EAF6; border-radius:8px; padding:8px 14px;
        font-weight:700; font-size:14px; color:#3F51B5;
        margin-bottom:12px; text-align:center;
    }
    .compare-label.fin { background:#E8F5E9; color:#2E7D32; }

    #MainMenu { visibility:hidden; }
    footer     { visibility:hidden; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# UI — UPLOAD MODULE
# ══════════════════════════════════════════════════════════════════════════════

def render_upload() -> tuple[Optional[str], Optional[str]]:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1565C0,#4A148C);
                padding:20px 25px;border-radius:12px;margin-bottom:20px;
                box-shadow:0 4px 15px rgba(21,101,192,0.3);">
        <h3 style="color:white;margin:0;font-size:20px;">📁 Importer votre cours</h3>
        <p style="color:rgba(255,255,255,0.75);margin:6px 0 0 0;font-size:13px;">
            Formats supportés : PDF · PPTX · DOCX · MP4
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Glissez-déposez ou sélectionnez un fichier",
        type=["pdf", "pptx", "docx", "mp4"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown("""
        <div style="border:2px dashed #90CAF9;border-radius:10px;padding:30px;
                    text-align:center;color:#5C6BC0;background:#F8F9FF;margin-top:10px;">
            <div style="font-size:40px;">📂</div>
            <div style="font-size:15px;margin-top:8px;">Aucun fichier sélectionné</div>
            <div style="font-size:12px;margin-top:4px;color:#9E9E9E;">
                PDF · PowerPoint · Word · Vidéo MP4</div>
        </div>
        """, unsafe_allow_html=True)
        return None, None

    file_type = uploaded.name.split(".")[-1].lower()

    st.markdown(f"""
    <div style="background:#E8F5E9;border-left:4px solid #43A047;
                border-radius:0 8px 8px 0;padding:10px 15px;margin:10px 0;
                font-size:14px;color:#1B5E20;">
        📄 <b>{uploaded.name}</b> — {format_size(uploaded.size)} — <code>.{file_type}</code>
    </div>
    """, unsafe_allow_html=True)

    # Whisper quality selector BEFORE processing (outside spinner)
    whisper_size = "base"
    if file_type == "mp4":
        whisper_size = st.selectbox(
            "Qualité de transcription Whisper",
            options=["tiny", "base", "small", "medium"],
            index=1,
            help="Plus le modèle est grand, plus la transcription est précise mais lente.",
        )

    # Cache by (file_hash + whisper_size) — avoids re-extraction on every Streamlit rerun
    fh = file_hash(uploaded)
    cache_key = f"extracted_{fh}_{whisper_size}"

    if cache_key in st.session_state:
        text = st.session_state[cache_key]
    else:
        if file_type == "mp4":
            with st.spinner("🎙️ Transcription Whisper en cours (peut prendre plusieurs minutes)..."):
                text = transcribe_video(uploaded, model_size=whisper_size)
        else:
            with st.spinner("Lecture du fichier..."):
                text = extract_document(uploaded, file_type)
        st.session_state[cache_key] = text

    c1, c2, c3 = st.columns(3)
    c1.metric("Caractères", f"{len(text):,}")
    c2.metric("Mots",       f"{len(text.split()):,}")
    c3.metric("Tokens (~)", f"{len(text) // 4:,}")

    with st.expander("👁️ Aperçu", expanded=False):
        st.text(text[:3000] + ("..." if len(text) > 3000 else ""))

    return text, uploaded.name


# ══════════════════════════════════════════════════════════════════════════════
# UI — FICHE DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

FICHE_CSS = """
<style>
    .compare-label      { background:#E8EAF6; border-radius:8px; padding:8px 14px;
                          font-weight:700; font-size:14px; color:#3F51B5;
                          margin-bottom:12px; text-align:center; }
    .compare-label.fin  { background:#E8F5E9; color:#2E7D32; }
    .critique-box       { background:#FFF8E1; border-left:5px solid #F9A825;
                          border-radius:0 10px 10px 0; padding:14px 18px; margin:10px 0; }
    .critique-box .lbl  { font-weight:700; color:#E65100; font-size:15px; margin-bottom:8px; }
</style>
"""


def render_fiche(content: str, badge: bool = True) -> None:
    st.markdown(FICHE_CSS, unsafe_allow_html=True)
    if badge:
        st.markdown(
            '<div class="consensus-badge">'
            '✅ Validé par consensus Gemini 1.5 Pro × Claude Opus — 2 tours'
            '</div>',
            unsafe_allow_html=True,
        )
    render_with_latex(content)


def render_comparison(draft: str, final: str) -> None:
    st.markdown(FICHE_CSS, unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown('<div class="compare-label">📄 Brouillon — Tour 1</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            render_with_latex(draft)
    with col2:
        st.markdown('<div class="compare-label fin">✅ Version Finale — Tour 2</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            render_with_latex(final)


def render_critique(critique: str) -> None:
    st.markdown(FICHE_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="critique-box">
        <div class="lbl">🔬 Critique Gemini — Tour 2</div>
    </div>
    """, unsafe_allow_html=True)
    render_with_latex(critique)


# ══════════════════════════════════════════════════════════════════════════════
# UI — SIDEBAR CHAT
# ══════════════════════════════════════════════════════════════════════════════

def _init_chat_state() -> None:
    if "gemini_agent"         not in st.session_state:
        st.session_state.gemini_agent = GeminiAgent()
    if "claude_agent"         not in st.session_state:
        st.session_state.claude_agent = ClaudeAgent()
    if "chat_messages"        not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_doc_hash"        not in st.session_state:
        st.session_state.chat_doc_hash = ""
    if "chat_fiche_hash"      not in st.session_state:
        st.session_state.chat_fiche_hash = ""


def render_sidebar(document_context: str = "", final_fiche: str = "") -> None:
    _init_chat_state()

    # Auto-reinitialize agents when context changes (e.g. fiche generated after doc upload)
    doc_h   = context_hash(document_context)
    fiche_h = context_hash(final_fiche)
    if (doc_h != st.session_state.chat_doc_hash or
            fiche_h != st.session_state.chat_fiche_hash) and (document_context or final_fiche):
        st.session_state.gemini_agent.initialize_chat(document_context, final_fiche)
        st.session_state.claude_agent.initialize_chat(document_context, final_fiche)
        st.session_state.chat_doc_hash   = doc_h
        st.session_state.chat_fiche_hash = fiche_h

    with st.sidebar:
        status = (
            "✅ Fiche + cours chargés" if final_fiche
            else ("📄 Cours chargé" if document_context else "Aucun contexte")
        )
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1A237E,#4A148C);
                    padding:16px 18px;border-radius:12px;margin-bottom:14px;">
            <div style="color:white;font-size:18px;font-weight:700;">🧠 Assistant Hybride</div>
            <div style="color:rgba(255,255,255,0.65);font-size:12px;margin-top:4px;">
                Gemini 1.5 Pro × Claude Opus 4.6</div>
            <div style="color:rgba(255,255,255,0.5);font-size:11px;">{status}</div>
        </div>
        """, unsafe_allow_html=True)

        llm_mode = st.radio(
            "Mode de réponse",
            ["🔵 Claude Opus", "🟢 Gemini Pro", "🟣 Consensus (les deux)"],
            index=2,
        )

        if not document_context and not final_fiche:
            st.warning("⚠️ Importez un document pour activer l'assistant.")
        elif not final_fiche:
            st.info("💡 Générez une fiche pour des réponses encore plus précises.")
        else:
            st.success("🎯 Fiche + cours chargés.")

        st.divider()

        # Chat history
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="background:#E3F2FD;border-radius:10px 10px 2px 10px;
                            padding:10px 13px;margin:6px 0;font-size:13px;color:#0D47A1;">
                    👤 <b>Vous</b><br>{msg['content']}</div>
                """, unsafe_allow_html=True)
            else:
                model = msg.get("model", "Assistant")
                icon  = "🔵" if "Claude" in model else ("🟢" if "Gemini" in model else "🟣")
                bg    = "#F3E5F5" if "Consensus" in model else ("#E8F5E9" if "Gemini" in model else "#EDE7F6")
                col   = "#4A148C" if "Consensus" in model else ("#1B5E20" if "Gemini" in model else "#311B92")
                st.markdown(f"""
                <div style="background:{bg};border-radius:10px 10px 10px 2px;
                            padding:10px 13px;margin:6px 0;font-size:13px;color:{col};">
                    {icon} <b>{model}</b></div>
                """, unsafe_allow_html=True)
                st.markdown(msg["content"])

        # Input
        user_input = st.chat_input("Posez une question sur le cours...")
        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})

            with st.spinner("Réflexion en cours..."):
                if "Claude" in llm_mode:
                    resp  = st.session_state.claude_agent.chat(user_input, document_context, final_fiche)
                    model = "Claude Opus"
                elif "Gemini" in llm_mode:
                    resp  = st.session_state.gemini_agent.chat(user_input, document_context, final_fiche)
                    model = "Gemini Pro"
                else:
                    cr = st.session_state.claude_agent.chat(user_input, document_context, final_fiche)
                    gr = st.session_state.gemini_agent.chat(user_input, document_context, final_fiche)
                    resp  = f"**🔵 Claude Opus :**\n\n{cr}\n\n---\n\n**🟢 Gemini Pro :**\n\n{gr}"
                    model = "Consensus"

            st.session_state.chat_messages.append({"role": "assistant", "content": resp, "model": model})
            st.rerun()

        # Clear
        if st.session_state.chat_messages:
            st.divider()
            if st.button("🗑️ Effacer le chat", use_container_width=True):
                st.session_state.chat_messages  = []
                st.session_state.gemini_agent   = GeminiAgent()
                st.session_state.claude_agent   = ClaudeAgent()
                st.session_state.chat_doc_hash  = ""
                st.session_state.chat_fiche_hash = ""
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MedFiche AI — Gemini × Claude",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏥 MedFiche AI</h1>
    <p class="sub">Génération de fiches médicales par consensus d'intelligence artificielle</p>
    <div class="badges">
        <span class="badge g">🟢 Gemini 1.5 Pro — Extraction & Validation</span>
        <span class="badge c">🔵 Claude Opus 4.6 — Rédaction médicale</span>
        <span class="badge x">⚡ Consensus 2 tours</span>
    </div>
</div>
""", unsafe_allow_html=True)

# API guard
missing = [k for k, v in {"ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
                           "GOOGLE_API_KEY": GOOGLE_API_KEY}.items() if not v]
if missing:
    st.markdown(
        f'<div class="api-warn">⚠️ <b>Clés API manquantes :</b> '
        f'{", ".join(f"<code>{k}</code>" for k in missing)}<br><br>'
        f'Ajoutez-les dans les <b>Secrets</b> Streamlit Cloud '
        f'(Settings → Secrets) ou dans un fichier <code>.env</code> local.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# Session state
for key, default in [
    ("document_text", None), ("file_name", None),
    ("results", None),       ("generating", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar (persistent across all tabs)
_fiche = (st.session_state.results or {}).get("final_fiche", "") or ""
render_sidebar(
    document_context=st.session_state.document_text or "",
    final_fiche=_fiche,
)

# Tabs
tab1, tab2, tab3 = st.tabs(["📁 Import & Génération", "📋 Fiche Finale", "🔬 Détails du Consensus"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    doc_text, file_name = render_upload()

    if doc_text:
        # Truncate and store
        truncated = doc_text[:MAX_DOC_CHARS] + (
            "\n\n[... CONTENU TRONQUÉ — limite de tokens atteinte ...]"
            if len(doc_text) > MAX_DOC_CHARS else ""
        )
        st.session_state.document_text = truncated
        st.session_state.file_name     = file_name
        st.markdown(
            f'<div class="step-box">✅ Document prêt — ~{len(truncated) // 4:,} tokens estimés. '
            'Cliquez sur "Générer" pour lancer le consensus.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.document_text:
        col_btn, _ = st.columns([2, 3])
        with col_btn:
            generate = st.button(
                "🚀 Générer la fiche médicale",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.generating,
            )

        if generate:
            st.session_state.generating = True
            bar      = st.progress(0.0)
            status   = st.empty()

            def _upd(v: float, msg: str) -> None:
                bar.progress(v)
                status.markdown(f'<div class="step-box">{msg}</div>', unsafe_allow_html=True)

            try:
                results = run_consensus(st.session_state.document_text, _upd)
                st.session_state.results    = results
                st.session_state.generating = False
                status.success("🎉 Fiche générée ! Consultez l'onglet **Fiche Finale**.")
            except Exception as e:
                st.session_state.generating = False
                st.error(f"❌ Erreur : {e}")

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    if not st.session_state.results:
        st.info("👆 Importez un document et cliquez sur **Générer la fiche médicale**.")
    else:
        res = st.session_state.results
        col_t, col_tog, col_dl = st.columns([3, 2, 1])
        with col_t:
            st.markdown("## 📋 Fiche Médicale Finale")
        with col_tog:
            show_cmp = st.toggle("🔀 Brouillon / Final", value=False)
        with col_dl:
            st.download_button(
                "⬇️ .md",
                data=res["final_fiche"],
                file_name=f"fiche_{st.session_state.file_name or 'medicale'}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        st.divider()
        if show_cmp:
            render_comparison(res["draft_fiche"], res["final_fiche"])
        else:
            render_fiche(res["final_fiche"], badge=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    if not st.session_state.results:
        st.info("Générez d'abord une fiche dans l'onglet 1.")
    else:
        res = st.session_state.results
        st.markdown("## 🔬 Détails du Processus de Consensus")
        st.markdown("""
        <div style="background:#E8EAF6;border-radius:10px;padding:14px 18px;
                    margin-bottom:20px;font-size:13px;color:#1A237E;">
            <b>Flux :</b> 📄 Document
            → <b>🟢 Gemini (extraction JSON)</b>
            → <b>🔵 Claude (rédaction fiche)</b>
            → <b>🟢 Gemini (critique)</b>
            → <b>🔵 Claude (corrections)</b>
            → ✅ Fiche validée
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🟢 Tour 1 — Extraction Gemini", expanded=False):
            st.markdown(res["extracted_data"])

        with st.expander("🔵 Tour 1 — Brouillon Claude", expanded=False):
            render_fiche(res["draft_fiche"], badge=False)

        with st.expander("🟢 Tour 2 — Critique Gemini", expanded=False):
            render_critique(res["critique"])

        with st.expander("✅ Tour 2 — Fiche Finale Claude", expanded=True):
            render_fiche(res["final_fiche"], badge=True)
