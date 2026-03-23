import hashlib
import json
import os
import re
import tempfile
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
XAI_API_KEY      = os.getenv("XAI_API_KEY")

GEMINI_MODEL   = "gemini-2.0-flash"   # fallback auto si indispo
DEEPSEEK_MODEL = "deepseek-reasoner"
GROK_MODEL     = "grok-3"

# Ordre de préférence Gemini — le premier disponible sera utilisé
GEMINI_FALLBACK = [
    "gemini-1.5-flash-8b",       # le plus petit, dispo partout
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
]

CHAT_CTX_CHARS = 60_000
MAX_DOC_CHARS  = 150_000

# ── PROMPTS ───────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """\
Tu es un agent d'extraction médicale (Gemini 1.5 Pro).
Analyse ce document de façon EXHAUSTIVE et retourne un JSON structuré avec :
- "titre"             : thème principal du cours
- "donnees_brutes"    : faits, chiffres, valeurs normales/pathologiques
- "schemas_causaux"   : cascades mécanistiques (A → B → conséquence)
- "entites_medicales" : pathologies, médicaments (DCI + posologie), examens
- "points_critiques"  : complications, urgences, contre-indications
- "mnémotechniques"   : acronymes, règles mémo
- "algorithmes"       : arbres décisionnels

DOCUMENT :
---
{document_text}
---
Retourne UNIQUEMENT le JSON. Sois exhaustif.
"""

PHYSIOPATH_PROMPT = """\
Tu es DeepSeek-R1, modèle de raisonnement avec Chain-of-Thought profond.
Mission EXCLUSIVE : produire la section Physiopathologie CAUSALE la plus rigoureuse possible.

DONNÉES EXTRAITES PAR GEMINI :
```json
{extracted_data}
```

Raisonne étape par étape, puis génère :

## ⚡ PHYSIOPATHOLOGIE CAUSALE

### 1. Mécanisme déclencheur initial
### 2. Cascade physiopathologique complète
(flèches causales : Trigger → Mécanisme → Effet cellulaire → Manifestation clinique)
### 3. Schéma logique (format ASCII)
### 4. Bottlenecks mécanistiques clés (tableau)
### 5. Corrélations anatomo-cliniques

Utilise LaTeX pour les constantes ($K_a$, $Ca^{{2+}}$, $HCO_3^-$)
et les équations : $$CO_2 + H_2O \\rightleftharpoons H^+ + HCO_3^-$$
"""

SYNTHESIS_PROMPT = """\
Tu es Grok-3 (xAI). Synthétise les deux entrées en une fiche médicale parfaite.

━━━ EXTRACTION GEMINI ━━━
{extracted_data}

━━━ PHYSIOPATHOLOGIE DEEPSEEK-R1 ━━━
{physiopath_analysis}

LaTeX obligatoire : ions → $Ca^{{2+}}$, équations → $$CO_2 + H_2O \\rightleftharpoons H^+ + HCO_3^-$$

# {title}

> Résumé en 2-3 phrases.

## 1. ⚡ PHYSIOPATHOLOGIE CAUSALE
[Intègre le contenu DeepSeek-R1 intégralement]

## 2. 🔬 BOTTLENECKS MÉCANISTIQUES
| Bottleneck | Mécanisme | Conséquence | Intervention |
|---|---|---|---|

## 3. 🩺 MANIFESTATIONS CLINIQUES
**Très fréquents (>50%) :** ...
**Fréquents (10-50%) :** ...
**Rares mais critiques :** ...

## 4. ⚖️ DIAGNOSTIC DIFFÉRENTIEL
| Pathologie | Arguments POUR | Arguments CONTRE | Examen discriminant |
|---|---|---|---|

## 5. 🔍 EXAMENS COMPLÉMENTAIRES
**Urgence :** ... **Confirmation :** ... **Suivi :** ...

## 6. 💊 TRAITEMENT ÉTIOLOGIQUE
## 7. 🩹 TRAITEMENT SYMPTOMATIQUE

## 8. ⚠️ COMPLICATIONS
> 🚨 POINTS DE VIGILANCE
| Complication | Facteurs de risque | Signes d'alarme | CAT |
|---|---|---|---|

## 9. 📊 PRONOSTIC

## 10. 🧠 ALGORITHME DÉCISIONNEL
```
PRÉSENTATION CLINIQUE
        ↓
[Critère A ?]
├── OUI → [Examen B] → Positif → Traitement X
└── NON → [Critère C ?] → OUI → Traitement Z
```

## 💡 POINTS CLÉS / MNÉMOTECHNIQUES

Rédige la fiche COMPLÈTE. Remplace tous les "..." par du vrai contenu.
"""

GROK_SYSTEM = """\
Tu es Grok-3 (xAI), assistant médical de MedFiche AI.

{fiche_block}

{context_block}

Règles : appuie-toi sur la fiche pour répondre aux questions précises.
Complète avec le cours si besoin. Réponds en français avec LaTeX pour les équations.
"""

# ── DOCUMENT EXTRACTION ───────────────────────────────────────────────────────
def _file_hash(f) -> str:
    f.seek(0); h = hashlib.md5(f.read()).hexdigest(); f.seek(0)
    return h

def extract_document(uploaded_file, file_type: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    try:
        if file_type == "pdf":
            import fitz
            doc = fitz.open(path)
            out = "\n".join(f"--- Page {i+1} ---\n{p.get_text()}" for i, p in enumerate(doc))
            doc.close()
            return out
        elif file_type == "pptx":
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
            return out
        elif file_type == "docx":
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return ""
    finally:
        os.unlink(path)

# ── AGENTS ────────────────────────────────────────────────────────────────────
class GeminiAgent:
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)

        # ── Diagnostic : liste tous les modèles disponibles ──────────────────
        try:
            available = [m.name for m in genai.list_models()
                         if "generateContent" in m.supported_generation_methods]
            print("=== GEMINI — modèles disponibles sur cette clé API ===")
            for name in available:
                print(f"  • {name}")
            if not available:
                print("  (liste vide)")
        except Exception as list_err:
            available = []
            print(f"=== GEMINI — impossible de lister les modèles : {list_err} ===")

        if not available:
            raise RuntimeError(
                "Aucun modèle Gemini accessible avec cette clé API.\n\n"
                "Solutions :\n"
                "  1. Activez l'API Gemini sur https://aistudio.google.com → 'Get API key'\n"
                "  2. Vérifiez que GOOGLE_API_KEY est bien définie dans .env / Streamlit Secrets\n"
                "  3. Attendez quelques minutes si la clé vient d'être créée"
            )

        print(f"=== GEMINI — tentative dans l'ordre : {GEMINI_FALLBACK} ===")

        self.model = None
        self.model_name = None
        for name in GEMINI_FALLBACK:
            # Normalise : accepte "gemini-1.5-flash" et "models/gemini-1.5-flash"
            normalized = name if name.startswith("models/") else f"models/{name}"
            if normalized not in available:
                print(f"  ✗ {name} — absent de la liste")
                continue
            try:
                m = genai.GenerativeModel(name)
                m.generate_content("test", generation_config={"max_output_tokens": 1, "temperature": 0})
                self.model = m
                self.model_name = name
                print(f"  ✓ {name} — sélectionné")
                break
            except Exception as e:
                print(f"  ✗ {name} — erreur : {e}")
                continue

        # Dernier recours : premier modèle disponible dans la liste
        if self.model is None and available:
            fallback_name = available[0].removeprefix("models/")
            print(f"  ⚠ Aucun modèle GEMINI_FALLBACK dispo — utilisation de {fallback_name}")
            self.model = genai.GenerativeModel(fallback_name)
            self.model_name = fallback_name

        if self.model is None:
            raise RuntimeError(f"Aucun modèle Gemini disponible. Essayés : {GEMINI_FALLBACK}")

    def extract(self, text: str) -> str:
        return self.model.generate_content(
            EXTRACTION_PROMPT.format(document_text=text),
            generation_config={"max_output_tokens": 8192, "temperature": 0.2},
        ).text


class DeepSeekAgent:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    def reason(self, data: str) -> tuple[str, str]:
        res = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": PHYSIOPATH_PROMPT.format(extracted_data=data)}],
        )
        msg = res.choices[0].message
        return getattr(msg, "reasoning_content", "") or "", msg.content or ""


class GrokAgent:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
        self._messages: list[dict] = []
        self._system = ""

    def synthesize(self, data: str, physio: str, title: str) -> str:
        res = self.client.chat.completions.create(
            model=GROK_MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": SYNTHESIS_PROMPT.format(
                extracted_data=data, physiopath_analysis=physio, title=title,
            )}],
        )
        return res.choices[0].message.content

    def initialize_chat(self, context: str = "", fiche: str = "") -> None:
        self._messages = []
        self._system = GROK_SYSTEM.format(
            fiche_block=(
                f"=== FICHE MÉDICALE (référence prioritaire) ===\n{fiche}\n=== FIN ==="
                if fiche else "Aucune fiche générée."
            ),
            context_block=(
                f"=== COURS SOURCE ===\n{context[:CHAT_CTX_CHARS]}\n=== FIN ==="
                if context else ""
            ),
        )

    def chat(self, message: str, context: str = "", fiche: str = "") -> str:
        if not self._system:
            self.initialize_chat(context, fiche)
        self._messages.append({"role": "user", "content": message})
        res = self.client.chat.completions.create(
            model=GROK_MODEL,
            max_tokens=4096,
            messages=[{"role": "system", "content": self._system}] + self._messages,
        )
        reply = res.choices[0].message.content
        self._messages.append({"role": "assistant", "content": reply})
        return reply

# ── LATEX RENDERER ────────────────────────────────────────────────────────────
_DISPLAY_MATH = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)

def render_latex(content: str) -> None:
    """$$...$$ → st.latex() | reste → st.markdown() (gère $...$ inline)."""
    for i, part in enumerate(_DISPLAY_MATH.split(content)):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            st.latex(part.strip())

# ── SIDEBAR CHAT ──────────────────────────────────────────────────────────────
def _ctx_hash(t: str) -> str:
    return str(hash(t[:500])) if t else ""

def render_sidebar(doc_ctx: str = "", fiche: str = "") -> None:
    for k, v in [("grok", None), ("chat_msgs", []), ("chat_dh", ""), ("chat_fh", "")]:
        if k not in st.session_state:
            st.session_state[k] = v

    dh, fh = _ctx_hash(doc_ctx), _ctx_hash(fiche)
    if (dh != st.session_state.chat_dh or fh != st.session_state.chat_fh) and (doc_ctx or fiche):
        if st.session_state.grok is None:
            st.session_state.grok = GrokAgent()
        st.session_state.grok.initialize_chat(doc_ctx, fiche)
        st.session_state.chat_dh = dh
        st.session_state.chat_fh = fh

    status = "✅ Fiche + cours" if fiche else ("📄 Cours chargé" if doc_ctx else "Aucun contexte")

    with st.sidebar:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0a0a23,#1a0533);
                    padding:16px;border-radius:12px;margin-bottom:12px;">
            <div style="color:#fff;font-size:17px;font-weight:700;">✨ Grok-3 — Chat</div>
            <div style="color:rgba(255,255,255,0.5);font-size:11px;margin-top:4px;">{status}</div>
        </div>""", unsafe_allow_html=True)

        if not fiche and not doc_ctx:
            st.warning("Importez un document pour activer le chat.")
        elif not fiche:
            st.info("💡 Générez une fiche pour des réponses précises.")
        else:
            st.success("🎯 Grok-3 a accès à votre fiche.")

        st.divider()

        for msg in st.session_state.chat_msgs:
            if msg["role"] == "user":
                st.markdown(f"""<div style="background:#e3f2fd;border-radius:10px 10px 2px 10px;
                    padding:9px 12px;margin:5px 0;font-size:13px;color:#0d47a1;">
                    👤 <b>Vous</b><br>{msg['content']}</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style="background:#ede7f6;border-radius:10px 10px 10px 2px;
                    padding:9px 12px;margin:3px 0;font-size:13px;color:#311b92;">
                    ✨ <b>Grok-3</b></div>""", unsafe_allow_html=True)
                st.markdown(msg["content"])

        user_input = st.chat_input("Posez une question...")
        if user_input:
            if st.session_state.grok is None:
                st.session_state.grok = GrokAgent()
            st.session_state.chat_msgs.append({"role": "user", "content": user_input})
            with st.spinner("Grok-3 réfléchit..."):
                reply = st.session_state.grok.chat(user_input, doc_ctx, fiche)
            st.session_state.chat_msgs.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state.chat_msgs:
            st.divider()
            if st.button("🗑️ Effacer", use_container_width=True):
                st.session_state.chat_msgs = []
                st.session_state.grok = None
                st.session_state.chat_dh = ""
                st.session_state.chat_fh = ""
                st.rerun()

# ── APP ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MedFiche AI — Triple Agent", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.hdr{background:linear-gradient(135deg,#0a0a23,#1a0533,#001a33);
     padding:24px 30px;border-radius:14px;margin-bottom:22px;
     box-shadow:0 8px 30px rgba(0,0,0,0.4);}
.hdr h1{color:#fff;margin:0;font-size:26px;}
.hdr p{color:rgba(255,255,255,0.6);margin:8px 0 0;font-size:13px;}
.hdr .tags{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;}
.tag{padding:4px 11px;border-radius:16px;font-size:11px;font-weight:600;color:#fff;}
.t1{background:linear-gradient(90deg,#1a73e8,#0d47a1);}
.t2{background:linear-gradient(90deg,#00bcd4,#006064);}
.t3{background:linear-gradient(90deg,#7c4dff,#311b92);}
.card{border-radius:10px;padding:13px 17px;margin:7px 0;
      display:flex;align-items:center;gap:11px;font-size:14px;}
.c1{background:#e3f2fd;border-left:4px solid #1565c0;}
.c2{background:#e0f7fa;border-left:4px solid #00838f;}
.c3{background:#ede7f6;border-left:4px solid #6a1b9a;}
.badge{display:inline-flex;align-items:center;gap:7px;
       background:linear-gradient(90deg,#1a73e8,#7c4dff);
       color:#fff;padding:5px 14px;border-radius:18px;
       font-size:12px;font-weight:600;margin-bottom:16px;}
#MainMenu,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hdr">
  <h1>🏥 MedFiche AI</h1>
  <p>Fiches médicales par consensus de trois intelligences artificielles</p>
  <div class="tags">
    <span class="tag t1">🔍 Gemini 1.5 Pro · Extraction</span>
    <span class="tag t2">🧠 DeepSeek-R1 · Raisonnement</span>
    <span class="tag t3">✨ Grok-3 · Synthèse & Chat</span>
  </div>
</div>""", unsafe_allow_html=True)

# API guard
if not all([GOOGLE_API_KEY, DEEPSEEK_API_KEY, XAI_API_KEY]):
    missing = [k for k, v in {"GOOGLE_API_KEY": GOOGLE_API_KEY,
               "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY, "XAI_API_KEY": XAI_API_KEY}.items() if not v]
    st.error(f"Clés API manquantes : {', '.join(missing)}")
    st.stop()

# Session state
for k, v in [("doc_text", None), ("file_name", None), ("results", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar
fiche_ctx = (st.session_state.results or {}).get("final_fiche", "") or ""
render_sidebar(st.session_state.doc_text or "", fiche_ctx)

# Tabs
tab1, tab2, tab3 = st.tabs(["📁 Import & Génération", "📋 Fiche Finale", "🔬 Détails"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader("Cours (PDF, PPTX, DOCX)", type=["pdf", "pptx", "docx"])

    if uploaded:
        ftype = uploaded.name.split(".")[-1].lower()
        # Cache par hash — évite la ré-extraction si même fichier, force si nouveau fichier
        fh = _file_hash(uploaded)
        cache_key = f"doc_{fh}"
        if cache_key not in st.session_state:
            with st.spinner("Lecture du fichier..."):
                raw = extract_document(uploaded, ftype)
            # Vider les anciens caches doc_ pour ne pas accumuler
            for k in list(st.session_state.keys()):
                if k.startswith("doc_") and k != cache_key:
                    del st.session_state[k]
            st.session_state[cache_key]  = raw[:MAX_DOC_CHARS]
            st.session_state.doc_text    = st.session_state[cache_key]
            st.session_state.file_name   = uploaded.name
            st.session_state.results     = None  # reset fiche si nouveau fichier

        text = st.session_state[cache_key]
        c1, c2, c3 = st.columns(3)
        c1.metric("Caractères", f"{len(text):,}")
        c2.metric("Mots",       f"{len(text.split()):,}")
        c3.metric("Tokens (~)", f"{len(text)//4:,}")

        with st.expander("👁️ Aperçu", expanded=False):
            st.text(text[:2000] + ("..." if len(text) > 2000 else ""))

    if st.session_state.doc_text:
        col, _ = st.columns([2, 3])
        with col:
            go = st.button("🚀 Lancer la Triplette Médicale", type="primary", use_container_width=True)

        if go:
            bar      = st.progress(0.0)
            status   = st.empty()
            cards    = st.empty()

            def _show_cards(active: int) -> None:
                agents = [
                    ("c1","🔍","Agent 1 — Gemini 1.5 Pro","Extraction OCR + schémas"),
                    ("c2","🧠","Agent 2 — DeepSeek-R1","Physiopathologie (Chain-of-Thought)"),
                    ("c3","✨","Agent 3 — Grok-3","Synthèse finale 10 points"),
                ]
                html = ""
                for i, (cls, ico, lbl, sub) in enumerate(agents, 1):
                    s = "⏳ En cours..." if i == active else ("✅ Terminé" if i < active else "⌛ En attente")
                    op = "1" if i <= active else "0.4"
                    html += f'<div class="card {cls}" style="opacity:{op};"><span style="font-size:20px">{ico}</span><div><b>{lbl}</b><br><span style="font-size:12px;color:#666">{sub} — {s}</span></div></div>'
                cards.markdown(html, unsafe_allow_html=True)

            try:
                # Agent 1
                bar.progress(0.05)
                status.info("🔍 Agent 1 — Gemini extrait le corpus...")
                _show_cards(1)
                ext = GeminiAgent().extract(st.session_state.doc_text)

                # Extraire le titre du JSON
                title = st.session_state.file_name or "Cours médical"
                try:
                    m = re.search(r'\{.*\}', ext, re.DOTALL)
                    if m:
                        title = json.loads(m.group()).get("titre", title)
                except Exception:
                    pass

                # Agent 2
                bar.progress(0.35)
                status.info("🧠 Agent 2 — DeepSeek-R1 raisonne sur la physiopathologie...")
                _show_cards(2)
                reasoning, physio = DeepSeekAgent().reason(ext)

                # Agent 3
                bar.progress(0.68)
                status.info("✨ Agent 3 — Grok-3 synthétise la fiche finale...")
                _show_cards(3)
                final = GrokAgent().synthesize(ext, physio, title)

                bar.progress(1.0)
                _show_cards(4)
                st.session_state.results = {
                    "extracted_data": ext,
                    "reasoning_trace": reasoning,
                    "physiopath": physio,
                    "final_fiche": final,
                    "title": title,
                }
                status.success("🎉 Terminé ! Consultez l'onglet **Fiche Finale**.")

            except Exception as e:
                status.error(f"❌ Erreur : {e}")

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    if not st.session_state.results:
        st.info("👆 Importez un document et lancez la Triplette.")
    else:
        res = st.session_state.results
        col_t, col_dl = st.columns([5, 1])
        with col_t:
            st.markdown(f"## 📋 {res['title']}")
        with col_dl:
            st.download_button("⬇️ .md", data=res["final_fiche"],
                file_name=f"fiche_{res['title']}.md", mime="text/markdown",
                use_container_width=True)
        st.markdown('<div class="badge">🤖 Gemini · DeepSeek-R1 · Grok-3</div>',
                    unsafe_allow_html=True)
        st.divider()
        render_latex(res["final_fiche"])

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    if not st.session_state.results:
        st.info("Lancez d'abord la Triplette.")
    else:
        res = st.session_state.results
        with st.expander("🔍 Agent 1 — Extraction Gemini", expanded=False):
            st.markdown(res["extracted_data"])
        with st.expander("🧠 Agent 2 — Physiopathologie DeepSeek-R1", expanded=False):
            if res.get("reasoning_trace"):
                with st.expander("🔎 Trace CoT interne", expanded=False):
                    st.code(res["reasoning_trace"][:5000])
            render_latex(res["physiopath"])
        with st.expander("✨ Agent 3 — Fiche Finale Grok-3", expanded=True):
            render_latex(res["final_fiche"])
