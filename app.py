"""
MedFiche AI — Triple-Agent Architecture (VERSION CORRIGÉE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent 1 : Gemini 1.5 Pro   — Extraction OCR + analyse des schémas
Agent 2 : DeepSeek-R1      — Physiopathologie causale (Chain-of-Thought)
Agent 3 : Grok-3 (xAI)    — Synthèse finale 10 points + LaTeX + Chat
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import json
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

# Modèle Pro sans le suffixe -latest pour éviter la 404 sur v1beta
GEMINI_MODEL   = "gemini-1.5-pro"
DEEPSEEK_MODEL = "deepseek-reasoner"
GROK_MODEL     = "grok-3"

MAX_DOC_CHARS  = 150_000
CHAT_CTX_CHARS =  60_000

# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """\
Tu es un agent d'extraction médicale de haut niveau (Gemini 1.5 Pro).
Analyse ce document de façon EXHAUSTIVE.

Extrais et retourne un JSON structuré avec :
1. "titre"             : titre ou thème principal du cours
2. "donnees_brutes"    : faits, chiffres, valeurs normales/pathologiques
3. "schemas_causaux"   : cascades mécanistiques (A → B → C)
4. "entites_medicales" : pathologies, médicaments (DCI), examens
5. "sequences"         : évolution temporelle, stades
6. "points_critiques"  : complications, urgences, contre-indications
7. "mnémotechniques"   : acronymes, classifications
8. "algorithmes"       : arbres décisionnels

DOCUMENT :
---
{document_text}
---

Retourne UNIQUEMENT le JSON.
"""

PHYSIOPATH_PROMPT = """\
Tu es DeepSeek-R1. Ta mission : produire la section "Physiopathologie CAUSALE".

DONNÉES EXTRAITES :
```json
{extracted_data}
