# 📺 YouTube Post Scraper — PWA

Application web mobile (PWA) **installable** et **100 % hors ligne** qui capture
les posts (publications communautaires) YouTube et les exporte en **Markdown**,
**TXT** ou **JSON**. Aucun serveur, aucun proxy, aucune donnée envoyée.

## ✨ Fonctionnalités

- **Capture automatique** d'un post YouTube (Mode Actif) avec toast + badge d'app
- **Extraction DOM intelligente** (sélecteurs YouTube par priorité, décodage des
  entités HTML, nettoyage du texte)
- **Formatage Markdown** avec auteur, URL, date, contenu, tags et métadonnées
- **Export** : Copier, Télécharger (.md / .txt / .json), Partager (Web Share)
- **Historique** local (IndexedDB) : jusqu'à 100 posts, purge auto après 30 jours,
  recherche, suppression, « Tout effacer »
- **PWA complète** : manifest, service worker (Workbox), installable, offline,
  icônes, cible de partage Android/iOS
- **Bonus** : thème sombre/clair, tags, statistiques (posts / mots)

## ⚠️ La contrainte technique (à lire)

YouTube **bloque toute lecture externe** : à cause de la *same-origin policy* du
navigateur, **aucune** page web (donc aucune PWA) ne peut lire le DOM de
`youtube.com` à distance — `fetch`, proxy et CORS sont des impasses, exactement
comme décrit dans le cahier des charges.

La **seule** façon d'extraire un post « dans le contexte de la page YouTube »
sans copier-coller manuel est d'exécuter du code **sur** la page YouTube. Cette
app le fait via un **bookmarklet** (favori) :

1. Onglet **Capturer** → installe le favori « Capturer ce post »
   (glisser-déposer sur ordinateur, ou copier le code sur mobile).
2. Ouvre un post YouTube, touche le favori.
3. Le bookmarklet lit le DOM **localement** (mêmes sélecteurs que `scraper.js`),
   encode le post et ouvre la PWA, qui l'enregistre automatiquement.

Alternative mobile : le menu **Partager** de YouTube → **YT Scraper** (l'app est
enregistrée comme *Web Share Target* une fois installée).

> Le bouton **« Capturer un exemple »** (onglet Capturer) permet de tester tout
> le flux (formatage + export) sans YouTube.

## 🚀 Développement

```bash
cd youtube-scraper-app
npm install
npm run dev      # serveur de dev (PWA activée)
npm run build    # génère les icônes + build de production dans dist/
npm run preview  # sert le build de production
```

## 🧱 Architecture

```
src/
├── components/   HistoryList, MarkdownPreview, ExportSettings,
│                 OverlayDetector, Navbar, Toast
├── hooks/        useYouTubeDetector, useScraper, useLocalStorage, useClipboard
├── utils/        scraper (extraction DOM), parser (nettoyage/entités),
│                 markdown (export MD/TXT/JSON), validators, bookmarklet, db (IndexedDB)
├── App.jsx · main.jsx · index.css
scripts/
└── generate-icons.mjs   génère les PNG (sans dépendance)
```

**Stack** : React 18 · Vite 6 · Tailwind CSS 4 · vite-plugin-pwa (Workbox) ·
IndexedDB (idb). Build < 200 KB (precache), bien sous la cible de 500 KB.
