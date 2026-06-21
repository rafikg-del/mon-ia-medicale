// Couche de persistance IndexedDB (via idb).
// - jusqu'à 100 posts conservés
// - purge automatique après 30 jours

import { openDB } from "idb";

const DB_NAME = "yt-scraper";
const STORE = "posts";
const MAX_POSTS = 100;
const MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000; // 30 jours

let _dbPromise = null;

function getDB() {
  if (!_dbPromise) {
    _dbPromise = openDB(DB_NAME, 1, {
      upgrade(db) {
        if (!db.objectStoreNames.contains(STORE)) {
          const store = db.createObjectStore(STORE, { keyPath: "id" });
          store.createIndex("capturedAt", "capturedAt");
          store.createIndex("author", "author");
        }
      },
    });
  }
  return _dbPromise;
}

function newId() {
  return (
    Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
  ).toUpperCase();
}

/** Purge les posts trop vieux et applique la limite de quantité. */
async function enforceLimits(db) {
  const tx = db.transaction(STORE, "readwrite");
  const all = await tx.store.getAll();
  const now = Date.now();
  const fresh = all.filter((p) => now - (p.capturedAt || 0) <= MAX_AGE_MS);
  const stale = all.filter((p) => now - (p.capturedAt || 0) > MAX_AGE_MS);
  for (const p of stale) await tx.store.delete(p.id);

  if (fresh.length > MAX_POSTS) {
    fresh.sort((a, b) => (b.capturedAt || 0) - (a.capturedAt || 0));
    for (const p of fresh.slice(MAX_POSTS)) await tx.store.delete(p.id);
  }
  await tx.done;
}

export async function getAllPosts() {
  const db = await getDB();
  const posts = await db.getAllFromIndex(STORE, "capturedAt");
  return posts.sort((a, b) => (b.capturedAt || 0) - (a.capturedAt || 0));
}

export async function savePost(post) {
  const db = await getDB();
  const record = {
    id: post.id || newId(),
    content: post.content || "",
    author: post.author || "",
    date: post.date || "",
    url: post.url || "",
    engagement: post.engagement || {},
    tags: post.tags || [],
    capturedAt: post.capturedAt || Date.now(),
  };

  // Déduplication : même URL + même contenu = on met à jour au lieu de dupliquer.
  if (record.url || record.content) {
    const all = await db.getAll(STORE);
    const dup = all.find(
      (p) => p.url === record.url && p.content === record.content,
    );
    if (dup) record.id = dup.id;
  }

  await db.put(STORE, record);
  await enforceLimits(db);
  return record;
}

export async function deletePost(id) {
  const db = await getDB();
  await db.delete(STORE, id);
}

export async function clearAllPosts() {
  const db = await getDB();
  await db.clear(STORE);
}

export async function updatePost(id, patch) {
  const db = await getDB();
  const existing = await db.get(STORE, id);
  if (!existing) return null;
  const updated = { ...existing, ...patch, id };
  await db.put(STORE, updated);
  return updated;
}
