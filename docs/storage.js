/**
 * IndexedDB persistence layer.
 *
 * Unlike the server version, there's no per-request cost to avoid here --
 * a browser tab holds the whole session in memory anyway -- so this module
 * only handles durable storage (writes, and one bulk load on startup).
 * app.js keeps the authoritative in-memory copy (movies/comparisons array,
 * model.js caches) and calls through here on every mutation; nothing reads
 * IndexedDB back out mid-session.
 */

const DB_NAME = "letterboxd-ranker";
const DB_VERSION = 1;

function idbRequest(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function txDone(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

let dbPromise = null;

export function openDB() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains("movies")) {
        db.createObjectStore("movies", { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains("comparisons")) {
        db.createObjectStore("comparisons", { keyPath: "id", autoIncrement: true });
      }
      if (!db.objectStoreNames.contains("modelRuns")) {
        db.createObjectStore("modelRuns", { keyPath: "id", autoIncrement: true });
      }
      if (!db.objectStoreNames.contains("meta")) {
        db.createObjectStore("meta", { keyPath: "key" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

async function storeAll(name) {
  const db = await openDB();
  const tx = db.transaction(name, "readonly");
  const result = await idbRequest(tx.objectStore(name).getAll());
  return result;
}

/** Everything currently in IndexedDB, as plain arrays -- called once on
 * startup to populate the in-memory session state. */
export async function loadAll() {
  const [movies, comparisons, modelRuns, metaRows] = await Promise.all([
    storeAll("movies"),
    storeAll("comparisons"),
    storeAll("modelRuns"),
    storeAll("meta"),
  ]);
  const meta = new Map(metaRows.map((r) => [r.key, r.value]));
  return { movies, comparisons, modelRuns, meta };
}

/** Insert or update movies by id (see model.js's movieKey). */
export async function putMovies(movies) {
  const db = await openDB();
  const tx = db.transaction("movies", "readwrite");
  const store = tx.objectStore("movies");
  for (const m of movies) store.put(m);
  await txDone(tx);
}

/** Add a new comparison row; returns the auto-assigned id. */
export async function addComparison(comparison) {
  const db = await openDB();
  const tx = db.transaction("comparisons", "readwrite");
  const id = await idbRequest(tx.objectStore("comparisons").add(comparison));
  await txDone(tx);
  return id;
}

/** Overwrite a comparison row in place (used for winner reassignment). */
export async function putComparison(comparison) {
  const db = await openDB();
  const tx = db.transaction("comparisons", "readwrite");
  tx.objectStore("comparisons").put(comparison);
  await txDone(tx);
}

export async function deleteComparison(id) {
  const db = await openDB();
  const tx = db.transaction("comparisons", "readwrite");
  tx.objectStore("comparisons").delete(id);
  await txDone(tx);
}

/** Bulk-insert comparisons (used by seeding and JSON import); returns the
 * auto-assigned ids in the same order. */
export async function addComparisons(comparisons) {
  const db = await openDB();
  const tx = db.transaction("comparisons", "readwrite");
  const store = tx.objectStore("comparisons");
  const ids = comparisons.map((c) => idbRequest(store.add(c)));
  const resolved = await Promise.all(ids);
  await txDone(tx);
  return resolved;
}

export async function addModelRun(run) {
  const db = await openDB();
  const tx = db.transaction("modelRuns", "readwrite");
  const id = await idbRequest(tx.objectStore("modelRuns").add(run));
  await txDone(tx);
  return id;
}

export async function setMeta(key, value) {
  const db = await openDB();
  const tx = db.transaction("meta", "readwrite");
  tx.objectStore("meta").put({ key, value });
  await txDone(tx);
}

/** Wipe every store -- used before a full JSON import (see app.js), not
 * exposed as a general-purpose "reset" action. */
export async function clearAll() {
  const db = await openDB();
  const tx = db.transaction(["movies", "comparisons", "modelRuns", "meta"], "readwrite");
  for (const name of ["movies", "comparisons", "modelRuns", "meta"]) tx.objectStore(name).clear();
  await txDone(tx);
}
