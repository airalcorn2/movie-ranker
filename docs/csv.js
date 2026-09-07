/**
 * Letterboxd diary.csv / ratings.csv parsing, client-side.
 *
 * Ports ingest.py's merge logic (dedupe by title+year, newest-row-wins,
 * a later unrated/unwatched row doesn't blank out a previously known
 * value) so the static app can read the same export files directly from
 * a <input type="file"> instead of a server-side data/ directory.
 */

import { movieKey } from "./model.js";

/** A minimal RFC4180-ish CSV parser: quoted fields, "" escaping, embedded
 * newlines/commas inside quotes, CRLF or LF, and a leading BOM (Letterboxd
 * exports one). Returns an array of rows, each an array of field strings. */
function parseCsv(text) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1); // Strip BOM.
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += c;
      }
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\r") {
      // Skip; the paired \n (or its absence) ends the row below.
    } else if (c === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += c;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((r) => !(r.length === 1 && r[0] === ""));
}

/** CSV rows -> array of objects keyed by the header row. */
function toRecords(rows) {
  if (rows.length === 0) return [];
  const [header, ...rest] = rows;
  return rest.map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ""])));
}

function parseYear(value) {
  const v = (value ?? "").trim();
  if (!v) return null;
  const n = parseInt(v, 10);
  return Number.isNaN(n) ? null : n;
}

function parseRating(value) {
  const v = (value ?? "").trim();
  if (!v) return null;
  const n = parseFloat(v);
  return Number.isNaN(n) ? null : n;
}

function parseDate(value) {
  const v = (value ?? "").trim();
  if (!v || !/^\d{4}-\d{2}-\d{2}$/.test(v)) return null;
  const d = new Date(`${v}T00:00:00Z`);
  return Number.isNaN(d.getTime()) ? null : d;
}

/** One diary.csv or ratings.csv file's text -> normalized rows. */
function readRows(text, watchedDateCol) {
  return toRecords(parseCsv(text)).flatMap((raw) => {
    const title = (raw["Name"] ?? "").trim();
    if (!title) return [];
    const year = parseYear(raw["Year"]);
    const rating = parseRating(raw["Rating"]);
    const watchedDate = (raw[watchedDateCol] ?? "").trim() || null;
    const sortDate = parseDate(watchedDate) ?? parseDate(raw["Date"]);
    return [{ sortDate, title, year, rating, watchedDate }];
  });
}

/** Merge diary.csv and/or ratings.csv text into upsert-ready movie records
 * (one per distinct title+year, newest-dated row's values win, without a
 * later unrated/unwatched row blanking out a previously known value).
 * Either argument may be null/omitted if that file wasn't provided. */
export function ingestText(diaryText, ratingsText) {
  const rows = [
    ...(diaryText ? readRows(diaryText, "Watched Date") : []),
    ...(ratingsText ? readRows(ratingsText, "Date") : []),
  ];
  if (rows.length === 0) return [];

  rows.sort((a, b) => (a.sortDate?.getTime() ?? -Infinity) - (b.sortDate?.getTime() ?? -Infinity));

  const merged = new Map(); // "title.toLowerCase()|year" -> record
  for (const row of rows) {
    const key = `${row.title.toLowerCase()}|${row.year}`;
    const existing = merged.get(key) ?? { title: row.title, year: row.year, rating: null, watchedDate: null };
    if (row.rating != null) existing.rating = row.rating;
    if (row.watchedDate != null) existing.watchedDate = row.watchedDate;
    merged.set(key, existing);
  }

  return [...merged.values()].map((r) => ({
    id: movieKey(r.title, r.year),
    title: r.title,
    year: r.year,
    rating: r.rating,
    watchedDate: r.watchedDate,
    posterUrl: null,
  }));
}
