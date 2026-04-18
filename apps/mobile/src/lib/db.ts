import * as SQLite from 'expo-sqlite';

import type { RegionScores, SeverityBucket, TimelineEntry } from '../types/models';

let dbPromise: Promise<SQLite.SQLiteDatabase> | null = null;

async function getDb() {
  if (!dbPromise) dbPromise = SQLite.openDatabaseAsync('skinjourney.db');
  return dbPromise;
}

export async function initDb() {
  const db = await getDb();
  await db.execAsync(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS settings (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS entries (
      id TEXT PRIMARY KEY,
      createdAt INTEGER NOT NULL,
      imageUri TEXT NOT NULL,
      severityScore REAL NOT NULL,
      severityBucket TEXT NOT NULL,
      regionScoresJson TEXT NOT NULL,
      heatmapUri TEXT NOT NULL
    );
  `);
}

export async function getSetting(key: string): Promise<string | null> {
  const db = await getDb();
  const row = await db.getFirstAsync<{ value: string }>(
    `SELECT value FROM settings WHERE key = ?`,
    [key],
  );
  return row?.value ?? null;
}

export async function setSetting(key: string, value: string): Promise<void> {
  const db = await getDb();
  await db.runAsync(
    `INSERT INTO settings (key, value) VALUES (?, ?)
     ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
    [key, value],
  );
}

export async function insertEntry(entry: TimelineEntry): Promise<void> {
  const db = await getDb();
  await db.runAsync(
    `INSERT INTO entries (id, createdAt, imageUri, severityScore, severityBucket, regionScoresJson, heatmapUri)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
    [
      entry.id,
      entry.createdAt,
      entry.imageUri,
      entry.severityScore,
      entry.severityBucket,
      JSON.stringify(entry.regionScores),
      entry.heatmapUri,
    ],
  );
}

export async function listEntries(limit = 60): Promise<TimelineEntry[]> {
  const db = await getDb();
  const rows = await db.getAllAsync<{
    id: string;
    createdAt: number;
    imageUri: string;
    severityScore: number;
    severityBucket: SeverityBucket;
    regionScoresJson: string;
    heatmapUri: string;
  }>(`SELECT * FROM entries ORDER BY createdAt DESC LIMIT ?`, [limit]);

  return rows.map((r) => ({
    id: r.id,
    createdAt: Number(r.createdAt),
    imageUri: r.imageUri,
    severityScore: Number(r.severityScore),
    severityBucket: r.severityBucket,
    regionScores: JSON.parse(r.regionScoresJson) as RegionScores,
    heatmapUri: r.heatmapUri,
  }));
}

export async function getLatestEntries(n = 14): Promise<TimelineEntry[]> {
  const db = await getDb();
  const rows = await db.getAllAsync<{
    id: string;
    createdAt: number;
    imageUri: string;
    severityScore: number;
    severityBucket: SeverityBucket;
    regionScoresJson: string;
    heatmapUri: string;
  }>(`SELECT * FROM entries ORDER BY createdAt DESC LIMIT ?`, [n]);

  return rows.map((r) => ({
    id: r.id,
    createdAt: Number(r.createdAt),
    imageUri: r.imageUri,
    severityScore: Number(r.severityScore),
    severityBucket: r.severityBucket,
    regionScores: JSON.parse(r.regionScoresJson) as RegionScores,
    heatmapUri: r.heatmapUri,
  }));
}

