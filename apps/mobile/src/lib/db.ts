import * as FileSystem from 'expo-file-system/legacy';
import * as SQLite from 'expo-sqlite';

import type { EczemaBucket, RegionScores, SeverityBucket, TimelineEntry } from '../types/models';

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
      heatmapUri TEXT NOT NULL,
      eczemaBucket TEXT NOT NULL DEFAULT 'none',
      eczemaLikelihood REAL NOT NULL DEFAULT 0
    );
  `);

  const cols = await db.getAllAsync<{ name: string }>('PRAGMA table_info(entries)');
  const names = new Set(cols.map((c) => c.name));
  if (!names.has('eczemaBucket')) {
    await db.execAsync(`ALTER TABLE entries ADD COLUMN eczemaBucket TEXT NOT NULL DEFAULT 'none';`);
  }
  if (!names.has('eczemaLikelihood')) {
    await db.execAsync(`ALTER TABLE entries ADD COLUMN eczemaLikelihood REAL NOT NULL DEFAULT 0;`);
  }
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
    `INSERT INTO entries (id, createdAt, imageUri, severityScore, severityBucket, regionScoresJson, heatmapUri, eczemaBucket, eczemaLikelihood)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      entry.id,
      entry.createdAt,
      entry.imageUri,
      entry.severityScore,
      entry.severityBucket,
      JSON.stringify(entry.regionScores),
      entry.heatmapUri,
      entry.eczemaBucket,
      entry.eczemaLikelihood,
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
    eczemaBucket?: EczemaBucket;
    eczemaLikelihood?: number;
  }>(`SELECT * FROM entries ORDER BY createdAt DESC LIMIT ?`, [limit]);

  return rows.map((r) => ({
    id: r.id,
    createdAt: Number(r.createdAt),
    imageUri: r.imageUri,
    severityScore: Number(r.severityScore),
    severityBucket: r.severityBucket,
    eczemaBucket: (r.eczemaBucket ?? 'none') as EczemaBucket,
    eczemaLikelihood: Number(r.eczemaLikelihood ?? 0),
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
    eczemaBucket?: EczemaBucket;
    eczemaLikelihood?: number;
  }>(`SELECT * FROM entries ORDER BY createdAt DESC LIMIT ?`, [n]);

  return rows.map((r) => ({
    id: r.id,
    createdAt: Number(r.createdAt),
    imageUri: r.imageUri,
    severityScore: Number(r.severityScore),
    severityBucket: r.severityBucket,
    eczemaBucket: (r.eczemaBucket ?? 'none') as EczemaBucket,
    eczemaLikelihood: Number(r.eczemaLikelihood ?? 0),
    regionScores: JSON.parse(r.regionScoresJson) as RegionScores,
    heatmapUri: r.heatmapUri,
  }));
}

/** Removes every timeline row and attempts to delete stored heatmap files for those rows. */
export async function clearAllTimelineEntries(): Promise<void> {
  const db = await getDb();
  const rows = await db.getAllAsync<{ heatmapUri: string }>(`SELECT heatmapUri FROM entries`);
  await db.execAsync(`DELETE FROM entries`);
  for (const row of rows) {
    try {
      const info = await FileSystem.getInfoAsync(row.heatmapUri);
      if (info.exists) {
        await FileSystem.deleteAsync(row.heatmapUri, { idempotent: true });
      }
    } catch {
      // ignore per-file errors
    }
  }
}

