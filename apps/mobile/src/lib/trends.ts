import type { TimelineEntry } from '../types/models';

function isCloseInTime(aMs: number, bMs: number) {
  return Math.abs(aMs - bMs) <= 36 * 60 * 60 * 1000;
}

export function computeWorseningStreakDays(entriesDesc: TimelineEntry[]): number {
  if (entriesDesc.length < 2) return 0;
  let streak = 0;
  for (let i = 0; i < entriesDesc.length - 1; i++) {
    const cur = entriesDesc[i];
    const prev = entriesDesc[i + 1];
    if (!isCloseInTime(cur.createdAt, prev.createdAt)) break;
    if (cur.severityScore > prev.severityScore + 0.2) {
      streak += 1;
    } else {
      break;
    }
  }
  return streak;
}

export function computeNoImprovementDays(entriesDesc: TimelineEntry[]): number {
  if (entriesDesc.length < 2) return 0;
  const latest = entriesDesc[0];
  // Find the last time we were meaningfully better than today.
  for (let i = 1; i < entriesDesc.length; i++) {
    const e = entriesDesc[i];
    if (e.severityScore <= latest.severityScore - 0.5) {
      const days = Math.round((latest.createdAt - e.createdAt) / (24 * 60 * 60 * 1000));
      return Math.max(0, days);
    }
  }
  // If never better in history window, approximate from oldest entry.
  const oldest = entriesDesc[entriesDesc.length - 1];
  const days = Math.round((latest.createdAt - oldest.createdAt) / (24 * 60 * 60 * 1000));
  return Math.max(0, days);
}

export function computeTrendLabel(entriesDesc: TimelineEntry[]): 'improving' | 'worsening' | 'stable' {
  if (entriesDesc.length < 3) return 'stable';
  const a = entriesDesc[0].severityScore;
  const b = entriesDesc[Math.min(2, entriesDesc.length - 1)].severityScore;
  const delta = a - b;
  if (delta <= -0.6) return 'worsening';
  if (delta >= 0.6) return 'improving';
  return 'stable';
}

export function primaryRegion(regionScores: TimelineEntry['regionScores']): string {
  let best = { name: 'forehead', v: -1 };
  for (const [k, v] of Object.entries(regionScores)) {
    if (v > best.v) best = { name: k, v };
  }
  return best.name;
}

