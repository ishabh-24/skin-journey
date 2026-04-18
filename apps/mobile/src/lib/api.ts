import type { AnalyzeResponse, RecommendationResponse, SeverityBucket } from '../types/models';

export async function analyzeImage(opts: { apiBaseUrl: string; imageUri: string }): Promise<AnalyzeResponse> {
  const { apiBaseUrl, imageUri } = opts;
  const url = `${apiBaseUrl.replace(/\/+$/, '')}/analyze`;

  const form = new FormData();
  form.append('image', {
    uri: imageUri,
    name: 'selfie.jpg',
    type: 'image/jpeg',
  } as any);

  const res = await fetch(url, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Analyze failed (${res.status}): ${text || res.statusText}`);
  }

  return (await res.json()) as AnalyzeResponse;
}

export async function getRecommendation(opts: {
  apiBaseUrl: string;
  severityBucket: SeverityBucket;
  worseningStreakDays: number;
  noImprovementDays: number;
  cysticSuspected: boolean;
}): Promise<RecommendationResponse> {
  const { apiBaseUrl, severityBucket, worseningStreakDays, noImprovementDays, cysticSuspected } = opts;
  const url = new URL(`${apiBaseUrl.replace(/\/+$/, '')}/recommend`);
  url.searchParams.set('severity_bucket', severityBucket);
  url.searchParams.set('worsening_streak_days', String(worseningStreakDays));
  url.searchParams.set('no_improvement_days', String(noImprovementDays));
  url.searchParams.set('cystic_suspected', String(cysticSuspected));

  const res = await fetch(url.toString());
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Recommend failed (${res.status}): ${text || res.statusText}`);
  }
  return (await res.json()) as RecommendationResponse;
}

