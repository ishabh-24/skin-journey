export type SeverityBucket = 'mild' | 'moderate' | 'severe';

export type EczemaBucket = 'none' | 'mild_eczema' | 'severe_eczema';

export type RegionName = 'forehead' | 'left_cheek' | 'right_cheek' | 'jawline';

export type RegionScores = Record<RegionName, number>;

export type AnalyzeResponse = {
  severity_score_0_10: number;
  severity_bucket: SeverityBucket;
  eczema_bucket: EczemaBucket;
  eczema_likelihood_0_10: number;
  components: Record<string, number>;
  region_scores_0_1: RegionScores;
  heatmap_png_base64: string;
};

export type RecommendationResponse = {
  decision: 'otc' | 'derm';
  title: string;
  bullets: string[];
  cautions?: string[];
};

export type TimelineEntry = {
  id: string;
  createdAt: number; // epoch ms
  imageUri: string;
  severityScore: number;
  severityBucket: SeverityBucket;
  eczemaBucket: EczemaBucket;
  eczemaLikelihood: number;
  regionScores: RegionScores;
  heatmapUri: string;
};

