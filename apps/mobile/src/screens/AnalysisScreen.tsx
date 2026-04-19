import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import React, { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Alert, Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import { analyzeImage, getEczemaRecommendation, getRecommendation } from '../lib/api';
import { getLatestEntries, getSetting, insertEntry } from '../lib/db';
import { writeBase64Png } from '../lib/files';
import { computeNoImprovementDays, computeWorseningStreakDays, primaryRegion } from '../lib/trends';
import type { EczemaBucket, TimelineEntry } from '../types/models';
import type { CaptureStackParamList } from '../types/nav';

type Props = NativeStackScreenProps<CaptureStackParamList, 'Analysis'>;

const DEFAULT_API_BASE_URL = 'http://localhost:8000';

function eczemaBucketLabel(b: EczemaBucket): string {
  if (b === 'none') return 'No eczema pattern';
  if (b === 'mild_eczema') return 'Mild eczema-type';
  return 'Severe eczema-type';
}

export function AnalysisScreen({ route, navigation }: Props) {
  const { imageUri } = route.params;
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE_URL);
  const [loading, setLoading] = useState(false);
  const [entry, setEntry] = useState<TimelineEntry | null>(null);
  const [rec, setRec] = useState<{ title: string; decision: string; bullets: string[]; cautions: string[] } | null>(
    null,
  );
  const [eczemaRec, setEczemaRec] = useState<{
    title: string;
    decision: string;
    bullets: string[];
    cautions: string[];
  } | null>(null);

  useEffect(() => {
    (async () => {
      const saved = await getSetting('apiBaseUrl');
      if (saved) setApiBaseUrl(saved);
    })();
  }, []);

  const trend = useMemo(() => (entry ? primaryRegion(entry.regionScores) : null), [entry]);

  async function onAnalyze() {
    setLoading(true);
    try {
      const analysis = await analyzeImage({ apiBaseUrl, imageUri });

      const heatmapUri = await writeBase64Png({
        base64: analysis.heatmap_png_base64,
        filename: `heat-${Date.now()}.png`,
      });

      const newEntry: TimelineEntry = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        createdAt: Date.now(),
        imageUri,
        severityScore: analysis.severity_score_0_10,
        severityBucket: analysis.severity_bucket,
        eczemaBucket: analysis.eczema_bucket,
        eczemaLikelihood: analysis.eczema_likelihood_0_10,
        regionScores: analysis.region_scores_0_1,
        heatmapUri,
      };

      const prev = await getLatestEntries(14);
      const nextList = [newEntry, ...prev];
      const worseningStreakDays = computeWorseningStreakDays(nextList);
      const noImprovementDays = computeNoImprovementDays(nextList);

      const [recommendation, eczemaRecommendation] = await Promise.all([
        getRecommendation({
          apiBaseUrl,
          severityBucket: newEntry.severityBucket,
          worseningStreakDays,
          noImprovementDays,
          cysticSuspected: false,
        }),
        getEczemaRecommendation({
          apiBaseUrl,
          eczemaBucket: analysis.eczema_bucket,
        }),
      ]);

      await insertEntry(newEntry);

      setEntry(newEntry);
      setRec({
        title: recommendation.title,
        decision: recommendation.decision,
        bullets: recommendation.bullets,
        cautions: recommendation.cautions ?? [],
      });
      setEczemaRec({
        title: eczemaRecommendation.title,
        decision: eczemaRecommendation.decision,
        bullets: eczemaRecommendation.bullets,
        cautions: eczemaRecommendation.cautions ?? [],
      });
    } catch (e: any) {
      Alert.alert('Analysis failed', e?.message ?? 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <ScrollView style={styles.root} contentContainerStyle={styles.content}>
      <Text style={styles.h1}>Analysis</Text>
      <Text style={styles.sub}>Photo captured. Run analysis to score severity and generate a heatmap.</Text>

      <View style={styles.previewCard}>
        <Image source={{ uri: imageUri }} style={styles.previewImg} />
        {entry?.heatmapUri ? <Image source={{ uri: entry.heatmapUri }} style={styles.heatmap} /> : null}
      </View>

      {!entry ? (
        <Pressable style={[styles.primaryBtn, loading && styles.btnDisabled]} onPress={onAnalyze} disabled={loading}>
          {loading ? <ActivityIndicator color="white" /> : <Text style={styles.primaryBtnText}>Analyze</Text>}
        </Pressable>
      ) : (
        <View style={styles.results}>
          <Text style={styles.sectionLabel}>Acne (inflammation / lesions)</Text>
          <View style={styles.scoreRow}>
            <View style={styles.scoreBox}>
              <Text style={styles.scoreLabel}>Severity</Text>
              <Text style={styles.scoreValue}>{entry.severityScore.toFixed(1)}</Text>
              <Text style={styles.scoreHint}>{entry.severityBucket.toUpperCase()}</Text>
            </View>
            <View style={styles.scoreBox}>
              <Text style={styles.scoreLabel}>Primary zone</Text>
              <Text style={styles.scoreValueSmall}>{trend?.replace('_', ' ')}</Text>
              <Text style={styles.scoreHint}>highest inflammation</Text>
            </View>
          </View>

          {rec ? (
            <View style={styles.card}>
              <Text style={styles.cardTitle}>
                Acne · {rec.decision === 'derm' ? 'Dermatology' : 'OTC'}: {rec.title}
              </Text>
              {rec.bullets.map((b) => (
                <Text key={b} style={styles.cardBullet}>
                  • {b}
                </Text>
              ))}
              {rec.cautions.length ? <View style={{ height: 8 }} /> : null}
              {rec.cautions.map((c) => (
                <Text key={c} style={styles.cardCaution}>
                  {c}
                </Text>
              ))}
            </View>
          ) : null}

          <Text style={[styles.sectionLabel, { marginTop: 18 }]}>Eczema-type pattern</Text>
          <View style={styles.scoreBox}>
            <Text style={styles.scoreLabel}>Assessment</Text>
            <Text style={styles.scoreValueSmall}>{eczemaBucketLabel(entry.eczemaBucket)}</Text>
            <Text style={styles.scoreHint}>likelihood {entry.eczemaLikelihood.toFixed(1)} / 10</Text>
          </View>

          {eczemaRec ? (
            <View style={styles.card}>
              <Text style={styles.cardTitle}>
                Eczema care · {eczemaRec.decision === 'derm' ? 'Dermatology' : 'OTC'}: {eczemaRec.title}
              </Text>
              {eczemaRec.bullets.map((b) => (
                <Text key={b} style={styles.cardBullet}>
                  • {b}
                </Text>
              ))}
              {eczemaRec.cautions.length ? <View style={{ height: 8 }} /> : null}
              {eczemaRec.cautions.map((c) => (
                <Text key={c} style={styles.cardCaution}>
                  {c}
                </Text>
              ))}
            </View>
          ) : null}

          <Pressable style={styles.secondaryBtn} onPress={() => navigation.popToTop()}>
            <Text style={styles.secondaryBtnText}>Capture another</Text>
          </Pressable>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0B0B10' },
  content: { padding: 16, paddingBottom: 28 },
  h1: { color: 'white', fontSize: 26, fontWeight: '900' },
  sub: { color: 'rgba(255,255,255,0.78)', marginTop: 6, marginBottom: 14, fontSize: 13 },
  sectionLabel: {
    color: 'rgba(255,255,255,0.55)',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginBottom: 8,
  },
  previewCard: {
    borderRadius: 18,
    overflow: 'hidden',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.16)',
    backgroundColor: 'rgba(255,255,255,0.06)',
  },
  previewImg: { width: '100%', height: 360 },
  heatmap: { position: 'absolute', left: 0, top: 0, right: 0, bottom: 0, opacity: 0.65 },
  primaryBtn: {
    marginTop: 14,
    height: 52,
    borderRadius: 14,
    backgroundColor: '#6C5CE7',
    alignItems: 'center',
    justifyContent: 'center',
  },
  btnDisabled: { opacity: 0.65 },
  primaryBtnText: { color: 'white', fontSize: 16, fontWeight: '900' },
  results: { marginTop: 16 },
  scoreRow: { flexDirection: 'row', gap: 12 },
  scoreBox: {
    flex: 1,
    padding: 14,
    borderRadius: 16,
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.14)',
  },
  scoreLabel: { color: 'rgba(255,255,255,0.75)', fontSize: 12, fontWeight: '700' },
  scoreValue: { color: 'white', fontSize: 34, fontWeight: '900', marginTop: 6, lineHeight: 38 },
  scoreValueSmall: { color: 'white', fontSize: 18, fontWeight: '900', marginTop: 10 },
  scoreHint: { color: 'rgba(255,255,255,0.7)', fontSize: 12, marginTop: 6 },
  card: {
    marginTop: 12,
    padding: 14,
    borderRadius: 16,
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.14)',
  },
  cardTitle: { color: 'white', fontSize: 14, fontWeight: '900', marginBottom: 8 },
  cardBullet: { color: 'rgba(255,255,255,0.88)', fontSize: 13, marginTop: 4, lineHeight: 18 },
  cardCaution: { color: 'rgba(255,255,255,0.7)', fontSize: 12, marginTop: 4 },
  secondaryBtn: {
    marginTop: 14,
    height: 48,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.22)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryBtnText: { color: 'white', fontSize: 14, fontWeight: '800' },
});

