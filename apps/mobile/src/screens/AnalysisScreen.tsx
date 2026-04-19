import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import React, { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Alert, Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import { RoutineProductWizardModal } from '../components/RoutineProductWizardModal';
import { analyzeImage, getEczemaRecommendation, getRecommendation } from '../lib/api';
import { getLatestEntries, getSetting, insertEntry } from '../lib/db';
import { writeBase64Png } from '../lib/files';
import { buildCombinedSkincareRoutine, buildWizardSteps, type RoutinePeriod } from '../lib/skincareRoutine';
import { computeNoImprovementDays, computeWorseningStreakDays, primaryRegion } from '../lib/trends';
import { palette } from '../theme/colors';
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
  const [routineExpanded, setRoutineExpanded] = useState(false);
  const [routineWizardOpen, setRoutineWizardOpen] = useState(false);
  /** Product index per wizard step id, after user taps "See my routine" — mirrored on this screen. */
  const [routineFinalSelections, setRoutineFinalSelections] = useState<Record<string, number> | null>(null);

  useEffect(() => {
    (async () => {
      const saved = await getSetting('apiBaseUrl');
      if (saved) setApiBaseUrl(saved);
    })();
  }, []);

  const trend = useMemo(() => (entry ? primaryRegion(entry.regionScores) : null), [entry]);

  const combinedRoutine = useMemo(() => {
    if (!entry || !trend) return null;
    return buildCombinedSkincareRoutine({
      severityBucket: entry.severityBucket,
      severityScore: entry.severityScore,
      eczemaBucket: entry.eczemaBucket,
      eczemaLikelihood: entry.eczemaLikelihood,
      regionScores: entry.regionScores,
      primaryZone: trend,
    });
  }, [entry, trend]);

  const wizardSteps = useMemo(() => (combinedRoutine ? buildWizardSteps(combinedRoutine) : []), [combinedRoutine]);

  const finalRoutineReady = useMemo(() => {
    if (!routineFinalSelections || wizardSteps.length === 0) return false;
    return wizardSteps.every((s) => routineFinalSelections[s.id] !== undefined);
  }, [routineFinalSelections, wizardSteps]);

  useEffect(() => {
    setRoutineExpanded(false);
    setRoutineWizardOpen(false);
    setRoutineFinalSelections(null);
  }, [entry?.id]);

  async function onAnalyze() {
    setLoading(true);
    try {
      // Prefer URL from DB so we never analyze with a stale default (e.g. localhost on a
      // physical phone) before Settings hydration finishes.
      const saved = (await getSetting('apiBaseUrl'))?.trim() ?? '';
      const base =
        saved && /^https?:\/\//.test(saved)
          ? saved.replace(/\/+$/, '')
          : apiBaseUrl.trim().replace(/\/+$/, '');
      if (base !== apiBaseUrl) setApiBaseUrl(base);

      const analysis = await analyzeImage({ apiBaseUrl: base, imageUri });

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
        userNote: '',
      };

      const prev = await getLatestEntries(14);
      const nextList = [newEntry, ...prev];
      const worseningStreakDays = computeWorseningStreakDays(nextList);
      const noImprovementDays = computeNoImprovementDays(nextList);

      const [recommendation, eczemaRecommendation] = await Promise.all([
        getRecommendation({
          apiBaseUrl: base,
          severityBucket: newEntry.severityBucket,
          worseningStreakDays,
          noImprovementDays,
          cysticSuspected: false,
        }),
        getEczemaRecommendation({
          apiBaseUrl: base,
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
      <Text style={styles.sub}>Photo captured. Run analysis for severity scores and guidance.</Text>

      <View style={styles.previewCard}>
        <Image source={{ uri: imageUri }} style={styles.previewImg} />
      </View>

      {!entry ? (
        <Pressable style={[styles.primaryBtn, loading && styles.btnDisabled]} onPress={onAnalyze} disabled={loading}>
          {loading ? <ActivityIndicator color={palette.onPrimary} /> : <Text style={styles.primaryBtnText}>Analyze</Text>}
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

          {combinedRoutine ? (
            <View style={styles.routineWrap}>
              <Pressable
                onPress={() => setRoutineExpanded((e) => !e)}
                style={({ pressed }) => [styles.routineToggle, pressed && { opacity: 0.88 }]}
                accessibilityRole="button"
                accessibilityState={{ expanded: routineExpanded }}
              >
                <View style={{ flex: 1 }}>
                  <Text style={styles.routineToggleTitle}>Personalized skincare routine</Text>
                  <Text style={styles.routineToggleSub}>
                    {finalRoutineReady
                      ? 'Your picks are saved on this screen — expand to review'
                      : 'Pick 1 of 3 per step · final summary in a sheet'}
                  </Text>
                </View>
                <Text style={styles.routineChevron}>{routineExpanded ? '▲' : '▼'}</Text>
              </Pressable>

              {routineExpanded ? (
                <View style={styles.routinePanel}>
                  <Text style={styles.routineHeadline}>{combinedRoutine.headline}</Text>
                  <Text style={styles.routineIntro}>{combinedRoutine.intro}</Text>
                  <Text style={styles.routineZone}>{combinedRoutine.zoneLine}</Text>

                  <Text style={styles.routineBuilderHint}>
                    Walk through each morning and night step and choose one of three OTC-style options. When you finish,
                    you will get a clear final routine you can follow.
                  </Text>
                  <Pressable
                    onPress={() => setRoutineWizardOpen(true)}
                    style={({ pressed }) => [styles.routineOpenWizardBtn, pressed && { opacity: 0.9 }]}
                  >
                    <Text style={styles.routineOpenWizardBtnText}>Pick products & build routine</Text>
                  </Pressable>

                  {finalRoutineReady && routineFinalSelections && combinedRoutine ? (
                    <View style={styles.routineSavedBlock}>
                      <Text style={styles.routineSavedTitle}>Your routine (saved here)</Text>
                      <Text style={styles.routineSavedSub}>Products you picked · scroll to review anytime</Text>
                      {(['Morning', 'Night'] as const).map((period: RoutinePeriod) => {
                        const list = wizardSteps.filter((s) => s.period === period);
                        if (!list.length) return null;
                        return (
                          <View key={period} style={styles.routineSavedPeriod}>
                            <Text style={styles.routineSavedPeriodLabel}>{period}</Text>
                            {list.map((s) => {
                              const idx = routineFinalSelections[s.id];
                              const p = s.options[idx];
                              return (
                                <View key={s.id} style={styles.routineSavedStep}>
                                  <Text style={styles.routineSavedStepTitle}>
                                    {s.order}. {s.title}
                                  </Text>
                                  <Text style={styles.routineSavedPick}>
                                    <Text style={styles.routineSavedPickTag}>Your pick: </Text>
                                    {p.name}
                                  </Text>
                                  <Text style={styles.routineSavedPickNote}>{p.note}</Text>
                                  <Text style={styles.routineSavedHow}>{s.howToUse}</Text>
                                </View>
                              );
                            })}
                          </View>
                        );
                      })}
                    </View>
                  ) : null}

                  <Text style={[styles.routineSectionLabel, { marginTop: 18 }]}>Important</Text>
                  {combinedRoutine.cautions.map((c) => (
                    <Text key={c} style={styles.routineCaution}>
                      • {c}
                    </Text>
                  ))}
                </View>
              ) : null}
              <RoutineProductWizardModal
                visible={routineWizardOpen}
                onClose={() => setRoutineWizardOpen(false)}
                routine={combinedRoutine}
                resetKey={entry?.id}
                onFinalizeSelections={(s) => {
                  setRoutineFinalSelections({ ...s });
                  setRoutineExpanded(true);
                }}
              />
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
  root: { flex: 1, backgroundColor: palette.bg },
  content: { padding: 16, paddingBottom: 28 },
  h1: { color: palette.ink, fontSize: 26, fontWeight: '900' },
  sub: { color: palette.inkSubtle, marginTop: 6, marginBottom: 14, fontSize: 13 },
  sectionLabel: {
    color: palette.inkFaint,
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
    borderColor: palette.border,
    backgroundColor: palette.surface,
  },
  previewImg: { width: '100%', height: 360 },
  primaryBtn: {
    marginTop: 14,
    height: 52,
    borderRadius: 14,
    backgroundColor: palette.sage,
    alignItems: 'center',
    justifyContent: 'center',
  },
  btnDisabled: { opacity: 0.65 },
  primaryBtnText: { color: palette.onPrimary, fontSize: 16, fontWeight: '900' },
  results: { marginTop: 16 },
  scoreRow: { flexDirection: 'row', gap: 12 },
  scoreBox: {
    flex: 1,
    padding: 14,
    borderRadius: 16,
    backgroundColor: palette.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.border,
  },
  scoreLabel: { color: palette.inkSubtle, fontSize: 12, fontWeight: '700' },
  scoreValue: { color: palette.sageDark, fontSize: 34, fontWeight: '900', marginTop: 6, lineHeight: 38 },
  scoreValueSmall: { color: palette.ink, fontSize: 18, fontWeight: '900', marginTop: 10 },
  scoreHint: { color: palette.inkMuted, fontSize: 12, marginTop: 6 },
  card: {
    marginTop: 12,
    padding: 14,
    borderRadius: 16,
    backgroundColor: palette.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.border,
  },
  cardTitle: { color: palette.ink, fontSize: 14, fontWeight: '900', marginBottom: 8 },
  cardBullet: { color: palette.inkMuted, fontSize: 13, marginTop: 4, lineHeight: 18 },
  cardCaution: { color: palette.cardCautionMuted, fontSize: 12, marginTop: 4 },
  secondaryBtn: {
    marginTop: 14,
    height: 48,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: palette.border,
    backgroundColor: palette.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryBtnText: { color: palette.sageDark, fontSize: 14, fontWeight: '800' },
  routineWrap: { marginTop: 16 },
  routineToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 14,
    paddingHorizontal: 14,
    borderRadius: 16,
    backgroundColor: palette.sageFill12,
    borderWidth: 1,
    borderColor: palette.border,
    gap: 10,
  },
  routineToggleTitle: { color: palette.ink, fontSize: 15, fontWeight: '900' },
  routineToggleSub: { color: palette.inkSubtle, fontSize: 12, marginTop: 4, lineHeight: 16 },
  routineChevron: { color: palette.sageDark, fontSize: 16, fontWeight: '900', paddingLeft: 4 },
  routinePanel: {
    marginTop: 10,
    padding: 14,
    borderRadius: 16,
    backgroundColor: palette.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.borderSoft,
  },
  routineHeadline: { color: palette.ink, fontSize: 16, fontWeight: '900', marginBottom: 10 },
  routineIntro: { color: palette.inkMuted, fontSize: 13, lineHeight: 20, marginBottom: 10 },
  routineZone: { color: palette.sageDark, fontSize: 13, lineHeight: 19, marginBottom: 6, fontStyle: 'italic' },
  routineBuilderHint: {
    color: palette.inkMuted,
    fontSize: 13,
    lineHeight: 20,
    marginBottom: 14,
  },
  routineOpenWizardBtn: {
    backgroundColor: palette.sage,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  routineOpenWizardBtnText: { color: palette.onPrimary, fontSize: 15, fontWeight: '900' },
  routineSavedBlock: {
    marginTop: 18,
    padding: 14,
    borderRadius: 14,
    backgroundColor: palette.sageFill8,
    borderWidth: 1,
    borderColor: palette.border,
  },
  routineSavedTitle: { color: palette.ink, fontSize: 16, fontWeight: '900', marginBottom: 4 },
  routineSavedSub: { color: palette.inkSubtle, fontSize: 12, lineHeight: 17, marginBottom: 14 },
  routineSavedPeriod: { marginBottom: 14 },
  routineSavedPeriodLabel: {
    color: palette.sageDark,
    fontSize: 13,
    fontWeight: '900',
    marginBottom: 10,
  },
  routineSavedStep: {
    marginBottom: 14,
    paddingBottom: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: palette.borderSoft,
  },
  routineSavedStepTitle: { color: palette.ink, fontSize: 14, fontWeight: '900', marginBottom: 6 },
  routineSavedPick: { color: palette.ink, fontSize: 13, fontWeight: '700', marginBottom: 4 },
  routineSavedPickTag: { color: palette.sageDark, fontWeight: '800' },
  routineSavedPickNote: { color: palette.inkMuted, fontSize: 12, lineHeight: 17, marginBottom: 8 },
  routineSavedHow: { color: palette.inkMuted, fontSize: 12, lineHeight: 18 },
  routineSectionLabel: {
    color: palette.inkFaint,
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 0.8,
    marginBottom: 10,
    marginTop: 4,
  },
  routineStep: {
    marginBottom: 16,
    paddingBottom: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: palette.borderSoft,
  },
  routineStepTitle: { color: palette.ink, fontSize: 14, fontWeight: '900', marginBottom: 6 },
  routineStepHow: { color: palette.inkMuted, fontSize: 13, lineHeight: 19, marginBottom: 8 },
  routineProductsLabel: { color: palette.inkFaint, fontSize: 11, fontWeight: '800', marginBottom: 4 },
  routineProductLine: { color: palette.inkMuted, fontSize: 12, lineHeight: 18, marginTop: 3 },
  routineCaution: { color: palette.caution, fontSize: 12, lineHeight: 18, marginTop: 6 },
});

