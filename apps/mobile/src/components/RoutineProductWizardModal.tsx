import React, { useEffect, useMemo, useState } from 'react';
import { Modal, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import type { CombinedSkincareRoutine } from '../lib/skincareRoutine';
import { buildWizardSteps, type RoutinePeriod } from '../lib/skincareRoutine';
import { palette } from '../theme/colors';

type Props = {
  visible: boolean;
  onClose: () => void;
  routine: CombinedSkincareRoutine | null;
  /** When this changes while the modal is open, wizard state resets (e.g. new analysis entry). */
  resetKey?: string;
  /** Fired when the user completes the last pick (same moment the in-modal summary appears). */
  onFinalizeSelections?: (selections: Record<string, number>) => void;
};

export function RoutineProductWizardModal({ visible, onClose, routine, resetKey, onFinalizeSelections }: Props) {
  const insets = useSafeAreaInsets();
  const steps = useMemo(() => (routine ? buildWizardSteps(routine) : []), [routine]);
  const [stepIndex, setStepIndex] = useState(0);
  const [phase, setPhase] = useState<'pick' | 'summary'>('pick');
  const [selections, setSelections] = useState<Record<string, number>>({});

  useEffect(() => {
    if (!visible) return;
    setStepIndex(0);
    setPhase('pick');
    setSelections({});
  }, [visible, resetKey]);

  const current = steps[stepIndex];
  const selectedIdx = current ? selections[current.id] : undefined;
  const canAdvance = current !== undefined && selectedIdx !== undefined;
  const isLastPick = phase === 'pick' && stepIndex >= steps.length - 1;

  function selectOption(idx: number) {
    if (!current) return;
    setSelections((prev) => ({ ...prev, [current.id]: idx }));
  }

  function onPrimary() {
    if (!current || !canAdvance) return;
    if (isLastPick) {
      onFinalizeSelections?.(selections);
      setPhase('summary');
    } else {
      setStepIndex((i) => i + 1);
    }
  }

  function onBack() {
    if (phase === 'summary') {
      setPhase('pick');
      setStepIndex(Math.max(0, steps.length - 1));
      return;
    }
    if (stepIndex > 0) setStepIndex((i) => i - 1);
  }

  function renderSummary() {
    const byPeriod = (period: RoutinePeriod) => steps.filter((s) => s.period === period);
    return (
      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <Pressable
          onPress={() => {
            setPhase('pick');
            setStepIndex(Math.max(0, steps.length - 1));
          }}
          style={({ pressed }) => [styles.editPicks, pressed && { opacity: 0.8 }]}
        >
          <Text style={styles.editPicksText}>← Edit picks</Text>
        </Pressable>
        <Text style={styles.summaryTitle}>Your routine</Text>
        <Text style={styles.summarySub}>Products you picked · how to use each step</Text>

        {(['Morning', 'Night'] as const).map((period) => {
          const list = byPeriod(period);
          if (!list.length) return null;
          return (
            <View key={period} style={styles.summaryBlock}>
              <Text style={styles.summaryPeriod}>{period}</Text>
              {list.map((s) => {
                const idx = selections[s.id];
                const p = idx !== undefined ? s.options[idx] : s.options[0];
                return (
                  <View key={s.id} style={styles.summaryStep}>
                    <Text style={styles.summaryStepTitle}>
                      {s.order}. {s.title}
                    </Text>
                    <Text style={styles.summaryPick}>
                      <Text style={styles.summaryPickLabel}>Your pick: </Text>
                      {p.name}
                    </Text>
                    <Text style={styles.summaryPickNote}>{p.note}</Text>
                    <Text style={styles.summaryHow}>{s.howToUse}</Text>
                  </View>
                );
              })}
            </View>
          );
        })}

        {routine?.cautions.length ? (
          <View style={styles.summaryCautions}>
            <Text style={styles.summaryCautionsTitle}>Important</Text>
            {routine.cautions.map((c) => (
              <Text key={c} style={styles.summaryCautionLine}>
                • {c}
              </Text>
            ))}
          </View>
        ) : null}

        <Pressable onPress={onClose} style={({ pressed }) => [styles.doneBtn, pressed && { opacity: 0.9 }]}>
          <Text style={styles.doneBtnText}>Done</Text>
        </Pressable>
      </ScrollView>
    );
  }

  function renderPick() {
    if (!current || steps.length === 0) {
      return (
        <View style={styles.empty}>
          <Text style={styles.emptyText}>No routine steps to customize.</Text>
        </View>
      );
    }

    return (
      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <Text style={styles.stepProgress}>
          Step {stepIndex + 1} of {steps.length}
        </Text>
        <View style={styles.periodBadge}>
          <Text style={styles.periodBadgeText}>{current.period}</Text>
        </View>
        <Text style={styles.stepTitle}>
          {current.order}. {current.title}
        </Text>
        <Text style={styles.stepHow}>{current.howToUse}</Text>
        <Text style={styles.pickPrompt}>Pick one product</Text>

        {current.options.map((opt, idx) => {
          const selected = selectedIdx === idx;
          return (
            <Pressable
              key={`${current.id}-opt-${idx}`}
              onPress={() => selectOption(idx)}
              style={({ pressed }) => [
                styles.optionCard,
                selected && styles.optionCardSelected,
                pressed && { opacity: 0.92 },
              ]}
            >
              <Text style={styles.optionName}>{opt.name}</Text>
              <Text style={styles.optionNote}>{opt.note}</Text>
            </Pressable>
          );
        })}

        <View style={styles.navRow}>
          <Pressable
            onPress={onBack}
            disabled={phase === 'pick' && stepIndex === 0}
            style={({ pressed }) => [
              styles.navBtn,
              styles.navBtnSecondary,
              phase === 'pick' && stepIndex === 0 && styles.navBtnDisabled,
              pressed && !(phase === 'pick' && stepIndex === 0) && { opacity: 0.88 },
            ]}
          >
            <Text style={styles.navBtnSecondaryText}>Back</Text>
          </Pressable>
          <Pressable
            onPress={onPrimary}
            disabled={!canAdvance}
            style={({ pressed }) => [styles.navBtn, styles.navBtnPrimary, !canAdvance && styles.navBtnDisabled, pressed && canAdvance && { opacity: 0.92 }]}
          >
            <Text style={styles.navBtnPrimaryText}>{isLastPick ? 'See my routine' : 'Next'}</Text>
          </Pressable>
        </View>
      </ScrollView>
    );
  }

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet" onRequestClose={onClose}>
      <View style={[styles.shell, { paddingTop: insets.top + 8, paddingBottom: insets.bottom + 12 }]}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Build routine</Text>
          <Pressable onPress={onClose} hitSlop={12} style={({ pressed }) => [styles.closeBtn, pressed && { opacity: 0.75 }]}>
            <Text style={styles.closeBtnText}>Close</Text>
          </Pressable>
        </View>
        {phase === 'summary' ? renderSummary() : renderPick()}
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  shell: {
    flex: 1,
    backgroundColor: palette.sheetBg,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingBottom: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: palette.sheetHeaderLine,
    backgroundColor: palette.surface,
  },
  headerTitle: { color: palette.ink, fontSize: 17, fontWeight: '800' },
  closeBtn: { paddingVertical: 6, paddingHorizontal: 4 },
  closeBtnText: { color: palette.sageDark, fontSize: 16, fontWeight: '700' },
  scroll: { flex: 1 },
  scrollContent: { paddingHorizontal: 16, paddingTop: 14, paddingBottom: 24 },
  stepProgress: { color: palette.inkSubtle, fontSize: 12, fontWeight: '700', marginBottom: 8 },
  periodBadge: {
    alignSelf: 'flex-start',
    backgroundColor: palette.sageFill18,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
    marginBottom: 10,
  },
  periodBadgeText: { color: palette.sageDark, fontSize: 12, fontWeight: '800' },
  stepTitle: { color: palette.ink, fontSize: 17, fontWeight: '900', marginBottom: 8, lineHeight: 22 },
  stepHow: { color: palette.inkMuted, fontSize: 14, lineHeight: 21, marginBottom: 16 },
  pickPrompt: { color: palette.inkFaint, fontSize: 11, fontWeight: '800', marginBottom: 8, letterSpacing: 0.3 },
  optionCard: {
    borderWidth: 1,
    borderColor: palette.border,
    backgroundColor: palette.surface,
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
  },
  optionCardSelected: {
    borderColor: palette.sage,
    backgroundColor: palette.sageFill12,
  },
  optionName: { color: palette.ink, fontSize: 15, fontWeight: '800', marginBottom: 6 },
  optionNote: { color: palette.inkMuted, fontSize: 13, lineHeight: 19 },
  navRow: { flexDirection: 'row', gap: 12, marginTop: 22 },
  navBtn: { flex: 1, paddingVertical: 14, borderRadius: 12, alignItems: 'center' },
  navBtnSecondary: { backgroundColor: palette.surfaceMuted, borderWidth: 1, borderColor: palette.border },
  navBtnSecondaryText: { color: palette.sageDark, fontSize: 15, fontWeight: '800' },
  navBtnPrimary: { backgroundColor: palette.sage },
  navBtnPrimaryText: { color: palette.onPrimary, fontSize: 15, fontWeight: '900' },
  navBtnDisabled: { opacity: 0.38 },
  editPicks: { alignSelf: 'flex-start', marginBottom: 10, paddingVertical: 4 },
  editPicksText: { color: palette.sageDark, fontSize: 14, fontWeight: '800' },
  summaryTitle: { color: palette.ink, fontSize: 20, fontWeight: '900', marginBottom: 4 },
  summarySub: { color: palette.inkSubtle, fontSize: 13, marginBottom: 18 },
  summaryBlock: { marginBottom: 20 },
  summaryPeriod: {
    color: palette.sageDark,
    fontSize: 14,
    fontWeight: '900',
    marginBottom: 12,
  },
  summaryStep: {
    marginBottom: 16,
    paddingBottom: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: palette.borderSoft,
  },
  summaryStepTitle: { color: palette.ink, fontSize: 15, fontWeight: '900', marginBottom: 6 },
  summaryPick: { color: palette.ink, fontSize: 14, fontWeight: '700', marginBottom: 4 },
  summaryPickLabel: { color: palette.sageDark, fontWeight: '800' },
  summaryPickNote: { color: palette.inkMuted, fontSize: 13, lineHeight: 19, marginBottom: 8 },
  summaryHow: { color: palette.inkMuted, fontSize: 13, lineHeight: 20 },
  summaryCautions: { marginTop: 8, marginBottom: 8 },
  summaryCautionsTitle: { color: palette.caution, fontSize: 13, fontWeight: '900', marginBottom: 8 },
  summaryCautionLine: { color: palette.caution, fontSize: 12, lineHeight: 18, marginTop: 4, opacity: 0.92 },
  doneBtn: {
    marginTop: 12,
    backgroundColor: palette.sage,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  doneBtnText: { color: palette.onPrimary, fontSize: 16, fontWeight: '900' },
  empty: { padding: 24, alignItems: 'center' },
  emptyText: { color: palette.inkMuted, fontSize: 14 },
});
