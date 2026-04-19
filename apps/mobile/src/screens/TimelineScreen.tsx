import { useFocusEffect } from '@react-navigation/native';
import React, { useCallback, useMemo, useState } from 'react';
import {
  Alert,
  FlatList,
  Image,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  RefreshControl,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { clearAllTimelineEntries, listEntries, updateEntryFields } from '../lib/db';
import { formatMsToYmd, parseYmdToLocalMs } from '../lib/entryDate';
import { computeFlareEntryIds, computeTrendLabel, primaryRegion } from '../lib/trends';
import { palette } from '../theme/colors';
import type { TimelineEntry } from '../types/models';

function fmtDate(ms: number) {
  const d = new Date(ms);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function TimelineScreen() {
  const [entries, setEntries] = useState<TimelineEntry[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [editEntry, setEditEntry] = useState<TimelineEntry | null>(null);
  const [draftDate, setDraftDate] = useState('');
  const [draftNote, setDraftNote] = useState('');

  const trend = useMemo(() => computeTrendLabel(entries.slice(0, 7)), [entries]);
  const flareIds = useMemo(() => computeFlareEntryIds(entries), [entries]);

  const load = useCallback(async () => {
    const rows = await listEntries(60);
    setEntries(rows);
  }, []);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load]),
  );

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await load();
    } finally {
      setRefreshing(false);
    }
  }, [load]);

  const onClearHistory = useCallback(() => {
    Alert.alert(
      'Clear timeline?',
      'Removes all saved analysis entries and heatmaps from this device. Your camera roll photos are not deleted.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear all',
          style: 'destructive',
          onPress: () => {
            void (async () => {
              try {
                await clearAllTimelineEntries();
                await load();
              } catch (e: unknown) {
                const msg = e instanceof Error ? e.message : 'Unknown error';
                Alert.alert('Could not clear', msg);
              }
            })();
          },
        },
      ],
    );
  }, [load]);

  const openEdit = useCallback((e: TimelineEntry) => {
    setEditEntry(e);
    setDraftDate(formatMsToYmd(e.createdAt));
    setDraftNote(e.userNote ?? '');
  }, []);

  const closeEdit = useCallback(() => {
    setEditEntry(null);
    setDraftDate('');
    setDraftNote('');
  }, []);

  const saveEdit = useCallback(() => {
    if (!editEntry) return;
    const t = parseYmdToLocalMs(draftDate);
    if (t === null) {
      Alert.alert('Invalid date', 'Use YYYY-MM-DD (e.g. 2026-04-18).');
      return;
    }
    void (async () => {
      try {
        await updateEntryFields(editEntry.id, { createdAt: t, userNote: draftNote });
        await load();
        closeEdit();
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Unknown error';
        Alert.alert('Could not save', msg);
      }
    })();
  }, [editEntry, draftDate, draftNote, load, closeEdit]);

  return (
    <View style={styles.root}>
      <View style={styles.header}>
        <View style={styles.headerTop}>
          <Text style={styles.h1}>Timeline</Text>
          {entries.length > 0 ? (
            <Pressable onPress={onClearHistory} style={({ pressed }) => [styles.clearBtn, pressed && styles.clearBtnPressed]}>
              <Text style={styles.clearBtnText}>Clear history</Text>
            </Pressable>
          ) : null}
        </View>
        <Text style={styles.sub}>
          Trend: <Text style={styles.subStrong}>{trend}</Text>
        </Text>
        <Text style={styles.hint}>Tap an entry to log possible causes or correct its date. FLARE marks a clear acne worsening vs your previous shot.</Text>
      </View>

      <FlatList
        data={entries}
        keyExtractor={(e) => e.id}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={palette.sage} />}
        contentContainerStyle={styles.list}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyTitle}>No entries yet</Text>
            <Text style={styles.emptyBody}>Capture a daily photo to start your skin timeline.</Text>
          </View>
        }
        renderItem={({ item }) => (
          <Pressable
            onPress={() => openEdit(item)}
            style={({ pressed }) => [styles.row, pressed && styles.rowPressed]}
            android_ripple={{ color: palette.sageFill18 }}
          >
            <View style={styles.thumbWrap}>
              <Image source={{ uri: item.imageUri }} style={styles.thumb} />
            </View>
            <View style={{ flex: 1 }}>
              <View style={styles.titleRow}>
                <Text style={styles.rowTitle}>
                  {fmtDate(item.createdAt)} · {item.severityBucket.toUpperCase()}
                </Text>
                {flareIds.has(item.id) ? (
                  <View style={styles.flarePill}>
                    <Text style={styles.flarePillText}>FLARE</Text>
                  </View>
                ) : null}
              </View>
              <Text style={styles.rowBody}>
                Acne {item.severityScore.toFixed(1)} ({item.severityBucket}) · Eczema {item.eczemaLikelihood.toFixed(1)} (
                {item.eczemaBucket.replace(/_/g, ' ')})
              </Text>
              <Text style={[styles.rowBody, { marginTop: 2 }]}>
                Primary zone {primaryRegion(item.regionScores).replace('_', ' ')}
              </Text>
              {item.userNote?.trim() ? (
                <Text style={styles.notePreview} numberOfLines={2}>
                  Note: {item.userNote.trim()}
                </Text>
              ) : null}
              <MiniBars entries={entries} highlightId={item.id} />
            </View>
          </Pressable>
        )}
      />

      <Modal visible={editEntry !== null} animationType="fade" transparent onRequestClose={closeEdit}>
        <View style={styles.modalRoot}>
          <Pressable style={StyleSheet.absoluteFill} onPress={closeEdit} accessibilityLabel="Dismiss" />
          <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : undefined}
            style={styles.modalKav}
            pointerEvents="box-none"
          >
            <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Edit entry</Text>
            <Text style={styles.modalLabel}>Date on timeline (YYYY-MM-DD)</Text>
            <TextInput
              value={draftDate}
              onChangeText={setDraftDate}
              placeholder="2026-04-18"
              placeholderTextColor={palette.inkFaint}
              style={styles.input}
              autoCapitalize="none"
              autoCorrect={false}
            />
            <Text style={[styles.modalLabel, { marginTop: 12 }]}>Possible causes / what changed</Text>
            <TextInput
              value={draftNote}
              onChangeText={setDraftNote}
              placeholder="e.g. new cleanser, late night, stress, dairy…"
              placeholderTextColor={palette.inkFaint}
              style={[styles.input, styles.inputMultiline]}
              multiline
              textAlignVertical="top"
            />
            <View style={styles.modalActions}>
              <Pressable onPress={closeEdit} style={({ pressed }) => [styles.modalBtnGhost, pressed && { opacity: 0.8 }]}>
                <Text style={styles.modalBtnGhostText}>Cancel</Text>
              </Pressable>
              <Pressable onPress={saveEdit} style={({ pressed }) => [styles.modalBtnPrimary, pressed && { opacity: 0.9 }]}>
                <Text style={styles.modalBtnPrimaryText}>Save</Text>
              </Pressable>
            </View>
            </View>
          </KeyboardAvoidingView>
        </View>
      </Modal>
    </View>
  );
}

function MiniBars({ entries, highlightId }: { entries: TimelineEntry[]; highlightId: string }) {
  const points = entries.slice(0, 10).reverse();
  const max = Math.max(1, ...points.map((p) => p.severityScore));
  return (
    <View style={styles.bars}>
      {points.map((p) => {
        const h = 6 + Math.round((p.severityScore / max) * 22);
        const isHi = p.id === highlightId;
        return <View key={p.id} style={[styles.bar, { height: h }, isHi && styles.barHi]} />;
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: palette.bg },
  header: { padding: 16, paddingBottom: 10 },
  headerTop: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  h1: { color: palette.ink, fontSize: 26, fontWeight: '900', flexShrink: 1 },
  clearBtn: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.destructiveBorder,
    backgroundColor: palette.destructiveBg,
  },
  clearBtnPressed: { opacity: 0.75 },
  clearBtnText: { color: palette.destructiveText, fontSize: 13, fontWeight: '800' },
  sub: { color: palette.inkSubtle, marginTop: 6, fontSize: 13 },
  subStrong: { color: palette.sageDark, fontWeight: '900' },
  hint: { color: palette.inkFaint, marginTop: 8, fontSize: 12, lineHeight: 17 },
  list: { paddingHorizontal: 16, paddingBottom: 18, gap: 12 },
  row: {
    flexDirection: 'row',
    gap: 12,
    padding: 12,
    borderRadius: 16,
    backgroundColor: palette.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.border,
  },
  rowPressed: { backgroundColor: palette.sageFill8 },
  thumbWrap: { width: 72, height: 72, borderRadius: 14, overflow: 'hidden', backgroundColor: palette.surfaceMuted },
  thumb: { width: 72, height: 72 },
  titleRow: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: 8 },
  rowTitle: { color: palette.ink, fontSize: 13, fontWeight: '900' },
  flarePill: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 8,
    backgroundColor: palette.flareBg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.flareBorder,
  },
  flarePillText: { color: palette.flareText, fontSize: 10, fontWeight: '900', letterSpacing: 0.5 },
  rowBody: { color: palette.inkMuted, fontSize: 12, marginTop: 4 },
  notePreview: { color: palette.sageDark, fontSize: 12, marginTop: 6, fontStyle: 'italic' },
  bars: { flexDirection: 'row', gap: 4, marginTop: 8, alignItems: 'flex-end' },
  bar: { width: 10, borderRadius: 5, backgroundColor: palette.sageFill18 },
  barHi: { backgroundColor: palette.sage },
  empty: { padding: 24, margin: 16, borderRadius: 16, backgroundColor: palette.surface },
  emptyTitle: { color: palette.ink, fontSize: 16, fontWeight: '900', marginBottom: 6 },
  emptyBody: { color: palette.inkMuted, fontSize: 13 },
  modalRoot: {
    flex: 1,
    backgroundColor: palette.scrim,
  },
  modalKav: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    padding: 20,
  },
  modalCard: {
    borderRadius: 18,
    padding: 18,
    backgroundColor: palette.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.border,
  },
  modalTitle: { color: palette.ink, fontSize: 18, fontWeight: '900', marginBottom: 14 },
  modalLabel: { color: palette.inkSubtle, fontSize: 12, fontWeight: '700', marginBottom: 6 },
  input: {
    borderRadius: 12,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.border,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: palette.ink,
    fontSize: 15,
    backgroundColor: palette.surfaceMuted,
  },
  inputMultiline: { minHeight: 100, paddingTop: 10 },
  modalActions: { flexDirection: 'row', justifyContent: 'flex-end', gap: 10, marginTop: 18 },
  modalBtnGhost: {
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: palette.border,
    backgroundColor: palette.surface,
  },
  modalBtnGhostText: { color: palette.sageDark, fontSize: 14, fontWeight: '800' },
  modalBtnPrimary: {
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 12,
    backgroundColor: palette.sage,
  },
  modalBtnPrimaryText: { color: palette.onPrimary, fontSize: 14, fontWeight: '900' },
});
