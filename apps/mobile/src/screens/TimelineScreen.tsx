import { useFocusEffect } from '@react-navigation/native';
import React, { useCallback, useMemo, useState } from 'react';
import { FlatList, Image, RefreshControl, StyleSheet, Text, View } from 'react-native';

import { listEntries } from '../lib/db';
import { computeTrendLabel, primaryRegion } from '../lib/trends';
import type { TimelineEntry } from '../types/models';

function fmtDate(ms: number) {
  const d = new Date(ms);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function TimelineScreen() {
  const [entries, setEntries] = useState<TimelineEntry[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  const trend = useMemo(() => computeTrendLabel(entries.slice(0, 7)), [entries]);

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

  return (
    <View style={styles.root}>
      <View style={styles.header}>
        <Text style={styles.h1}>Timeline</Text>
        <Text style={styles.sub}>
          Trend: <Text style={styles.subStrong}>{trend}</Text>
        </Text>
      </View>

      <FlatList
        data={entries}
        keyExtractor={(e) => e.id}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="white" />}
        contentContainerStyle={styles.list}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyTitle}>No entries yet</Text>
            <Text style={styles.emptyBody}>Capture a daily photo to start your skin timeline.</Text>
          </View>
        }
        renderItem={({ item }) => (
          <View style={styles.row}>
            <View style={styles.thumbWrap}>
              <Image source={{ uri: item.imageUri }} style={styles.thumb} />
              <Image source={{ uri: item.heatmapUri }} style={styles.thumbHeat} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.rowTitle}>
                {fmtDate(item.createdAt)} · {item.severityBucket.toUpperCase()}
              </Text>
              <Text style={styles.rowBody}>
                Severity {item.severityScore.toFixed(1)} · Primary zone {primaryRegion(item.regionScores).replace('_', ' ')}
              </Text>
              <MiniBars entries={entries} highlightId={item.id} />
            </View>
          </View>
        )}
      />
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
  root: { flex: 1, backgroundColor: '#0B0B10' },
  header: { padding: 16, paddingBottom: 10 },
  h1: { color: 'white', fontSize: 26, fontWeight: '900' },
  sub: { color: 'rgba(255,255,255,0.78)', marginTop: 6, fontSize: 13 },
  subStrong: { color: 'white', fontWeight: '900' },
  list: { paddingHorizontal: 16, paddingBottom: 18, gap: 12 },
  row: {
    flexDirection: 'row',
    gap: 12,
    padding: 12,
    borderRadius: 16,
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.14)',
  },
  thumbWrap: { width: 72, height: 72, borderRadius: 14, overflow: 'hidden', backgroundColor: 'rgba(255,255,255,0.08)' },
  thumb: { width: 72, height: 72 },
  thumbHeat: { position: 'absolute', left: 0, top: 0, right: 0, bottom: 0, opacity: 0.7 },
  rowTitle: { color: 'white', fontSize: 13, fontWeight: '900' },
  rowBody: { color: 'rgba(255,255,255,0.78)', fontSize: 12, marginTop: 4 },
  bars: { flexDirection: 'row', gap: 4, marginTop: 8, alignItems: 'flex-end' },
  bar: { width: 10, borderRadius: 5, backgroundColor: 'rgba(255,255,255,0.25)' },
  barHi: { backgroundColor: '#6C5CE7' },
  empty: { padding: 24, margin: 16, borderRadius: 16, backgroundColor: 'rgba(255,255,255,0.06)' },
  emptyTitle: { color: 'white', fontSize: 16, fontWeight: '900', marginBottom: 6 },
  emptyBody: { color: 'rgba(255,255,255,0.78)', fontSize: 13 },
});

