import React, { useEffect, useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';

import { getSetting, setSetting } from '../lib/db';

const DEFAULT_API_BASE_URL = 'http://localhost:8000';

export function SettingsScreen() {
  const [apiBaseUrl, setApiBaseUrlState] = useState(DEFAULT_API_BASE_URL);
  const [saved, setSaved] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const v = await getSetting('apiBaseUrl');
      if (v) {
        setApiBaseUrlState(v);
        setSaved(v);
      } else {
        setSaved(DEFAULT_API_BASE_URL);
      }
    })();
  }, []);

  async function onSave() {
    const v = apiBaseUrl.trim();
    if (!/^https?:\/\//.test(v)) {
      Alert.alert('Invalid URL', 'API Base URL must start with http:// or https://');
      return;
    }
    await setSetting('apiBaseUrl', v);
    setSaved(v);
    Alert.alert('Saved', 'API Base URL updated.');
  }

  return (
    <View style={styles.root}>
      <Text style={styles.h1}>Settings</Text>

      <View style={styles.card}>
        <Text style={styles.label}>API Base URL</Text>
        <Text style={styles.help}>
          iOS simulator: <Text style={styles.mono}>http://localhost:8000</Text> · Device: use your LAN IP.
        </Text>
        <TextInput
          value={apiBaseUrl}
          onChangeText={setApiBaseUrlState}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          placeholder="http://localhost:8000"
          placeholderTextColor="rgba(255,255,255,0.35)"
          style={styles.input}
        />
        <Pressable style={styles.primaryBtn} onPress={onSave}>
          <Text style={styles.primaryBtnText}>Save</Text>
        </Pressable>
        {saved ? <Text style={styles.saved}>Current: {saved}</Text> : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0B0B10', padding: 16 },
  h1: { color: 'white', fontSize: 26, fontWeight: '900' },
  card: {
    marginTop: 14,
    padding: 14,
    borderRadius: 16,
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.14)',
  },
  label: { color: 'white', fontSize: 14, fontWeight: '900' },
  help: { color: 'rgba(255,255,255,0.75)', fontSize: 12, marginTop: 6, lineHeight: 16 },
  mono: { fontFamily: 'Courier', color: 'white' },
  input: {
    marginTop: 10,
    height: 46,
    borderRadius: 12,
    paddingHorizontal: 12,
    backgroundColor: 'rgba(0,0,0,0.25)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.16)',
    color: 'white',
  },
  primaryBtn: {
    marginTop: 12,
    height: 48,
    borderRadius: 14,
    backgroundColor: '#6C5CE7',
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryBtnText: { color: 'white', fontSize: 14, fontWeight: '900' },
  saved: { marginTop: 10, color: 'rgba(255,255,255,0.75)', fontSize: 12 },
});

