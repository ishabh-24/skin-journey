import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';

import { getSetting, setSetting } from '../lib/db';

const DEFAULT_API_BASE_URL = 'http://localhost:8000';

export function SettingsScreen() {
  const [apiBaseUrl, setApiBaseUrlState] = useState(DEFAULT_API_BASE_URL);
  const [saved, setSaved] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);

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

  async function onTestConnection() {
    const base = apiBaseUrl.trim().replace(/\/+$/, '');
    if (!/^https?:\/\//.test(base)) {
      Alert.alert('Invalid URL', 'Must start with http:// or https://');
      return;
    }
    setTesting(true);
    const url = `${base}/health`;
    try {
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), 10000);
      const res = await fetch(url, { method: 'GET', signal: ac.signal });
      clearTimeout(timer);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const j = (await res.json().catch(() => null)) as { ok?: boolean } | null;
      Alert.alert('Connected', j && typeof j === 'object' && 'ok' in j ? 'API health check succeeded.' : `OK (${res.status})`);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      Alert.alert(
        'Connection failed',
        `${msg}\n\n` +
          'On a real phone, use http://YOUR_LAPTOP_IP:8000 (not localhost). On Mac, run: ipconfig getifaddr en0\n' +
          'Start API with: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 from the services/api folder.\n' +
          'If it still fails: turn off VPN, allow Python in the firewall, and avoid guest Wi‑Fi (AP isolation).',
      );
    } finally {
      setTesting(false);
    }
  }

  return (
    <View style={styles.root}>
      <Text style={styles.h1}>Settings</Text>

      <View style={styles.card}>
        <Text style={styles.label}>API Base URL</Text>
        <Text style={styles.help}>
          Simulator: <Text style={styles.mono}>http://localhost:8000</Text>
          {'\n'}
          Physical phone: <Text style={styles.mono}>http://192.168.x.x:8000</Text> (your Mac’s Wi‑Fi IP — no trailing slash).
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
        <Pressable
          style={[styles.secondaryBtn, testing && styles.secondaryBtnDisabled]}
          onPress={onTestConnection}
          disabled={testing}
        >
          {testing ? <ActivityIndicator color="white" /> : <Text style={styles.secondaryBtnText}>Test connection</Text>}
        </Pressable>
        {saved ? <Text style={styles.saved}>Saved URL: {saved}</Text> : null}
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
  secondaryBtn: {
    marginTop: 10,
    height: 46,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.22)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryBtnDisabled: { opacity: 0.6 },
  secondaryBtnText: { color: 'white', fontSize: 14, fontWeight: '800' },
  saved: { marginTop: 10, color: 'rgba(255,255,255,0.75)', fontSize: 12 },
});

