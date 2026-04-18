import { CameraView, useCameraPermissions } from 'expo-camera';
import React, { useMemo, useRef, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { FaceGridOverlay } from '../components/FaceGridOverlay';
import type { CaptureStackParamList } from '../types/nav';

type Props = NativeStackScreenProps<CaptureStackParamList, 'Capture'>;

export function CaptureScreen({ navigation }: Props) {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [busy, setBusy] = useState(false);

  const canUseCamera = !!permission?.granted;

  const tips = useMemo(
    () => [
      'Same lighting each day (avoid harsh shadows)',
      'Hold phone at eye level',
      'Center your face inside the guide',
    ],
    [],
  );

  async function onCapture() {
    if (!cameraRef.current || busy) return;
    try {
      setBusy(true);
      const pic = await cameraRef.current.takePictureAsync({
        quality: 0.7,
        exif: false,
        skipProcessing: true,
      });
      if (!pic?.uri) return;
      navigation.navigate('Analysis', { imageUri: pic.uri });
    } finally {
      setBusy(false);
    }
  }

  if (!permission) {
    return (
      <View style={styles.center}>
        <Text style={styles.title}>Camera</Text>
        <Text style={styles.body}>Loading permissions…</Text>
      </View>
    );
  }

  if (!canUseCamera) {
    return (
      <View style={styles.center}>
        <Text style={styles.title}>Enable camera</Text>
        <Text style={styles.body}>We need camera access to capture your daily photo.</Text>
        <Pressable style={styles.primaryBtn} onPress={requestPermission}>
          <Text style={styles.primaryBtnText}>Grant permission</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.cameraWrap}>
        <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="front" />
        <FaceGridOverlay />

        <View style={styles.tipBox}>
          <Text style={styles.tipTitle}>Guided capture</Text>
          {tips.map((t) => (
            <Text key={t} style={styles.tipText}>
              • {t}
            </Text>
          ))}
        </View>
      </View>

      <View style={styles.bottomBar}>
        <Pressable style={[styles.shutter, busy && styles.shutterDisabled]} onPress={onCapture}>
          <Text style={styles.shutterText}>{busy ? '…' : 'Capture'}</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0B0B10' },
  cameraWrap: { flex: 1, overflow: 'hidden' },
  bottomBar: {
    padding: 16,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: 'rgba(255,255,255,0.12)',
    backgroundColor: '#0B0B10',
  },
  tipBox: {
    position: 'absolute',
    left: 14,
    right: 14,
    bottom: 14,
    padding: 12,
    borderRadius: 14,
    backgroundColor: 'rgba(0,0,0,0.35)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: 'rgba(255,255,255,0.18)',
  },
  tipTitle: { color: 'white', fontSize: 14, fontWeight: '700', marginBottom: 4 },
  tipText: { color: 'rgba(255,255,255,0.9)', fontSize: 12, marginTop: 2 },
  shutter: {
    height: 52,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#6C5CE7',
  },
  shutterDisabled: { opacity: 0.6 },
  shutterText: { color: 'white', fontSize: 16, fontWeight: '800' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24, backgroundColor: '#0B0B10' },
  title: { color: 'white', fontSize: 22, fontWeight: '800', marginBottom: 8 },
  body: { color: 'rgba(255,255,255,0.85)', fontSize: 14, textAlign: 'center', marginBottom: 14 },
  primaryBtn: { backgroundColor: '#6C5CE7', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 12 },
  primaryBtnText: { color: 'white', fontWeight: '800' },
});

