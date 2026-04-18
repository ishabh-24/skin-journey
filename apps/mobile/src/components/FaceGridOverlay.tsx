import React from 'react';
import { StyleSheet, View } from 'react-native';

export function FaceGridOverlay() {
  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      <View style={styles.border} />
      <View style={styles.v1} />
      <View style={styles.v2} />
      <View style={styles.h1} />
      <View style={styles.h2} />
      <View style={styles.centerDot} />
    </View>
  );
}

const styles = StyleSheet.create({
  border: {
    ...StyleSheet.absoluteFillObject,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.45)',
    borderRadius: 18,
    margin: 18,
  },
  v1: {
    position: 'absolute',
    top: 18,
    bottom: 18,
    left: '33%',
    width: 1,
    backgroundColor: 'rgba(255,255,255,0.25)',
  },
  v2: {
    position: 'absolute',
    top: 18,
    bottom: 18,
    left: '66%',
    width: 1,
    backgroundColor: 'rgba(255,255,255,0.25)',
  },
  h1: {
    position: 'absolute',
    left: 18,
    right: 18,
    top: '33%',
    height: 1,
    backgroundColor: 'rgba(255,255,255,0.25)',
  },
  h2: {
    position: 'absolute',
    left: 18,
    right: 18,
    top: '66%',
    height: 1,
    backgroundColor: 'rgba(255,255,255,0.25)',
  },
  centerDot: {
    position: 'absolute',
    left: '50%',
    top: '50%',
    width: 6,
    height: 6,
    marginLeft: -3,
    marginTop: -3,
    borderRadius: 3,
    backgroundColor: 'rgba(255,255,255,0.7)',
  },
});

