import { NavigationContainer, DarkTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

import { initDb } from './src/lib/db';
import { AnalysisScreen } from './src/screens/AnalysisScreen';
import { CaptureScreen } from './src/screens/CaptureScreen';
import { SettingsScreen } from './src/screens/SettingsScreen';
import { TimelineScreen } from './src/screens/TimelineScreen';
import type { CaptureStackParamList, RootTabParamList } from './src/types/nav';

const Tab = createBottomTabNavigator<RootTabParamList>();
const CaptureStack = createNativeStackNavigator<CaptureStackParamList>();

function CaptureStackNavigator() {
  return (
    <CaptureStack.Navigator
      screenOptions={{
        headerStyle: { backgroundColor: '#0B0B10' },
        headerTintColor: 'white',
        contentStyle: { backgroundColor: '#0B0B10' },
      }}
    >
      <CaptureStack.Screen name="Capture" component={CaptureScreen} options={{ title: 'Capture' }} />
      <CaptureStack.Screen name="Analysis" component={AnalysisScreen} options={{ title: 'Analysis' }} />
    </CaptureStack.Navigator>
  );
}

export default function App() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    (async () => {
      await initDb();
      setReady(true);
    })();
  }, []);

  if (!ready) {
    return (
      <View style={styles.splash}>
        <Text style={styles.splashTitle}>Skin Journey AI</Text>
        <ActivityIndicator color="white" />
        <StatusBar style="light" />
      </View>
    );
  }

  return (
    <NavigationContainer theme={SJTheme}>
      <StatusBar style="light" />
      <Tab.Navigator
        screenOptions={{
          headerStyle: { backgroundColor: '#0B0B10' },
          headerTintColor: 'white',
          tabBarStyle: { backgroundColor: '#0B0B10', borderTopColor: 'rgba(255,255,255,0.14)' },
          tabBarActiveTintColor: 'white',
          tabBarInactiveTintColor: 'rgba(255,255,255,0.6)',
        }}
      >
        <Tab.Screen name="CaptureTab" component={CaptureStackNavigator} options={{ title: 'Capture', headerShown: false }} />
        <Tab.Screen name="Timeline" component={TimelineScreen} options={{ title: 'Timeline' }} />
        <Tab.Screen name="Settings" component={SettingsScreen} options={{ title: 'Settings' }} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

const SJTheme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    background: '#0B0B10',
    card: '#0B0B10',
    border: 'rgba(255,255,255,0.14)',
    text: 'white',
    primary: '#6C5CE7',
  },
};

const styles = StyleSheet.create({
  splash: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#0B0B10', gap: 14 },
  splashTitle: { color: 'white', fontSize: 22, fontWeight: '900' },
});
