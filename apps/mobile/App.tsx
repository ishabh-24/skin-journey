import { DefaultTheme, NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { initDb } from './src/lib/db';
import { palette } from './src/theme/colors';
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
        headerStyle: { backgroundColor: palette.headerBg },
        headerTintColor: palette.headerTint,
        contentStyle: { backgroundColor: palette.bg },
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
        <ActivityIndicator color={palette.sage} />
        <StatusBar style="dark" />
      </View>
    );
  }

  return (
    <SafeAreaProvider>
      <NavigationContainer theme={SJTheme}>
        <StatusBar style="dark" />
        <Tab.Navigator
          screenOptions={{
            headerStyle: { backgroundColor: palette.headerBg },
            headerTintColor: palette.headerTint,
            tabBarStyle: {
              backgroundColor: palette.tabBarBg,
              borderTopColor: palette.tabBorder,
            },
            tabBarActiveTintColor: palette.tabActive,
            tabBarInactiveTintColor: palette.tabInactive,
          }}
        >
          <Tab.Screen name="CaptureTab" component={CaptureStackNavigator} options={{ title: 'Capture', headerShown: false }} />
          <Tab.Screen name="Timeline" component={TimelineScreen} options={{ title: 'Timeline' }} />
          <Tab.Screen name="Settings" component={SettingsScreen} options={{ title: 'Settings' }} />
        </Tab.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

const SJTheme = {
  ...DefaultTheme,
  dark: false,
  colors: {
    ...DefaultTheme.colors,
    primary: palette.sage,
    background: palette.bg,
    card: palette.surface,
    text: palette.ink,
    border: palette.border,
    notification: palette.sage,
  },
};

const styles = StyleSheet.create({
  splash: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: palette.bg, gap: 14 },
  splashTitle: { color: palette.sageDark, fontSize: 22, fontWeight: '900' },
});
