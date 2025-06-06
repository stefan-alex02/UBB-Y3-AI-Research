import React from 'react';
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider, useThemeMode } from './contexts/ThemeContext';
import AppRoutes from './routes/AppRoutes';
import { createCustomTheme } from './theme'; // We'll define this

function ThemedApp() {
  const { mode } = useThemeMode();
  const theme = React.useMemo(() => createCustomTheme(mode), [mode]);

  return (
      <MuiThemeProvider theme={theme}>
        <CssBaseline /> {/* Normalizes styles and applies background based on theme */}
        <AppRoutes />
      </MuiThemeProvider>
  );
}

function App() {
  return (
      <AuthProvider>
        <ThemeProvider> {/* Our custom theme provider */}
          <ThemedApp />
        </ThemeProvider>
      </AuthProvider>
  );
}

export default App;