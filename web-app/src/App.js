import React from 'react';
import {ThemeProvider as MuiThemeProvider} from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import {AuthProvider} from './contexts/AuthContext';
import {ThemeProvider, useThemeMode} from './contexts/ThemeContext';
import AppRoutes from './routes/AppRoutes';
import {createCustomTheme} from './theme';
import useAuth from "./hooks/useAuth";

function ThemedApp() {
  const { mode } = useThemeMode();
    const { user } = useAuth();
    const theme = React.useMemo(() => createCustomTheme(mode, user?.role), [mode, user?.role]);


  return (
      <MuiThemeProvider theme={theme}>
        <CssBaseline />
        <AppRoutes />
      </MuiThemeProvider>
  );
}

function App() {
  return (
      <AuthProvider>
        <ThemeProvider>
          <ThemedApp />
        </ThemeProvider>
      </AuthProvider>
  );
}

export default App;