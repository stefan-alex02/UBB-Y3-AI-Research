import React, { createContext, useState, useContext, useMemo, useEffect } from 'react';

const ThemeContext = createContext({
    mode: 'light',
    toggleThemeMode: () => {},
    setThemeMode: () => {},
});

export const ThemeProvider = ({ children }) => {
    const prefersDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const storedTheme = localStorage.getItem('themeMode');
    const initialMode = storedTheme || (prefersDarkMode ? 'dark' : 'light');

    const [mode, setMode] = useState(initialMode);

    useEffect(() => {
        localStorage.setItem('themeMode', mode);
    }, [mode]);

    const toggleThemeMode = () => {
        setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
    };

    const setTheme = (newMode) => {
        if (newMode === 'system') {
            setMode(prefersDarkMode ? 'dark' : 'light');
        } else if (newMode === 'light' || newMode === 'dark') {
            setMode(newMode);
        }
    }

    const contextValue = useMemo(() => ({
        mode,
        toggleThemeMode,
        setThemeMode: setTheme,
    }), [mode, prefersDarkMode]); // Add prefersDarkMode to dependencies

    return (
        <ThemeContext.Provider value={contextValue}>
            {children}
        </ThemeContext.Provider>
    );
};

export const useThemeMode = () => useContext(ThemeContext);