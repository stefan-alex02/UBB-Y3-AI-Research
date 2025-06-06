import { createTheme } from '@mui/material/styles';
import { amber, deepOrange, grey, blue, pink } from '@mui/material/colors';

export const getDesignTokens = (mode) => ({
    palette: {
        mode,
        ...(mode === 'light'
            ? {
                // palette values for light mode
                primary: blue, // Example: blue for light mode
                secondary: pink,
                divider: amber[200],
                text: {
                    primary: grey[900],
                    secondary: grey[800],
                },
                background: {
                    default: grey[100], // Light grey background
                    paper: '#ffffff',
                }
            }
            : {
                // palette values for dark mode
                primary: { main: '#90caf9' }, // Lighter blue for dark mode primary
                secondary: { main: '#f48fb1' }, // Lighter pink for dark mode secondary
                divider: deepOrange[700],
                background: {
                    default: grey[900], // Dark background
                    paper: grey[800],   // Slightly lighter paper for cards, etc.
                },
                text: {
                    primary: '#fff',
                    secondary: grey[500],
                },
            }),
    },
    typography: {
        fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
        // You can customize typography further here
    },
    components: {
        // Example: customize AppBar
        MuiAppBar: {
            styleOverrides: {
                root: ({ theme }) => ({
                    backgroundColor: mode === 'light' ? theme.palette.primary.main : theme.palette.background.paper,
                    color: mode === 'light' ? theme.palette.primary.contrastText : theme.palette.text.primary,
                }),
            },
        },
        MuiDrawer: {
            styleOverrides: {
                paper: ({theme}) => ({
                    backgroundColor: mode === 'light' ? theme.palette.background.paper : grey[800],
                }),
            },
        }
        // You can override styles for other components globally
    }
});

export const createCustomTheme = (mode) => createTheme(getDesignTokens(mode));