// src/theme.js
import { createTheme, alpha } from '@mui/material/styles';
import { blue, pink, green, grey, amber, lightBlue, lightGreen } from '@mui/material/colors'; // Added more colors

// Function to get design tokens based on mode and optional userRole
export const getDesignTokens = (mode, userRole = 'NORMAL') => { // Default role if not provided
    const isMeteorologist = userRole === 'METEOROLOGIST';

    // Define focus colors based on role and mode
    let lightFocusColor, darkFocusColor, darkFocusColor2, dividerColor, primaryColor;
    if (isMeteorologist) {
        primaryColor = green; // Use green for meteorologists
        lightFocusColor = green[600];
        darkFocusColor = lightGreen[400];
        darkFocusColor2 = green[300]; // Lighter green for dark mode primary
        dividerColor = green[400];
    } else {
        primaryColor = blue; // Default to blue for other roles
        lightFocusColor = blue[600];
        darkFocusColor = lightBlue[400];
        darkFocusColor2 = blue[300]; // Lighter blue for dark mode primary
        dividerColor = blue[400];
    }

    return {
        palette: {
            mode,
            ...(mode === 'light'
                ? {
                    // Palette values for light mode
                    primary: primaryColor,
                    secondary: pink,
                    divider: amber[200],
                    focusRing: lightFocusColor, // Custom focus ring color for light mode
                    text: {
                        primary: grey[900],
                        secondary: grey[700], // Slightly darker for better contrast on light bg
                    },
                    background: {
                        default: grey[50], // Very light grey background
                        paper: '#ffffff',
                    },
                    action: {
                        active: lightFocusColor, // For active elements like selected NavLink
                        hover: alpha(lightFocusColor, 0.08),
                        selected: alpha(lightFocusColor, 0.16),
                    }
                }
                : {
                    // Palette values for dark mode
                    primary: { main: darkFocusColor }, // Lighter blue for dark mode primary
                    secondary: { main: pink[300] },   // Lighter pink
                    divider: dividerColor,
                    focusRing: darkFocusColor, // Custom focus ring color for dark mode
                    background: {
                        default: '#121212',    // Common dark mode background
                        paper: grey[900],      // Slightly lighter paper for cards, elevation
                    },
                    text: {
                        primary: '#ffffff',
                        secondary: grey[400], // Lighter grey for secondary text on dark
                    },
                    action: {
                        active: darkFocusColor2,
                        hover: alpha(darkFocusColor2, 0.1),
                        selected: alpha(darkFocusColor2, 0.2),
                    }
                }),
        },
        typography: {
            fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
            h4: { fontWeight: 500 },
            h5: { fontWeight: 500 },
            h6: { fontWeight: 500 },
        },
        components: {
            MuiAppBar: {
                styleOverrides: {
                    root: ({ theme }) => ({
                        // Keep app bar distinct in dark mode, or match paper
                        backgroundColor: mode === 'light' ? theme.palette.primary.main : theme.palette.grey[800], // Example
                        color: mode === 'light' ? theme.palette.primary.contrastText : theme.palette.text.primary,
                    }),
                },
            },
            MuiDrawer: {
                styleOverrides: {
                    paper: ({ theme }) => ({
                        backgroundColor: mode === 'light' ? theme.palette.background.paper : theme.palette.grey[800], // Consistent with AppBar dark
                    }),
                },
            },
            // Global focus styles
            MuiButtonBase: { // Targets Buttons, IconButtons, etc.
                defaultProps: {
                    disableRipple: false, // Keep ripple unless specifically unwanted
                },
                styleOverrides: {
                    root: ({ theme }) => ({
                        '&.Mui-focusVisible': {
                            outline: `2px solid ${theme.palette.focusRing}`,
                            outlineOffset: '2px',
                            // boxShadow: `0 0 0 2px ${alpha(theme.palette.focusRing, 0.5)}`, // Alternative focus style
                        },
                    }),
                },
            },
            MuiListItemButton: {
                styleOverrides: {
                    root: ({ theme }) => ({
                        '&.Mui-focusVisible': {
                            outline: `2px solid ${theme.palette.focusRing}`,
                            outlineOffset: '1px',
                            backgroundColor: theme.palette.action.hover, // Add a subtle bg on focus
                        },
                        '&.active': { // For NavLink active state
                            backgroundColor: theme.palette.action.selected,
                            // color: theme.palette.primary.main, // If you want text color to change
                            '& .MuiListItemIcon-root': {
                                // color: theme.palette.primary.main, // If you want icon color to change
                            }
                        }
                    }),
                },
            },
            MuiOutlinedInput: { // For TextField, Select outlined variant
                styleOverrides: {
                    root: ({ theme }) => ({
                        '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                            borderColor: theme.palette.focusRing,
                            borderWidth: '1px', // Ensure border width doesn't jump too much
                        },
                    }),
                },
            },
            MuiChip: { // Make chips non-interactive by default if that's the desired behavior
                defaultProps: {
                    // clickable: false, // This would make it non-interactive
                },
                styleOverrides: {
                    root: ({ theme }) => ({
                        // If you want a focus style for chips that ARE clickable:
                        // '&.Mui-focusVisible': {
                        //   outline: `2px solid ${theme.palette.focusRing}`,
                        //   backgroundColor: alpha(theme.palette.focusRing, 0.2),
                        // },
                    }),
                },
            },
            MuiPaper: {
                defaultProps: {
                    elevation: mode === 'light' ? 2 : 4, // Slightly more pronounced elevation in dark mode
                }
            },
            MuiCard: {
                defaultProps: {
                    elevation: mode === 'light' ? 2 : 4,
                }
            }
        },
    };
};

// This function creates the theme instance
export const createCustomTheme = (mode, userRole) => createTheme(getDesignTokens(mode, userRole));