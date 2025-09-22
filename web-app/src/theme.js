import {alpha, createTheme} from '@mui/material/styles';
import {amber, blue, green, grey, lightBlue, lightGreen} from '@mui/material/colors';

export const getDesignTokens = (mode, userRole = 'NORMAL') => {
    const isMeteorologist = userRole === 'METEOROLOGIST';

    let lightFocusColor, darkFocusColor, darkFocusColor2, dividerColor, primaryColor, secondaryColor;
    if (isMeteorologist) {
        primaryColor = green;
        secondaryColor = amber;
        lightFocusColor = green[600];
        darkFocusColor = lightGreen[400];
        darkFocusColor2 = green[300];
        dividerColor = green[400];
    } else {
        primaryColor = blue;
        secondaryColor = amber;
        lightFocusColor = blue[600];
        darkFocusColor = lightBlue[400];
        darkFocusColor2 = blue[300];
        dividerColor = blue[400];
    }

    return {
        palette: {
            mode,
            ...(mode === 'light'
                ? {
                    // Palette values for light mode
                    primary: primaryColor,
                    secondary: secondaryColor,
                    divider: secondaryColor[200],
                    focusRing: lightFocusColor,
                    text: {
                        primary: grey[900],
                        secondary: grey[700],
                    },
                    background: {
                        default: grey[50],
                        paper: '#ffffff',
                    },
                    action: {
                        active: lightFocusColor,
                        hover: alpha(lightFocusColor, 0.08),
                        selected: alpha(lightFocusColor, 0.16),
                    }
                }
                : {
                    // Palette values for dark mode
                    primary: { main: darkFocusColor },
                    secondary: { main: secondaryColor[300] },
                    divider: dividerColor,
                    focusRing: darkFocusColor,
                    background: {
                        default: '#121212',
                        paper: grey[900],
                    },
                    text: {
                        primary: '#ffffff',
                        secondary: grey[400],
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
                        backgroundColor: mode === 'light' ? theme.palette.primary.main : theme.palette.grey[800],
                        color: mode === 'light' ? theme.palette.primary.contrastText : theme.palette.text.primary,
                    }),
                },
            },
            MuiDrawer: {
                styleOverrides: {
                    paper: ({ theme }) => ({
                        backgroundColor: mode === 'light' ? theme.palette.background.paper : theme.palette.grey[800],
                    }),
                },
            },
            // Global focus styles
            MuiButtonBase: {
                defaultProps: {
                    disableRipple: false,
                },
                styleOverrides: {
                    root: ({ theme }) => ({
                        '&.Mui-focusVisible': {
                            outline: `2px solid ${theme.palette.focusRing}`,
                            outlineOffset: '2px',
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
                            backgroundColor: theme.palette.action.hover,
                        },
                        '&.active': {
                            backgroundColor: theme.palette.action.selected,
                            // color: theme.palette.primary.main,
                            '& .MuiListItemIcon-root': {
                                // color: theme.palette.primary.main,
                            }
                        }
                    }),
                },
            },
            MuiOutlinedInput: {
                styleOverrides: {
                    root: ({ theme }) => ({
                        '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                            borderColor: theme.palette.focusRing,
                            borderWidth: '1px',
                        },
                    }),
                },
            },
            MuiChip: {
                defaultProps: {
                    // clickable: false,
                },
                styleOverrides: {
                    root: ({ theme }) => ({
                        // '&.Mui-focusVisible': {
                        //   outline: `2px solid ${theme.palette.focusRing}`,
                        //   backgroundColor: alpha(theme.palette.focusRing, 0.2),
                        // },
                    }),
                },
            },
            MuiPaper: {
                defaultProps: {
                    elevation: mode === 'light' ? 2 : 4,
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

export const createCustomTheme = (mode, userRole) => createTheme(getDesignTokens(mode, userRole));