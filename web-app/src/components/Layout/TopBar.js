import React from 'react';
import {
    AppBar,
    Avatar,
    Box,
    Button,
    IconButton,
    Menu,
    MenuItem,
    Select,
    Toolbar,
    Tooltip,
    Typography,
    useTheme
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import Brightness4Icon from '@mui/icons-material/Brightness4'; // Dark mode
import Brightness7Icon from '@mui/icons-material/Brightness7'; // Light mode
import SettingsIcon from '@mui/icons-material/Settings'; // System mode (example)
import useAuth from '../../hooks/useAuth';
import {useThemeMode} from '../../contexts/ThemeContext';
import {useNavigate} from 'react-router-dom';

const TopBar = ({
                    onMobileMenuClick,
                    isMobile,
                    // Props for AppBar styling based on permanent drawer state (only for non-mobile)
                    isPermanentDrawerOpen,
                    permanentDrawerWidth,
                    miniDrawerWidth
                }) => {
    const theme = useTheme();
    const { user, logout } = useAuth();
    const { mode, setThemeMode } = useThemeMode();
    const navigate = useNavigate();
    const [anchorElUser, setAnchorElUser] = React.useState(null);

    const handleOpenUserMenu = (event) => setAnchorElUser(event.currentTarget);
    const handleCloseUserMenu = () => setAnchorElUser(null);
    const handleLogout = () => { logout(); handleCloseUserMenu(); navigate('/login'); };
    const handleSettings = () => { navigate('/settings'); handleCloseUserMenu(); };

    const appBarMarginLeft = !isMobile ? (isPermanentDrawerOpen ? permanentDrawerWidth : miniDrawerWidth) : 0;
    const appBarWidth = !isMobile ? `calc(100% - ${appBarMarginLeft}px)` : '100%';

    return (
        <AppBar
            position="fixed"
            sx={{
                zIndex: (theme) => theme.zIndex.drawer + 1,
                width: appBarWidth,
                marginLeft: `${appBarMarginLeft}px`,
                transition: theme.transitions.create(['width', 'margin'], {
                    easing: theme.transitions.easing.sharp,
                    duration: isPermanentDrawerOpen // Use current state for duration hint
                        ? theme.transitions.duration.enteringScreen
                        : theme.transitions.duration.leavingScreen,
                }),
            }}
        >
            <Toolbar>
                {/* Hamburger icon ONLY for mobile to toggle the temporary mobile drawer */}
                {isMobile && (
                    <IconButton
                        color="inherit"
                        aria-label="open drawer"
                        edge="start"
                        onClick={onMobileMenuClick}
                        sx={{ mr: 2 }} // No display: {sm: 'none'} needed if parent check is isMobile
                    >
                        <MenuIcon />
                    </IconButton>
                )}
                <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
                    Cloud Classifier
                </Typography>

                <Select
                    value={localStorage.getItem('themeModePreference') || 'system'}
                    onChange={(e) => {
                        const newPreference = e.target.value;
                        localStorage.setItem('themeModePreference', newPreference);
                        setThemeMode(newPreference);
                    }}
                    size="small"
                    sx={{
                        mr: 2,
                        color: 'inherit',
                        '.MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(255,255,255,0.5)' },
                        '.MuiSvgIcon-root': { color: 'inherit' }
                    }}
                    renderValue={(value) => {
                        const options = {
                            light: { icon: <Brightness7Icon fontSize="small" sx={{ mr: 1 }} />, label: 'Light' },
                            dark: { icon: <Brightness4Icon fontSize="small" sx={{ mr: 1 }} />, label: 'Dark' },
                            system: { icon: <SettingsIcon fontSize="small" sx={{ mr: 1 }} />, label: 'System' }
                        };
                        const option = options[value] || options['system'];
                        return (
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                {option.icon}
                                {option.label}
                            </Box>
                        );
                    }}
                >
                    <MenuItem value="light" sx={{ display: 'flex', alignItems: 'center' }}>
                        <Brightness7Icon sx={{ mr: 1 }} fontSize="small" /> Light
                    </MenuItem>
                    <MenuItem value="dark" sx={{ display: 'flex', alignItems: 'center' }}>
                        <Brightness4Icon sx={{ mr: 1 }} fontSize="small" /> Dark
                    </MenuItem>
                    <MenuItem value="system" sx={{ display: 'flex', alignItems: 'center' }}>
                        <SettingsIcon sx={{ mr: 1 }} fontSize="small" /> System
                    </MenuItem>
                </Select>


                {user ? (
                    <Box sx={{ flexGrow: 0 }}>
                        <Tooltip title="Open settings">
                            <IconButton onClick={handleOpenUserMenu} sx={{ p: 0 }}>
                                <Avatar alt={user.username} >{user.username.charAt(0).toUpperCase()}</Avatar>
                            </IconButton>
                        </Tooltip>
                        <Menu
                            sx={{ mt: '45px' }}
                            id="menu-appbar"
                            anchorEl={anchorElUser}
                            anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
                            keepMounted
                            transformOrigin={{ vertical: 'top', horizontal: 'right' }}
                            open={Boolean(anchorElUser)}
                            onClose={handleCloseUserMenu}
                        >
                            <MenuItem onClick={handleSettings}>Settings</MenuItem>
                            <MenuItem onClick={handleLogout}>Logout</MenuItem>
                        </Menu>
                    </Box>
                ) : (
                    <Button color="inherit" onClick={() => navigate('/login')}>Login</Button>
                )}
            </Toolbar>
        </AppBar>
    );
};

export default TopBar;