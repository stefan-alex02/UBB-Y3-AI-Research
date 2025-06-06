import React from 'react';
import { AppBar, Toolbar, IconButton, Typography, Button, Box, Tooltip, Menu, MenuItem, Avatar, Select } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import Brightness4Icon from '@mui/icons-material/Brightness4'; // Dark mode
import Brightness7Icon from '@mui/icons-material/Brightness7'; // Light mode
import SettingsIcon from '@mui/icons-material/Settings'; // System mode (example)
import AccountCircle from '@mui/icons-material/AccountCircle';
import useAuth from '../../hooks/useAuth';
import { useThemeMode } from '../../contexts/ThemeContext';
import { useNavigate } from 'react-router-dom';


const TopBar = ({ drawerWidth, handleDrawerToggle }) => {
    const { user, logout } = useAuth();
    const { mode, setThemeMode } = useThemeMode();
    const navigate = useNavigate();

    const [anchorElUser, setAnchorElUser] = React.useState(null);

    const handleOpenUserMenu = (event) => {
        setAnchorElUser(event.currentTarget);
    };
    const handleCloseUserMenu = () => {
        setAnchorElUser(null);
    };
    const handleLogout = () => {
        logout();
        handleCloseUserMenu();
        navigate('/login');
    };
    const handleSettings = () => {
        navigate('/settings');
        handleCloseUserMenu();
    }

    return (
        <AppBar
            position="fixed"
            sx={{
                width: { sm: `calc(100% - ${drawerWidth}px)` },
                ml: { sm: `${drawerWidth}px` },
            }}
        >
            <Toolbar>
                <IconButton
                    color="inherit"
                    aria-label="open drawer"
                    edge="start"
                    onClick={handleDrawerToggle}
                    sx={{ mr: 2, display: { sm: 'none' } }}
                >
                    <MenuIcon />
                </IconButton>
                <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
                    Cloud Classifier
                </Typography>

                <Select
                    value={localStorage.getItem('themeModePreference') || 'system'} // Store preference separately from actual mode
                    onChange={(e) => {
                        const newPreference = e.target.value;
                        localStorage.setItem('themeModePreference', newPreference);
                        setThemeMode(newPreference);
                    }}
                    size="small"
                    sx={{ mr: 2, color: 'inherit', '.MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(255,255,255,0.5)'}, '.MuiSvgIcon-root': { color: 'inherit'} }}
                >
                    <MenuItem value="light"><Brightness7Icon sx={{mr:1}} fontSize="small"/> Light</MenuItem>
                    <MenuItem value="dark"><Brightness4Icon sx={{mr:1}} fontSize="small"/> Dark</MenuItem>
                    <MenuItem value="system"><SettingsIcon sx={{mr:1}} fontSize="small"/> System</MenuItem>
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