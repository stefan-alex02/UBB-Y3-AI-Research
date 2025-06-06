import React from 'react';
import { Box, Drawer, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Toolbar, Divider } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload'; // For Uploaded Images
import ScienceIcon from '@mui/icons-material/Science';       // For Experiments
import SettingsIcon from '@mui/icons-material/Settings';     // For Settings
import HomeIcon from '@mui/icons-material/Home';             // For Home
import { NavLink } from 'react-router-dom'; // For active styling
import useAuth from '../../hooks/useAuth';

const SideNav = ({ drawerWidth, mobileOpen, handleDrawerToggle }) => {
    const { user } = useAuth();

    const navItems = [
        { text: 'Home', icon: <HomeIcon />, path: '/', roles: ['NORMAL', 'METEOROLOGIST'] },
        { text: 'Uploaded Images', icon: <CloudUploadIcon />, path: '/images', roles: ['NORMAL', 'METEOROLOGIST'] },
        { text: 'Experiments', icon: <ScienceIcon />, path: '/experiments', roles: ['METEOROLOGIST'] },
    ];

    const commonBottomItems = [
        { text: 'Settings', icon: <SettingsIcon />, path: '/settings', roles: ['NORMAL', 'METEOROLOGIST'] },
    ]

    const drawerContent = (
        <div>
            <Toolbar /> {/* To align content below AppBar */}
            <Divider />
            <List>
                {navItems.filter(item => user && item.roles.includes(user.role)).map((item) => (
                    <ListItem key={item.text} disablePadding>
                        <ListItemButton
                            component={NavLink}
                            to={item.path}
                            sx={{
                                '&.active': {
                                    backgroundColor: 'action.selected',
                                    fontWeight: 'fontWeightBold',
                                },
                            }}
                        >
                            <ListItemIcon>{item.icon}</ListItemIcon>
                            <ListItemText primary={item.text} />
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
            <Divider sx={{ mt: 'auto' }} /> {/* Pushes settings to bottom if List is not full height */}
            <List sx={{ marginTop: 'auto' }}> {/* For items at the bottom */}
                {commonBottomItems.filter(item => user && item.roles.includes(user.role)).map((item) => (
                    <ListItem key={item.text} disablePadding>
                        <ListItemButton
                            component={NavLink}
                            to={item.path}
                            sx={{ '&.active': { backgroundColor: 'action.selected', fontWeight: 'fontWeightBold' } }}
                        >
                            <ListItemIcon>{item.icon}</ListItemIcon>
                            <ListItemText primary={item.text} />
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
        </div>
    );

    return (
        <Box
            component="nav"
            sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
            aria-label="mailbox folders"
        >
            {/* Temporary Drawer for mobile */}
            <Drawer
                variant="temporary"
                open={mobileOpen}
                onClose={handleDrawerToggle}
                ModalProps={{
                    keepMounted: true, // Better open performance on mobile.
                }}
                sx={{
                    display: { xs: 'block', sm: 'none' },
                    '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
                }}
            >
                {drawerContent}
            </Drawer>
            {/* Permanent Drawer for desktop */}
            <Drawer
                variant="permanent"
                sx={{
                    display: { xs: 'none', sm: 'block' },
                    '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
                }}
                open
            >
                {drawerContent}
            </Drawer>
        </Box>
    );
};

export default SideNav;