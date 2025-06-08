import React from 'react';
import {
    Box,
    Divider,
    Drawer, IconButton,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Toolbar,
    Tooltip,
    Typography,
    useTheme
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload'; // For Uploaded Images
import ScienceIcon from '@mui/icons-material/Science'; // For Experiments
import SettingsIcon from '@mui/icons-material/Settings'; // For Settings
import HomeIcon from '@mui/icons-material/Home'; // For Home
import {NavLink} from 'react-router-dom'; // For active styling
import useAuth from '../../hooks/useAuth';
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import MenuOpenIcon from '@mui/icons-material/MenuOpen';


const openedMixin = (theme, drawerWidth) => ({
    width: drawerWidth,
    transition: theme.transitions.create('width', {
        easing: theme.transitions.easing.sharp,
        duration: theme.transitions.duration.enteringScreen,
    }),
    overflowX: 'hidden',
});

const closedMixin = (theme, miniDrawerWidth) => ({
    transition: theme.transitions.create('width', {
        easing: theme.transitions.easing.sharp,
        duration: theme.transitions.duration.leavingScreen,
    }),
    overflowX: 'hidden',
    width: miniDrawerWidth,
});

// SideNavHeader: Clickable area at the top of the permanent drawer to toggle it
const SideNavHeader = ({ open, handleToggle, appName = "Dashboard", miniDrawerWidth }) => {
    const theme = useTheme();
    return (
        <Box
            onClick={handleToggle}
            sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: open ? 'space-between' : 'center', // Center icon when closed
                height: `64px`, // Match AppBar height
                px: open ? 2 : 0, // More padding when open for text
                cursor: 'pointer',
                borderBottom: `1px solid ${theme.palette.divider}`,
                '&:hover': { backgroundColor: theme.palette.action.hover },
                // Crucial for centering the icon in mini mode:
                // The ListItemIcon below will be centered within this Box.
            }}
        >
            {open && (
                <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 500, ml: 1 }}>
                    {appName}
                </Typography>
            )}
            {/* Icon is always present, changes based on 'open' state */}
            <IconButton sx={{ color: 'inherit', p: open ? 1 : ((miniDrawerWidth / 8 - (24/8)) / 2) /* Adjust padding for centering in mini */ }}>
                {/* The padding calculation for the IconButton in mini mode is tricky.
                     Target: (miniDrawerWidth - iconSize) / 2.
                     If iconSize is 24px, miniDrawerWidth is 72px: (72 - 24) / 2 = 24px padding on each side.
                     IconButton default padding is theme.spacing(1) = 8px. So we need to add more.
                     Let's simplify and let the parent Box with justifyContent: 'center' handle it mostly.
                     A simple padding for the IconButton itself might be enough.
                 */}
                {open ? <ChevronLeftIcon /> : <MenuOpenIcon />}
            </IconButton>
        </Box>
    );
};


const SideNav = ({
                     permanentDrawerWidth,
                     miniDrawerWidth,
                     open, // This is isPermanentDrawerFull from Layout
                     onTogglePermanentDrawer, // This is handlePermanentDrawerToggle from Layout
                     mobileOpen,
                     onMobileClose,
                     isMobile
                 }) => {
    const { user } = useAuth();
    const theme = useTheme();

    // Define navItems and commonBottomItems with unique keys for mapping
    const navItems = [
        { id: 'home', text: 'Home', icon: <HomeIcon />, path: '/', roles: ['NORMAL', 'METEOROLOGIST'] },
        { id: 'images', text: 'Uploaded Images', icon: <CloudUploadIcon />, path: '/images', roles: ['NORMAL', 'METEOROLOGIST'] },
        { id: 'experiments', text: 'Experiments', icon: <ScienceIcon />, path: '/experiments', roles: ['METEOROLOGIST'] },
    ];

    const commonBottomItems = [
        { id: 'settings', text: 'Settings', icon: <SettingsIcon />, path: '/settings', roles: ['NORMAL', 'METEOROLOGIST'] },
    ];

    const drawerListContent = (isDrawerFullyOpen) => ( // Parameter name clarifies its purpose
        <>
            {/* Header/Toggle for Permanent Drawer (not on mobile) */}
            {!isMobile && (
                <SideNavHeader
                    open={isDrawerFullyOpen}
                    handleToggle={onTogglePermanentDrawer}
                    appName="Dashboard"
                    miniDrawerWidth={miniDrawerWidth} // Pass for potential centering calculations
                />
            )}
            {/* Standard Toolbar for spacing on Mobile (temporary drawer) */}
            {isMobile && <Toolbar />}
            {!isMobile && <Divider />} {/* Divider only if SideNavHeader was rendered */}

            <List sx={{pt: 0.5, flexGrow: 1, overflowY: 'auto', overflowX: 'hidden'}}>
                {navItems
                    .filter(item => user && item.roles.includes(user.role))
                    .map((item) => (
                        <ListItem key={item.id} disablePadding sx={{ display: 'block' }}>
                            <Tooltip title={!isDrawerFullyOpen ? item.text : ""} placement="right" arrow>
                                <ListItemButton
                                    component={NavLink}
                                    to={item.path}
                                    sx={{
                                        minHeight: 48,
                                        justifyContent: isDrawerFullyOpen ? 'flex-start' : 'center', // Center content (icon) when mini
                                        px: isDrawerFullyOpen ? 2.5 : 0, // No horizontal padding on button when mini, icon handles its own
                                        mb: 0.5,
                                        borderRadius: '4px',
                                        mx: 1,
                                        '&.active': {
                                            backgroundColor: theme.palette.action.selected,
                                            '& .MuiListItemIcon-root, & .MuiListItemText-primary': {
                                                color: theme.palette.primary.main,
                                                fontWeight: 500,
                                            },
                                        },
                                    }}
                                >
                                    <ListItemIcon
                                        sx={{
                                            minWidth: 0,
                                            // When mini, ListItemIcon takes full width of button (minus its own internal padding if any)
                                            // to help center its child icon
                                            width: isDrawerFullyOpen ? 'auto' : '100%',
                                            mr: isDrawerFullyOpen ? 2 : 0,
                                            display: 'flex', // Added for centering icon
                                            justifyContent: 'center',
                                            alignItems: 'center', // Added for centering icon
                                            color: 'inherit',
                                        }}
                                    >
                                        {item.icon}
                                    </ListItemIcon>
                                    <ListItemText
                                        primary={item.text}
                                        primaryTypographyProps={{ fontWeight: 'inherit', noWrap: true }}
                                        sx={{
                                            opacity: isDrawerFullyOpen ? 1 : 0,
                                            transition: theme.transitions.create('opacity', { /* ... */ }),
                                            color: 'inherit',
                                            display: isDrawerFullyOpen ? 'block' : 'none', // Hide text completely when mini
                                        }}
                                    />
                                </ListItemButton>
                            </Tooltip>
                        </ListItem>
                    ))}
            </List>
            <Box sx={{ flexGrow: 1 }} />
            <Divider />
            <List sx={{pt:0.5, pb:1}}>
                {commonBottomItems
                    .filter(item => user && item.roles.includes(user.role))
                    .map((item) => (
                        <ListItem key={item.id} disablePadding sx={{ display: 'block' }}>
                            <Tooltip title={!isDrawerFullyOpen ? item.text : ""} placement="right" arrow>
                                <ListItemButton component={NavLink} to={item.path}
                                                sx={{ minHeight: 48, justifyContent: isDrawerFullyOpen ? 'flex-start' : 'center', px: isDrawerFullyOpen ? 2.5 : 0, mb: 0.5, borderRadius: '4px', mx: 1, '&.active': { /* active styles */ } }}
                                >
                                    <ListItemIcon sx={{ minWidth: 0, width: isDrawerFullyOpen ? 'auto' : '100%', mr: isDrawerFullyOpen ? 2 : 0, display:'flex', justifyContent: 'center', alignItems:'center', color: 'inherit' }}>{item.icon}</ListItemIcon>
                                    <ListItemText primary={item.text} primaryTypographyProps={{ fontWeight: 'inherit', noWrap:true }} sx={{ opacity: isDrawerFullyOpen ? 1 : 0, transition: theme.transitions.create('opacity', {duration: theme.transitions.duration.short}), color: 'inherit', display: isDrawerFullyOpen ? 'block' : 'none' }} />
                                </ListItemButton>
                            </Tooltip>
                        </ListItem>
                    ))}
            </List>
        </>
    );

    if (isMobile) {
        return (
            <Drawer
                variant="temporary" open={mobileOpen} onClose={onMobileClose}
                ModalProps={{ keepMounted: true }}
                sx={{ display: { xs: 'block', sm: 'none' }, '& .MuiDrawer-paper': { boxSizing: 'border-box', width: permanentDrawerWidth, display:'flex', flexDirection:'column' } }}
            >
                {drawerListContent(true, true)} {/* Pass true for isFullWidthOpen and isForMobileDrawer */}
            </Drawer>
        );
    }

    return ( // This is the permanent Drawer for desktop
        <Drawer
            variant="permanent"
            sx={{
                display: { xs: 'none', sm: 'block' }, flexShrink: 0, whiteSpace: 'nowrap', boxSizing: 'border-box',
                // These mixins control the DRAWER'S width and its paper's width.
                ...(open && { // 'open' is 'isPermanentDrawerFull'
                    ...openedMixin(theme, 0),
                    '& .MuiDrawer-paper': {
                        ...openedMixin(theme, permanentDrawerWidth),
                        display:'flex', flexDirection:'column'
                    }
                }),
                ...(!open && { // When 'open' is false (drawer is mini)
                    ...closedMixin(theme, 0),
                    '& .MuiDrawer-paper': {
                        ...closedMixin(theme, miniDrawerWidth),
                        display:'flex', flexDirection:'column'
                    }
                }),
            }}
            open={open} // This prop primarily helps MUI apply transitions and accessibility
        >
            {drawerListContent(open)}
        </Drawer>
    );
};

export default SideNav;