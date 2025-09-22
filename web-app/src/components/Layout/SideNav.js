import React from 'react';
import {
    Box,
    Divider,
    Drawer,
    IconButton,
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
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ScienceIcon from '@mui/icons-material/Science';
import SettingsIcon from '@mui/icons-material/Settings';
import HomeIcon from '@mui/icons-material/Home';
import {NavLink} from 'react-router-dom';
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

const SideNavHeader = ({ open, handleToggle, appName = "Dashboard", miniDrawerWidth }) => {
    const theme = useTheme();
    return (
        <Box
            onClick={handleToggle}
            sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: open ? 'space-between' : 'center',
                height: `64px`,
                px: open ? 2 : 0,
                cursor: 'pointer',
                borderBottom: `1px solid ${theme.palette.divider}`,
                '&:hover': { backgroundColor: theme.palette.action.hover },
            }}
        >
            {open && (
                <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 500, ml: 1 }}>
                    {appName}
                </Typography>
            )}
            <IconButton sx={{ color: 'inherit', p: open ? 1 : ((miniDrawerWidth / 8 - (24/8)) / 2)  }}>
                {open ? <ChevronLeftIcon /> : <MenuOpenIcon />}
            </IconButton>
        </Box>
    );
};


const SideNav = ({
                     permanentDrawerWidth,
                     miniDrawerWidth,
                     open,
                     onTogglePermanentDrawer,
                     mobileOpen,
                     onMobileClose,
                     isMobile
                 }) => {
    const { user } = useAuth();
    const theme = useTheme();

    const navItems = [
        { id: 'home', text: 'Home', icon: <HomeIcon />, path: '/', roles: ['NORMAL', 'METEOROLOGIST'] },
        { id: 'images', text: 'Uploaded Images', icon: <CloudUploadIcon />, path: '/images', roles: ['NORMAL', 'METEOROLOGIST'] },
        { id: 'experiments', text: 'Experiments', icon: <ScienceIcon />, path: '/experiments', roles: ['METEOROLOGIST'] },
    ];

    const commonBottomItems = [
        { id: 'settings', text: 'Settings', icon: <SettingsIcon />, path: '/settings', roles: ['NORMAL', 'METEOROLOGIST'] },
    ];

    const drawerListContent = (isDrawerFullyOpen) => (
        <>
            {!isMobile && (
                <SideNavHeader
                    open={isDrawerFullyOpen}
                    handleToggle={onTogglePermanentDrawer}
                    appName="Dashboard"
                    miniDrawerWidth={miniDrawerWidth}
                />
            )}
            {isMobile && <Toolbar />}
            {!isMobile && <Divider />}

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
                                        justifyContent: isDrawerFullyOpen ? 'flex-start' : 'center',
                                        px: isDrawerFullyOpen ? 2.5 : 0,
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
                                            width: isDrawerFullyOpen ? 'auto' : '100%',
                                            mr: isDrawerFullyOpen ? 2 : 0,
                                            display: 'flex',
                                            justifyContent: 'center',
                                            alignItems: 'center',
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
                                            transition: theme.transitions.create('opacity', ),
                                            color: 'inherit',
                                            display: isDrawerFullyOpen ? 'block' : 'none',
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
                                                sx={{ minHeight: 48, justifyContent: isDrawerFullyOpen ? 'flex-start' : 'center', px: isDrawerFullyOpen ? 2.5 : 0, mb: 0.5, borderRadius: '4px', mx: 1, '&.active': {  } }}
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
                {drawerListContent(true, true)}
            </Drawer>
        );
    }

    return (
        <Drawer
            variant="permanent"
            sx={{
                display: { xs: 'none', sm: 'block' }, flexShrink: 0, whiteSpace: 'nowrap', boxSizing: 'border-box',
                ...(open && {
                    ...openedMixin(theme, 0),
                    '& .MuiDrawer-paper': {
                        ...openedMixin(theme, permanentDrawerWidth),
                        display:'flex', flexDirection:'column'
                    }
                }),
                ...(!open && {
                    ...closedMixin(theme, 0),
                    '& .MuiDrawer-paper': {
                        ...closedMixin(theme, miniDrawerWidth),
                        display:'flex', flexDirection:'column'
                    }
                }),
            }}
            open={open}
        >
            {drawerListContent(open)}
        </Drawer>
    );
};

export default SideNav;