// src/components/Layout/Layout.js
import React, { useState, useEffect } from 'react';
import { Box, useTheme, useMediaQuery, CssBaseline, Toolbar } from '@mui/material'; // Added Toolbar
import TopBar from './TopBar';
import SideNav from './SideNav';
import { Outlet } from 'react-router-dom';

export const PERMANENT_DRAWER_WIDTH = 240;
export const MINI_DRAWER_WIDTH = 72;

const Layout = () => {
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

    const [isPermanentDrawerFull, setIsPermanentDrawerFull] = useState(true);
    const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);

    const handlePermanentDrawerToggle = () => {
        setIsPermanentDrawerFull(!isPermanentDrawerFull);
    };

    const handleMobileDrawerToggle = () => {
        setMobileDrawerOpen(!mobileDrawerOpen);
    };

    const currentVisibleDrawerWidth = isMobile ? 0 : (isPermanentDrawerFull ? PERMANENT_DRAWER_WIDTH : MINI_DRAWER_WIDTH);

    return (
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
            <CssBaseline />
            <TopBar
                onMobileMenuClick={handleMobileDrawerToggle}
                isMobile={isMobile}
                isPermanentDrawerOpen={isPermanentDrawerFull} // Pass this to help TopBar decide its own state
                permanentDrawerWidth={PERMANENT_DRAWER_WIDTH}
                miniDrawerWidth={MINI_DRAWER_WIDTH}
            />
            <SideNav
                permanentDrawerWidth={PERMANENT_DRAWER_WIDTH}
                miniDrawerWidth={MINI_DRAWER_WIDTH}
                open={isPermanentDrawerFull}
                onTogglePermanentDrawer={handlePermanentDrawerToggle}
                mobileOpen={mobileDrawerOpen}
                onMobileClose={handleMobileDrawerToggle}
                isMobile={isMobile}
            />
            <Box
                component="main"
                sx={{
                    flexGrow: 1,
                    // No paddingTop needed here if we add a Toolbar component for spacing
                    pt: 3, // Keep top padding for content below AppBar
                    pb: 3, // Keep bottom padding
                    px: 3, // Keep horizontal padding
                    height: '100vh',
                    overflowY: 'auto',
                    boxSizing: 'border-box',
                    transition: theme.transitions.create('margin-left', {
                        easing: theme.transitions.easing.sharp,
                        duration: isPermanentDrawerFull ? theme.transitions.duration.enteringScreen : theme.transitions.duration.leavingScreen,
                    }),
                    marginLeft: isMobile ? 0 : `${currentVisibleDrawerWidth}px`,
                }}
            >
                {/* Add an empty Toolbar here to create space for the fixed AppBar */}
                {/* This is the standard MUI way to offset content below a fixed AppBar */}
                <Toolbar />
                <Outlet />
            </Box>
        </Box>
    );
};

export default Layout;