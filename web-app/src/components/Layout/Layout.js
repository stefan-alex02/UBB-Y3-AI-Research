import React, {useState} from 'react';
import {Box, CssBaseline, Toolbar, useMediaQuery, useTheme} from '@mui/material';
import TopBar from './TopBar';
import SideNav from './SideNav';
import {Outlet} from 'react-router-dom';

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
                isPermanentDrawerOpen={isPermanentDrawerFull}
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
                    pt: 3,
                    pb: 3,
                    px: 3,
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
                <Toolbar />
                <Outlet />
            </Box>
        </Box>
    );
};

export default Layout;