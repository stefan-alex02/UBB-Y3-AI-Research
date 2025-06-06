import React, { useState } from 'react';
import { Box, Toolbar } from '@mui/material';
import TopBar from './TopBar';
import SideNav from './SideNav';
import { Outlet } from 'react-router-dom'; // For rendering child routes

const drawerWidth = 240;

const Layout = () => {
    const [mobileOpen, setMobileOpen] = useState(false);

    const handleDrawerToggle = () => {
        setMobileOpen(!mobileOpen);
    };

    return (
        <Box sx={{ display: 'flex' }}>
            <TopBar drawerWidth={drawerWidth} handleDrawerToggle={handleDrawerToggle} />
            <SideNav drawerWidth={drawerWidth} mobileOpen={mobileOpen} handleDrawerToggle={handleDrawerToggle} />
            <Box
                component="main"
                sx={{
                    flexGrow: 1,
                    p: 3,
                    width: { sm: `calc(100% - ${drawerWidth}px)` },
                    minHeight: '100vh', // Ensure content area takes full height
                }}
            >
                <Toolbar /> {/* For spacing below the AppBar */}
                <Outlet /> {/* Child routes will render here */}
            </Box>
        </Box>
    );
};

export default Layout;