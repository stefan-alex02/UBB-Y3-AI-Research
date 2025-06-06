import React, { useState, useEffect } from 'react';
import { Box, Typography, CircularProgress, Paper, Alert, Tabs, Tab, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import JsonViewer from './JsonViewer';
import PlotViewer from './PlotViewer';
// You might need a CSV parsing library like 'papaparse' if you want to render CSVs as tables nicely
// import Papa from 'papaparse';

const ArtifactViewer = ({ artifactName, artifactType, artifactContent, artifactUrl, title }) => {
    // artifactContent: for JSON, text, log, CSV (as string)
    // artifactUrl: for images/plots directly (can also be a data URL if content is blob)

    const [parsedCsvData, setParsedCsvData] = useState(null);

    useEffect(() => {
        if (artifactType === 'csv' && artifactContent && typeof artifactContent === 'string') {
            try {
                // Basic CSV parsing (for more complex CSVs, use a library like Papaparse)
                const rows = artifactContent.trim().split('\n');
                const headers = rows[0].split(',');
                const data = rows.slice(1).map(row => {
                    const values = row.split(',');
                    let obj = {};
                    headers.forEach((header, i) => obj[header.trim()] = values[i]?.trim());
                    return obj;
                });
                setParsedCsvData({ headers, data });
            } catch (e) {
                console.error("Error parsing CSV content:", e);
                setParsedCsvData(null); // Fallback to text display
            }
        } else {
            setParsedCsvData(null);
        }
    }, [artifactType, artifactContent]);


    if (!artifactType) {
        return <Alert severity="warning">Artifact type not specified.</Alert>;
    }

    switch (artifactType.toLowerCase()) {
        case 'json':
            return <JsonViewer jsonData={artifactContent} title={title || artifactName} />;
        case 'png':
        case 'jpeg':
        case 'jpg':
        case 'gif':
        case 'svg': // Add other image types if needed
            return <PlotViewer artifactUrl={artifactUrl || URL.createObjectURL(artifactContent)} altText={artifactName} title={title || artifactName} />;
        case 'log':
        case 'txt':
            return (
                <Paper elevation={1} sx={{ p: 2, my: 1, maxHeight: '500px', overflow: 'auto', backgroundColor: 'background.default' }}>
                    {title && <Typography variant="subtitle1" gutterBottom>{title || artifactName}</Typography>}
                    <Box component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: '0.75rem' }}>
                        {artifactContent || 'No content.'}
                    </Box>
                </Paper>
            );
        case 'csv':
            if (parsedCsvData) {
                return (
                    <Paper elevation={1} sx={{ p: 2, my: 1, overflow: 'auto' }}>
                        {title && <Typography variant="subtitle1" gutterBottom>{title || artifactName}</Typography>}
                        <TableContainer>
                            <Table stickyHeader size="small">
                                <TableHead>
                                    <TableRow>
                                        {parsedCsvData.headers.map((header) => (
                                            <TableCell key={header} sx={{fontWeight: 'bold'}}>{header}</TableCell>
                                        ))}
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {parsedCsvData.data.map((row, rowIndex) => (
                                        <TableRow key={rowIndex}>
                                            {parsedCsvData.headers.map((header) => (
                                                <TableCell key={`${rowIndex}-${header}`}>{row[header]}</TableCell>
                                            ))}
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Paper>
                );
            }
            // Fallback for CSV if parsing fails or not implemented fully
            return (
                <Paper elevation={1} sx={{ p: 2, my: 1, maxHeight: '500px', overflow: 'auto' }}>
                    {title && <Typography variant="subtitle1" gutterBottom>{title || artifactName} (Raw CSV)</Typography>}
                    <Box component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: '0.75rem' }}>
                        {artifactContent || 'No CSV content.'}
                    </Box>
                </Paper>
            );
        default:
            return <Alert severity="info">Unsupported artifact type: {artifactType}. Cannot display preview for "{artifactName}".</Alert>;
    }
};

export default ArtifactViewer;