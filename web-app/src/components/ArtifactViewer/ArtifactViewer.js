import React, {useState, useEffect, useMemo} from 'react';
import {
    Box,
    Typography,
    CircularProgress,
    Paper,
    Alert,
    Tabs,
    Tab,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TableSortLabel
} from '@mui/material';
import JsonViewer from './JsonViewer';
import PlotViewer from './PlotViewer';
import Papa from 'papaparse';
import {getComparator, stableSort} from "../../utils/tableUtils";
// You might need a CSV parsing library like 'papaparse' if you want to render CSVs as tables nicely
// import Papa from 'papaparse';

const ArtifactViewer = ({
                            artifactName, artifactType, artifactContent, artifactUrl, title,
                            onImageZoom, // For zoom icon on PlotViewer
                            // CSV Sorting Props
                            csvOrder,
                            csvOrderBy,
                            onCsvSortRequest,
                            isSortable
                        }) => {
    const [parsedCsvData, setParsedCsvData] = useState(null);
    const [csvError, setCsvError] = useState(null);
    const typeLower = artifactType?.toLowerCase();

    useEffect(() => {
        if (typeLower === 'csv' && artifactContent && typeof artifactContent === 'string') {
            setCsvError(null); // Reset CSV error
            Papa.parse(artifactContent, {
                header: true,
                skipEmptyLines: true,
                dynamicTyping: true, // Tries to convert numbers/booleans
                complete: (results) => {
                    if (results.errors.length > 0) {
                        console.error("Papaparse errors:", results.errors);
                        const errorMessages = results.errors.map(err => err.message).join('; ');
                        setCsvError(`Error parsing CSV: ${errorMessages}`);
                        setParsedCsvData(null);
                    } else {
                        setParsedCsvData({ headers: results.meta.fields || [], data: results.data || [] });
                    }
                },
                error: (error) => {
                    console.error("Papaparse critical error:", error);
                    setCsvError(`Critical error parsing CSV: ${error.message}`);
                    setParsedCsvData(null);
                }
            });
        } else {
            setParsedCsvData(null); // Clear if not CSV or no content
            setCsvError(null);
        }
    }, [typeLower, artifactContent]);

    const sortedCsvDataForDisplay = useMemo(() => {
        // Check if parsedCsvData and its data property exist AND data is an array
        if (typeLower === 'csv' && isSortable && csvOrderBy &&
            parsedCsvData && Array.isArray(parsedCsvData.data) && parsedCsvData.data.length > 0) {
            return stableSort(parsedCsvData.data, getComparator(csvOrder, csvOrderBy));
        }
        // If not sorting, or not CSV, or parsedCsvData is null, or data is not an array, return original or empty
        return (parsedCsvData && Array.isArray(parsedCsvData.data)) ? parsedCsvData.data : [];
    }, [typeLower, parsedCsvData, csvOrder, csvOrderBy, isSortable]);

    if (!typeLower) {
        return <Alert severity="warning">Artifact type not specified.</Alert>;
    }

    switch (typeLower) {
        case 'json':
            console.log("ArtifactViewer received for JSON:", typeof artifactContent, artifactContent); // DEBUG
            return <JsonViewer jsonData={artifactContent} title={title || artifactName} />;
        case 'image':
            if (!artifactUrl) return <Alert severity="warning">Image URL missing.</Alert>;
            return <PlotViewer artifactUrl={artifactUrl} altText={artifactName} title={title || artifactName} onZoom={onImageZoom} />; // Pass onImageZoom
        case 'log':
        case 'txt':
            return (
                <Paper elevation={0} sx={{ p: 1, height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                    {title && <Typography variant="subtitle2" gutterBottom sx={{px:1, pt:1, flexShrink:0}}>{title || artifactName}</Typography>}
                    <Box component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: '0.75rem', flexGrow: 1, overflow: 'auto', margin: 0, padding: (theme) => theme.spacing(1), backgroundColor: theme => theme.palette.mode === 'dark' ? theme.palette.grey[800] : theme.palette.grey[100], borderRadius: (theme) => theme.shape.borderRadius / 2 }}>
                        {artifactContent || 'No content.'}
                    </Box>
                </Paper>
            );
        case 'csv':
            if (csvError) {
                return <Alert severity="error" sx={{m:1, flexShrink:0}}>{csvError}</Alert>; // Display CSV parsing error
            }
            // Ensure parsedCsvData AND parsedCsvData.headers exist before trying to render table
            if (parsedCsvData && parsedCsvData.headers && Array.isArray(sortedCsvDataForDisplay)) {
                const createSortHandler = (property) => (event) => {
                    if (onCsvSortRequest && isSortable) {
                        onCsvSortRequest(property);
                    }
                };

                return (
                    <Paper elevation={0} sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow:'hidden', alignItems: 'center' /* Center table if narrower than container */ }}>
                        {title && <Typography variant="subtitle2" gutterBottom sx={{px:1, pt:1, flexShrink:0, alignSelf: 'flex-start'}}>{title || artifactName}</Typography>}
                        <TableContainer sx={{ flexGrow: 1, overflow: 'auto', width: 'fit-content', maxWidth: '100%'}}>
                            <Table stickyHeader size="small">
                                <TableHead>
                                    <TableRow>
                                        {parsedCsvData.headers.map((header) => (
                                            <TableCell key={header} sortDirection={isSortable && csvOrderBy === header ? csvOrder : false} sx={{fontWeight: 'bold', backgroundColor: 'background.paper'}}>
                                                {isSortable ? (
                                                    <TableSortLabel
                                                        active={isSortable && csvOrderBy === header}
                                                        direction={csvOrderBy === header ? csvOrder : 'asc'}
                                                        onClick={createSortHandler(header)}
                                                    >
                                                        {header}
                                                    </TableSortLabel>
                                                ) : header }
                                            </TableCell>
                                        ))}
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {/* sortedCsvDataForDisplay is now guaranteed to be an array here */}
                                    {sortedCsvDataForDisplay.map((row, rowIndex) => (
                                        <TableRow key={rowIndex} hover>
                                            {parsedCsvData.headers.map((header) => (
                                                <TableCell key={`${rowIndex}-${header}`}>{String(row[header] ?? '')}</TableCell>
                                            ))}
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                        {sortedCsvDataForDisplay.length === 0 && !csvError && <Typography sx={{p:2, textAlign:'center'}}>CSV is empty or no data to display.</Typography>}
                    </Paper>
                );
            }
            // If parsedCsvData or parsedCsvData.headers is null/undefined, show processing or empty state
            return <Box sx={{p:2, textAlign:'center'}}><Typography>Processing CSV data or no headers found...</Typography></Box>;
        case 'model':
        case 'info': // Handling for .pt files
            return (
                <Paper elevation={1} sx={{ p: 2, my: 1, backgroundColor: 'action.disabledBackground' }}>
                    <Typography variant="subtitle1" gutterBottom>{title || artifactName}</Typography>
                    <Typography variant="body2" color="text.secondary">
                        Model files (.pt, .pth) are binary and cannot be previewed directly.
                        They are intended for use by the ML pipeline.
                    </Typography>
                </Paper>
            );
        default:
            return <Alert severity="info">Unsupported/Unknown artifact type: '{artifactType}'. Cannot display preview for "{artifactName}".</Alert>;
    }
};

export default ArtifactViewer;