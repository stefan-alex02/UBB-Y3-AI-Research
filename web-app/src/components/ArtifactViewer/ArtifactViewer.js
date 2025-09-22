import React, {useEffect, useMemo, useState} from 'react';
import {
    Alert,
    Box,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TableSortLabel,
    Typography
} from '@mui/material';
import JsonViewer from './JsonViewer';
import PlotViewer from './PlotViewer';
import Papa from 'papaparse';
import {getComparator, stableSort} from "../../utils/tableUtils";

const ArtifactViewer = ({
                            artifactName, artifactType, artifactContent, artifactUrl, title,
                            onImageZoom,
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
            setCsvError(null);
            Papa.parse(artifactContent, {
                header: true,
                skipEmptyLines: true,
                dynamicTyping: true,
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
            setParsedCsvData(null);
            setCsvError(null);
        }
    }, [typeLower, artifactContent]);

    const sortedCsvDataForDisplay = useMemo(() => {
        if (typeLower === 'csv' && isSortable && csvOrderBy &&
            parsedCsvData && Array.isArray(parsedCsvData.data) && parsedCsvData.data.length > 0) {
            return stableSort(parsedCsvData.data, getComparator(csvOrder, csvOrderBy));
        }
        return (parsedCsvData && Array.isArray(parsedCsvData.data)) ? parsedCsvData.data : [];
    }, [typeLower, parsedCsvData, csvOrder, csvOrderBy, isSortable]);

    if (!typeLower) {
        return <Alert severity="warning">Artifact type not specified.</Alert>;
    }

    switch (typeLower) {
        case 'json':
            console.log("ArtifactViewer received for JSON:", typeof artifactContent, artifactContent);
            return <JsonViewer jsonData={artifactContent} title={title || artifactName} />;
        case 'image':
            if (!artifactUrl) return <Alert severity="warning">Image URL missing.</Alert>;
            return <PlotViewer artifactUrl={artifactUrl} altText={artifactName} title={title || artifactName} onZoom={onImageZoom} />;
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
                return <Alert severity="error" sx={{m:1, flexShrink:0}}>{csvError}</Alert>;
            }
            if (parsedCsvData && parsedCsvData.headers && Array.isArray(sortedCsvDataForDisplay)) {
                const createSortHandler = (property) => (event) => {
                    if (onCsvSortRequest && isSortable) {
                        onCsvSortRequest(property);
                    }
                };

                return (
                    <Paper elevation={0} sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow:'hidden', alignItems: 'center'}}>
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
            return <Box sx={{p:2, textAlign:'center'}}><Typography>Processing CSV data or no headers found...</Typography></Box>;
        case 'model':
        case 'info':
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