// src/components/ConfirmDialog.js
import React from 'react';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';

export default function ConfirmDialog({
                                          open,
                                          onClose,    // This is for backdrop click, Esc, or the explicit "Cancel" button
                                          onConfirm,  // This is for the "Confirm" button
                                          title,
                                          message,
                                          confirmText = "Confirm",
                                          cancelText = "Cancel"
                                      }) {
    return (
        <Dialog
            open={open}
            onClose={onClose} // MUI calls this on backdrop/Esc
            aria-labelledby="confirm-dialog-title"
            aria-describedby="confirm-dialog-description"
        >
            <DialogTitle id="confirm-dialog-title">{title}</DialogTitle>
            <DialogContent>
                <DialogContentText id="confirm-dialog-description">
                    {message}
                </DialogContentText>
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose} color="primary"> {/* Correct: This button calls onClose */}
                    {cancelText}
                </Button>
                <Button onClick={onConfirm} color="error" autoFocus> {/* Correct: This button calls onConfirm */}
                    {confirmText}                                   {/* onConfirm (handleConfirmDeletion) will then close the dialog */}
                </Button>
            </DialogActions>
        </Dialog>
    );
}