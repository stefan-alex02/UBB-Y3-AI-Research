// src/utils/dateUtils.js
import { format as formatDateFns } from 'date-fns';

export const formatDateSafe = (timestampInput, formatString = 'PPpp') => {
    if (timestampInput === null || timestampInput === undefined) {
        return 'N/A';
    }

    let date;
    // Check if input is already a Date object (e.g., from DateTimePicker)
    if (timestampInput instanceof Date) {
        date = timestampInput;
    }
        // Check if it's a numeric timestamp (assume seconds if small, milliseconds if large - this heuristic can be tricky)
    // Or if it's a string that can be parsed into a number
    else if (!isNaN(Number(timestampInput))) {
        const numTimestamp = Number(timestampInput);
        // A common heuristic: if timestamp is less than 1 Jan 2000 in seconds, assume it's seconds.
        // Otherwise, assume milliseconds. Adjust this threshold if needed.
        // Timestamp for 1 Jan 2000 00:00:00 UTC is 946684800 seconds.
        if (numTimestamp < 2000000000) { // Likely seconds
            date = new Date(numTimestamp * 1000);
        } else { // Likely milliseconds
            date = new Date(numTimestamp);
        }
    }
    // Check if it's an ISO string
    else if (typeof timestampInput === 'string') {
        date = new Date(timestampInput);
    } else {
        console.warn("formatDateSafe: Unrecognized timestamp input type:", timestampInput);
        return 'Invalid Date Input';
    }

    try {
        if (isNaN(date.getTime())) { // Check if date is valid after parsing/creation
            console.warn("formatDateSafe: Resulting date is invalid for input:", timestampInput);
            return 'Invalid Date';
        }
        return formatDateFns(date, formatString); // Example: 'Jul 2, 2021, 5:07:59 PM'
    } catch (e) {
        console.error("Error formatting date with date-fns:", timestampInput, e);
        return 'Date Format Error';
    }
};