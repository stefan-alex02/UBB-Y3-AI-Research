import { format as formatDateFns } from 'date-fns';

export const formatDateSafe = (timestampInput, formatString = 'PPpp') => {
    if (timestampInput === null || timestampInput === undefined) {
        return 'N/A';
    }

    let date;
    if (timestampInput instanceof Date) {
        date = timestampInput;
    }
    else if (!isNaN(Number(timestampInput))) {
        const numTimestamp = Number(timestampInput);
        if (numTimestamp < 2000000000) {
            date = new Date(numTimestamp * 1000);
        } else {
            date = new Date(numTimestamp);
        }
    }
    else if (typeof timestampInput === 'string') {
        date = new Date(timestampInput);
    } else {
        console.warn("formatDateSafe: Unrecognized timestamp input type:", timestampInput);
        return 'Invalid Date Input';
    }

    try {
        if (isNaN(date.getTime())) {
            console.warn("formatDateSafe: Resulting date is invalid for input:", timestampInput);
            return 'Invalid Date';
        }
        return formatDateFns(date, formatString);
    } catch (e) {
        console.error("Error formatting date with date-fns:", timestampInput, e);
        return 'Date Format Error';
    }
};