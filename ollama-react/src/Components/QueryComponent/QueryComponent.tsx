import { Paper, InputBase, IconButton, Divider } from '@mui/material'
import SendIcon from '@mui/icons-material/Send';
import { useState } from 'react';

export const QueryComponent = (props: any) => {

    const MINCHARS = 3;
    const [message, setMessage] = useState('');

    const handleMessageChanged = (e: any) => {
        setMessage(e.target.value);
    }

    const sendMessage = () => {
        props.onNewQuery(message);
        setMessage('');
    }

    return (
        <Paper
            component="form"
            sx={{ p: '2px 4px', display: 'flex', alignItems: 'center', width: '100%' }}
        >
            <InputBase
                sx={{ ml: 1, flex: 1 }}
                placeholder="Enter message"
                inputProps={{ 'aria-label': 'enter message' }}
                value={message}
                multiline
                maxRows={5}
                onChange={handleMessageChanged}
            />
            <Divider sx={{ height: 28, m: 0.5 }} orientation="vertical" />
            <IconButton disabled={message.length <= MINCHARS} color="primary" sx={{ p: '10px' }} aria-label="directions" onClick={sendMessage}>
                <SendIcon />
            </IconButton>
        </Paper>
    )
}