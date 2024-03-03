import React, { useEffect, useState } from 'react'
import './ChatComponent.scss'
import { Typography, Box, Paper } from '@mui/material'
import { QueryComponent } from '../QueryComponent/QueryComponent'
import { IOllamaStatistics, OllamaService } from '../../Services/Ollama-Service'
import { Subscription } from 'rxjs'
import StatisticsComponent from '../StatisticsComponent/StatisticsComponent'

const ChatComponent = () => {
    const subscriptions = new Subscription();
    const llm = OllamaService.Instance;
    const [answer, setAnswer] = useState<string>('')
    const [statistics, setStatistics] = useState<IOllamaStatistics>();

    useEffect(() => {
        subscriptions.add(llm.onNewMessage.subscribe((obs: string) => {
            setAnswer(prevAnswer => prevAnswer + obs)
        }));
        subscriptions.add(llm.onNewStats.subscribe((stats: IOllamaStatistics) => {
            stats.total_duration = parseFloat(stats.total_duration.toFixed(2))
            stats.token_per_second = parseFloat(stats.token_per_second.toFixed(2))
            setStatistics({ ...stats })
        }));

        return (() => subscriptions.unsubscribe())
    }, []);

    const handleNewQuery = (query: string) => {
        setAnswer('');
        llm.sendQuery(query);
    }

    return (
        <div className='chat-container'>
            <div>
                <QueryComponent onNewQuery={handleNewQuery} ></QueryComponent>
            </div>
            <Paper elevation={5}>
                <Box sx={{ width: '100%' }}>
                    <Typography variant='subtitle1' gutterBottom>
                        {answer}
                    </Typography>
                </Box>
            </Paper>
            <div>
                <StatisticsComponent stats={statistics}></StatisticsComponent>
            </div>
        </div>
    )
}

export default ChatComponent