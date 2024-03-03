import React from 'react'
import './StatisticsComponent.scss'
import { IOllamaStatistics } from '../../Services/Ollama-Service'
import { Box, Paper, Typography } from '@mui/material'

const StatisticsComponent = (props: any) => {
  return (
    <div className='statistics-container'>
      <div>toks/sec :</div>
      <div>{props.stats?.token_per_second}</div>

      <div>Total(s)</div>
      <div>{props.stats?.total_duration}</div>
    </div>
  )
}

export default StatisticsComponent