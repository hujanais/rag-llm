import { FormControl, IconButton, InputLabel, MenuItem, Select, SelectChangeEvent } from '@mui/material'
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import RemoveCircleOutlineIcon from '@mui/icons-material/RemoveCircleOutline';
import { useState } from 'react';

import './ModelSelectComponent.scss'

const ModelSelectComponent = () => {
    const [model, setModel] = useState<string>('gemma:2b')

    const handleModelChange = (event: SelectChangeEvent) => {
        setModel(event.target.value as string)
    }

    return (
        <div className='model-select-container'>
            <FormControl>
                <InputLabel id="demo-simple-select-label">Model</InputLabel>
                <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    value={model}
                    label="Model"
                    onChange={handleModelChange}>
                    <MenuItem value={'gemma:2b'}>gemma:2b</MenuItem>
                </Select>
            </FormControl>
            <IconButton aria-label="delete">
                <AddCircleOutlineIcon color='primary'/>
            </IconButton>
            <IconButton aria-label="delete" color="error">
                <RemoveCircleOutlineIcon />
            </IconButton>
        </div>
    )
}

export default ModelSelectComponent