import { FormControl, IconButton, InputLabel, MenuItem, Select, SelectChangeEvent, Typography } from '@mui/material'
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import RemoveCircleOutlineIcon from '@mui/icons-material/RemoveCircleOutline';
import { useEffect, useState } from 'react';

import './ModelSelectComponent.scss'
import { OllamaModel, OllamaService } from '../../Services/Ollama-Service';

const ModelSelectComponent = () => {
    const llm: OllamaService = OllamaService.Instance;
    const [models, setModels] = useState<OllamaModel[]>([])
    const [selectedModelKey, setSelectedModelKey] = useState<string>('none');
    const [selectedModel, setSelectedModel] = useState<OllamaModel>();

    useEffect(() => {
        const fetchData = async () => {
            const _models: OllamaModel[] = await llm.getModels();
            setModels([..._models]);
        
            // auto select the first model in the list.
            if (_models.length > 0) {
                setSelectedModelKey(_models[0].model);
                setSelectedModel(models[0]);
            }
        }

        fetchData();
    }, [llm]);

    const handleModelChange = (event: SelectChangeEvent) => {
        const modelKey = event.target.value as string;
        setSelectedModelKey(modelKey);
        setSelectedModel(models.find(m => m.model === modelKey));
    }

    return (
        <div className='model-select-container'>
            <FormControl style={{minWidth: 150}}>
                <InputLabel id="demo-simple-select-label">Model</InputLabel>
                <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    label="Model"
                    value={selectedModelKey}
                    onChange={handleModelChange}>
                        {models.map((item: OllamaModel) => (
                            <MenuItem key={item.model} value={item.model}>
                                {item.name}
                        </MenuItem>
                    ))}
                </Select>
            </FormControl>
            <IconButton aria-label="delete" disabled={true}>
                <AddCircleOutlineIcon color='primary'/>
            </IconButton>
            <IconButton aria-label="delete" color="error" disabled={true}>
                <RemoveCircleOutlineIcon />
            </IconButton>
                <Typography variant='body1'>
                    <div className='div-model-details'>
                        <Typography variant='h6'>Size(GB)</Typography>
                        <div>{selectedModel?.size}</div>
                        <Typography variant='h6'>Parameters</Typography>
                        <div>{selectedModel?.details?.parameter_size}</div>
                    </div>
                </Typography>
        </div>
    )
}

export default ModelSelectComponent