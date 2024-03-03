import './App.scss';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import { MainViewComponent } from './Components/MainViewComponent/MainViewComponent';
import HeaderComponent from './Components/HeaderComponent/HeaderComponent';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <div className='root'>
        <HeaderComponent></HeaderComponent>
        <MainViewComponent></MainViewComponent>
      </div>
    </ThemeProvider>
  );
}

export default App;