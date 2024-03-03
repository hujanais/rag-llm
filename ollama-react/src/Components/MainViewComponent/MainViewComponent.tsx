import './MainViewComponent.scss'
import ModelSelectComponent from '../ModelSelectComponent/ModelSelectComponent'
import ChatComponent from '../ChatComponent/ChatComponent'
import StatisticsComponents from '../StatisticsComponent/StatisticsComponent'

export const MainViewComponent = () => {
  return (
    <div className='mainview-container'>
        <div><ModelSelectComponent></ModelSelectComponent></div>
        <div><ChatComponent></ChatComponent></div>
    </div>
  )
}
