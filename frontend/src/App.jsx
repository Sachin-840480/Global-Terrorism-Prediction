import React from 'react'
import Map from './components/Map'
import Slider from './components/Slider'

export default function App(){
  return (
    <div style={{height:'100vh', display:'flex', flexDirection:'column'}}>
      <div style={{flex:1}}>
        <Map />
      </div>
      <div style={{height:80}}>
        <Slider />
      </div>
    </div>
  )
}