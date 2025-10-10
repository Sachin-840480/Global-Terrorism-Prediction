import React, {useState} from 'react'
import axios from 'axios'

export default function Slider(){
  const [horizon, setHorizon] = useState(90)

  const update = (v)=>{
    setHorizon(v)
    axios.get(`/api/predict?horizon_days=${v}`).then(res=>{
      console.log('pred', res.data)
      // In full implementation, we'd set map source data here (mapRef passed via context or external state)
    })
  }

  return (
    <div style={{padding:16}}>
      <label>Prediction horizon (days): {horizon}</label>
      <input type='range' min='1' max='365' value={horizon} onChange={(e)=>update(e.target.value)} />
    </div>
  )
}
