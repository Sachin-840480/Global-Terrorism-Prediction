import React, {useEffect, useRef, useState} from 'react'
import mapboxgl from 'mapbox-gl'
import axios from 'axios'

export default function Map(){
  const mapRef = useRef(null)
  const [mapObj, setMapObj] = useState(null)

  useEffect(()=>{
    mapboxgl.accessToken = process.env.MAPBOX_TOKEN || ''
    const map = new mapboxgl.Map({
      container: mapRef.current,
      style: 'mapbox://styles/mapbox/light-v11',
      center: [0,20],
      zoom: 1.5
    })
    map.on('load', ()=> setMapObj(map))
    return ()=> map.remove()
  }, [])

  useEffect(()=>{
    if(!mapObj) return
    axios.get('/api/events').then(res=>{
      if(res.data && res.data.type==='FeatureCollection'){
        if(mapObj.getSource('events')){
          mapObj.getSource('events').setData(res.data)
        }else{
          mapObj.addSource('events',{type:'geojson', data: res.data})
          mapObj.addLayer({
            id:'events-heat',
            type:'heatmap',
            source:'events',
            paint: { 'heatmap-weight': ['get','nkill'] }
          })
        }
      }
    })
  }, [mapObj])

  return <div ref={mapRef} style={{width:'100%', height:'100%'}} />
}
