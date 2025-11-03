"""Weather tool using Open-Meteo API"""

from typing import Dict, Any, Optional
import requests
from datetime import datetime, timedelta
from rag_system.core.config import get_config

class WeatherTool:
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config.get('tools.weather.enabled', True)
    
    def get_weather(self, location: str, date: Optional[str] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'Weather tool is disabled'}
        
        try:
            coords = self._geocode(location)
            if not coords:
                return {'error': f'Could not geocode location: {location}'}
            
            lat, lon = coords
            
            if date:
                weather_data = self._get_forecast(lat, lon, date)
            else:
                weather_data = self._get_current(lat, lon)
            
            return {
                'location': location,
                'latitude': lat,
                'longitude': lon,
                'data': weather_data
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _geocode(self, location: str) -> Optional[tuple]:
        try:
            url = f"https://nominatim.openstreetmap.org/search"
            params = {
                'q': location,
                'format': 'json',
                'limit': 1
            }
            headers = {'User-Agent': 'RAG-System/1.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            results = response.json()
            if results:
                return (float(results[0]['lat']), float(results[0]['lon']))
            return None
        except Exception:
            return None
    
    def _get_current(self, lat: float, lon: float) -> Dict[str, Any]:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current_weather': True,
            'hourly': 'temperature_2m,precipitation,windspeed_10m'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data
    
    def _get_forecast(self, lat: float, lon: float, date: str) -> Dict[str, Any]:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': 'temperature_2m,precipitation,windspeed_10m',
            'start_date': date,
            'end_date': date
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data

_weather_tool = None

def get_weather_tool() -> WeatherTool:
    global _weather_tool
    if _weather_tool is None:
        _weather_tool = WeatherTool()
    return _weather_tool
