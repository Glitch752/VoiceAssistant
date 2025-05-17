import json
import requests
import urllib.parse

def get_weather(location: str) -> str:
    """
    Get the weather for a given location.
    """
    # https://wttr.in/{location}?format=j2 is a nice API that returns a JSON object with the weather information.
    url = f"https://wttr.in/{urllib.parse.quote(location)}?format=j2"
    response = requests.get(url)
    
    if response.status_code == 200:
        # Minify the JSON
        json_data = response.json()
        return json.dumps(json_data, separators=(',', ':'))
    else:
        return json.dumps({"error": "Unable to fetch weather data."})