import re
from helper_functions.dataFetching import *


CURRENT_URL = "https://weather.gc.ca/en/location/index.html?coords="
HISTORICAL_URL = "https://climate.weather.gc.ca/climate_data/hourly_data_e.html?StationID=<stationid>&Year=<year>&Month=<month>&Day=<day>"


def get_current_weather(locationCoords):
    weather_soup = get_soup(CURRENT_URL + locationCoords)
    if not weather_soup:
        return None

    try:
        text = weather_soup.get_text("\n")
        patterns = {
            "Temperature": r"Temperature:\s*(-?\d+\.\d+)",
            "Pressure": r"Pressure:\s*(\d+\.\d+)",
            "Tendency": r"Tendency:\s*([a-zA-Z\s]+)(?:\n|$)",
            "Condition": r"Condition:\s*([a-zA-Z\s]+)(?:\n|$)",
            "Dew Point": r"Dew point:\s*(-?\d+\.\d+)",
            "Humidity": r"Humidity:\s*(\d+)%",
            "Wind Speed": r"Wind:\s*\w*\s*\d*\s*\w*\s*(\d+)\s*km/h",
            "Visibility": r"Visibility:\s*(\d+)\s*km"
        }
        data = {k: (re.search(p, text).group(1) if re.search(p, text) else "Not observed") for k, p in patterns.items()}

    except AttributeError:
        print("[ERROR] Weather data missing.")
        return None

    return data


def get_historic_weather(stationId, year, month, day, hour):
    weather_url = HISTORICAL_URL.replace("<stationid>", stationId)
    weather_url = weather_url.replace("<year>", str(int(year)))
    weather_url = weather_url.replace("<month>", str(int(month)))
    weather_url = weather_url.replace("<day>", str(int(day)))

    weather_soup = get_soup(weather_url)
    if not weather_soup:
        return None

    try:
        text = weather_soup.get_text("|")
        text = re.sub(r"\|\s*\|", "|", text)  # Standardize "||" separators
        text = re.sub(r"\s+", "", text).strip()  # Remove excess whitespace
        pattern_str = (f"{hour:02d}:00\|(-?\d+\.\d+)?\|(-?\d+\.\d+)?\|(\d+)?\|"
                       f"(\d+\.?\d*)?\|(\d+)?\|(\d+)?\|(\d+\.\d+)?\|(\d+\.\d+)?\|")
        match = re.search(pattern_str, text)
        data = {
            "Temperature": match.group(1),
            "Dew Point": match.group(2),
            "Humidity": match.group(3),
            "Precipitation": match.group(4),
            "Wind Dir": match.group(5),
            "Wind": match.group(6),
            "Visibility": match.group(7),
            "Pressure": match.group(8),
        }

    except AttributeError:
        print("[ERROR] Weather data missing.")
        return None

    return data


niagara_coords = "43.108,-79.072"
niagara_station = "50131"