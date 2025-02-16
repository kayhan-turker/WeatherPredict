import subprocess, yt_dlp, requests, re, time
from bs4 import BeautifulSoup
from datetime import datetime
from settings import *


def get_soup(url):
    """Fetch and parse HTML with error handling"""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()  # Raise HTTP errors if any
        return BeautifulSoup(response.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None  # Return None so caller knows request failed


def get_weather_string():
    """Fetch weather and sun data, handling connection failures"""
    sun_soup = get_soup("https://www.timeanddate.com/sun/canada/niagara-falls")
    if not sun_soup:
        return None  # Skip this loop if request failed

    try:
        sun_alt = round(float(re.search(r"[-+]?\d*\.\d+|\d+", sun_soup.find("td", id="sunalt").text).group()), 1) \
            if sun_soup.find("td", id="sunalt") else "Not observed"
        if sun_alt < -20:
            return None
        sun_dir = round(float(re.search(r"[-+]?\d*\.\d+|\d+", sun_soup.find("td", id="sunaz").text).group()), 1) \
            if sun_soup.find("td", id="sunaz") else "Not observed"
    except AttributeError:
        print("[ERROR] Sun data missing, skipping this loop.")
        return None

    weather_soup = get_soup("https://weather.gc.ca/en/location/index.html?coords=43.108,-79.072")
    if not weather_soup:
        return None

    try:
        text = weather_soup.get_text("\n")
        patterns = {
            "Temperature": r"Temperature:\s*(-?\d+\.\d+)", "Pressure": r"Pressure:\s*(\d+\.\d+)",
            "Tendency": r"Tendency:\s*([a-zA-Z\s]+)(?:\n|$)", "Condition": r"Condition:\s*([a-zA-Z\s]+)(?:\n|$)",
            "Dew Point": r"Dew point:\s*(-?\d+\.\d+)", "Humidity": r"Humidity:\s*(\d+)%"
        }
        data = {
            k: (lambda v: "".join(w[0] for w in v.split()) if k in ["Tendency", "Condition"] else v.replace("%", ""))(
                re.search(p, text).group(1) if re.search(p, text) else "Not observed") for k, p in patterns.items()}
    except AttributeError:
        print("[ERROR] Weather data missing, skipping this loop.")
        return None

    ts = datetime.now()
    rounded_seconds = round(ts.second, -1) % 60  # Round to nearest 10
    ts = ts.replace(second=rounded_seconds, microsecond=0)  # Update time
    formatted_ts = ts.strftime("%Y_%m_%d_%H_%M_%S")

    return f"{formatted_ts}_{data['Temperature']}_{data['Pressure']}_{data['Tendency']}_{data['Condition']}_{data['Dew Point']}_{data['Humidity']}_{sun_dir}_{sun_alt}"


# Fetch YouTube live stream URL once
try:
    with yt_dlp.YoutubeDL({"format": "best", "quiet": True}) as ydl:
        stream_url = ydl.extract_info("https://www.youtube.com/watch?v=K5ZEJWCTSzQ", download=False)["url"]
except Exception as e:
    print(f"[ERROR] Failed to fetch YouTube stream URL: {e}")
    stream_url = None  # If stream URL fails, still run but skip ffmpeg

# Capture image at every exact hh:mm:x0
while True:
    time.sleep((10 - datetime.now().second % 10) % 10)  # Wait for next 10-second interval

    ws = get_weather_string()
    if ws is None:
        print(f"[SKIP] Weather data unavailable at {datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
        time.sleep(60)
        continue  # Skip this iteration and retry in the next loop

    if stream_url:  # Only run ffmpeg if the stream URL was fetched successfully
        output_image = f"{(TRAINING_DATA_PATH + ws)}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-i", stream_url, "-frames:v", "1", "-vf", "scale=960:-1", "-q:v", "1", output_image],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Screenshot saved as {output_image}")
    else:
        print("[ERROR] Skipping ffmpeg capture due to missing stream URL.")
        time.sleep(60)


