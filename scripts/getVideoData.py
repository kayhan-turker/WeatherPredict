import subprocess, yt_dlp, re

from settings import *
from config import *
from getWeatherData import *
from helper_functions.dataFetching import *


# Constants
SUN_URL = "https://www.timeanddate.com/sun/canada/niagara-falls"
YOUTUBE_URL = "https://www.youtube.com/watch?v=K5ZEJWCTSzQ"
NIAGARA_COORDS = "43.108,-79.072"
REFRESH_INTERVAL = 7200  # Refresh URL every 2 hours


def get_soup(url):
    # Fetch and parse HTML with error handling
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None  # Return None so caller knows request failed


def abbreviate(text):
    """Convert multi-word text into abbreviations while keeping numeric values unchanged."""
    if not text:
        return text  # Preserve missing values
    return "".join(w[0].upper() for w in text.split()) if not text.replace('.', '', 1).replace('-', '', 1).isdigit() else text


def get_weather_string():
    # Fetch weather and sun data, handling connection failures
    sun_soup = get_soup(SUN_URL)
    if not sun_soup:
        return None  # Skip if request failed

    try:
        sun_alt = sun_soup.find("td", id="sunalt")
        sun_dir = sun_soup.find("td", id="sunaz")
        sun_alt = round(float(re.search(r"[-+]?\d*\.\d+|\d+", sun_alt.text).group()), 1) if sun_alt else "Not observed"
        sun_dir = round(float(re.search(r"[-+]?\d*\.\d+|\d+", sun_dir.text).group()), 1) if sun_dir else "Not observed"
        if sun_alt < -20:
            return None
    except AttributeError:
        print("[ERROR] Sun data missing.")
        return None

    data = get_current_weather(NIAGARA_COORDS)
    data = {k: abbreviate(v) for k, v in data.items()}

    ts = datetime.now().replace(second=round(datetime.now().second, -1) % 60, microsecond=0)
    return (f"{ts.strftime('%Y_%m_%d_%H_%M_%S')}_{data['Temperature']}_{data['Pressure']}_{data['Tendency']}_"
            f"{data['Condition']}_{data['Dew Point']}_{data['Humidity']}_{sun_dir}_{sun_alt}")


def get_stream_url():
    # Fetch fresh YouTube live stream URL to avoid expiration
    try:
        with yt_dlp.YoutubeDL({"format": "best", "quiet": True}) as ydl:
            return ydl.extract_info(YOUTUBE_URL, download=False)["url"]
    except Exception as e:
        print(f"[ERROR] Failed to fetch YouTube stream URL: {e}")
        return None


# Initialize stream URL
stream_url = get_stream_url()
last_url_refresh = time.time()


while True:
    time.sleep((10 - datetime.now().second % 10) % 10)  # Align with 10-second intervals

    ws = get_weather_string()
    if not ws:
        print(f"[SKIP] Weather data unavailable at {datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
        time.sleep(60)
        continue  # Skip this iteration and retry

    # Refresh stream URL if expired
    if time.time() - last_url_refresh > REFRESH_INTERVAL or not stream_url:
        print("[INFO] Refreshing YouTube stream URL...")
        stream_url = get_stream_url()
        last_url_refresh = time.time()

    if stream_url:
        output_image = f"{TRAINING_DATA_PATH + ws}.jpg"
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", stream_url, "-frames:v", "1", "-vf", "scale=960:-1", "-q:v", "1", output_image],
            capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] ffmpeg failed: {result.stderr}")
            print("[INFO] Refreshing YouTube stream URL...")
            stream_url = get_stream_url()
            last_url_refresh = time.time()
            continue  # Retry with a fresh URL

        print(f"[INFO] Screenshot saved: {output_image}")
    else:
        print("[ERROR] Skipping ffmpeg capture due to missing stream URL.")
        time.sleep(60)
