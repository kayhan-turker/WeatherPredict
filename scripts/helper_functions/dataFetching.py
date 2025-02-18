import requests
from bs4 import BeautifulSoup


def get_soup(url):
    # Fetch and parse HTML with error handling
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None  # Return None so caller knows request failed
