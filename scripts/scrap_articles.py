from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time, json, re
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs, unquote

BASE = "https://www.deeplearning.ai"
# categories from which to scrape articles
CATEGORIES = {
    "culture": f"{BASE}/the-batch/tag/culture/",
    "research": f"{BASE}/the-batch/tag/research/",
    "data_point": f"{BASE}/the-batch/tag/data-points/",
    "business": f"{BASE}/the-batch/tag/business/",
    "science": f"{BASE}/the-batch/tag/science/",
    "hardware": f"{BASE}/the-batch/tag/hardware/",
}


def driver_setup(headless=False) -> webdriver.Chrome:
    opt = Options()
    if headless:
        opt.add_argument("--headless")
    opt.add_argument("--window-size=1920,1080")
    opt.add_argument("--disable-blink-features=AutomationControlled")
    return webdriver.Chrome(options=opt)


# load all articles of category by clicking Load More buttons
def load_all(driver, delay=3, max_attempts=35):
    print("Start to click 'Load More' buttons...")
    last_count = 0
    no_new_count = 0

    selector = "//div[contains(text(), 'Load')]"

    for attempt in range(max_attempts):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        found = False

        try:
            buttons = driver.find_elements(By.XPATH, selector)
            for button in buttons:
                if button.is_displayed() and button.is_enabled():
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                    time.sleep(1)
                    try:
                        button.click()
                    except Exception:
                        driver.execute_script("arguments[0].click();", button)
                    print(f"Click button on attempt {attempt + 1}")
                    found = True
                    break
        except Exception as e:
            print(f"Error while clicking: {e}")
        # check if new articles loaded using counter
        current_count = len(driver.find_elements(By.TAG_NAME, "article"))
        if current_count > last_count:
            last_count = current_count
            no_new_count = 0
        else:
            no_new_count += 1

        if no_new_count >= max_attempts:
            print("No articles loaded in the last attempts. Stopping.")
            break

        if not found:
            print("The button was not found or not clickable. Stopping.")
            break

        time.sleep(delay)


# find all article links
def article_links_from_tagpage(page_source):
    soup = BeautifulSoup(page_source, "html.parser")
    links = set()
    for a in soup.select("a[href*='/the-batch/']"):
        href = urljoin(BASE, a.get("href", ""))
        # skip links that are not articles or are tags
        if "/tag/" in href or href.rstrip("/").endswith("the-batch"):
            continue
        links.add(href)
    return sorted(links)


def extract_media(soup):
    media_urls = set()
    # blacklist of unwanted URLs which is used to filter out tracking and analytics links
    blacklist = [
        "googletagmanager.com", "facebook.com/tr", "/_next/static/media/",
        "analytics", "pixel", "metrics", "track", 'elevenlabs.io/player',
        "home-wordpress.deeplearning.ai"
    ]
    # extract urls of images from <img> tags
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if not src or src.startswith("data:"):
            continue
        if "_next/image" in src and "url=" in src:
            qs = parse_qs(urlparse(src).query)
            if "url" in qs:
                original = unquote(qs["url"][0])
                if not any(bad in original for bad in blacklist):
                    media_urls.add(original)
        else:
            full_url = urljoin(BASE, src)
            if not any(bad in full_url for bad in blacklist):
                media_urls.add(full_url)

    return list(media_urls)


def get_article(driver, url: str, category: str) -> dict | None:
    try:
        driver.get(url)
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # extract title, subtitle, date, and text
        title = ""
        subtitle = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.contents[0].strip() if h1.contents else h1.get_text(strip=True)
            span = h1.find("span")
            if span:
                subtitle = span.get_text(strip=True)

        # date extraction
        date = ""
        date_div = soup.find("div", string=re.compile(r"[A-Z][a-z]{2} \d{2}, \d{4}"))
        if date_div:
            date = date_div.get_text(strip=True)

        # text extraction
        art = soup.find("article") or soup.find("main") or soup
        paragraphs = []
        for tag in art.find_all(["p", "li"]):
            if tag.find_parent(["nav", "header", "footer"]):
                continue
            txt = tag.get_text(" ", strip=True)
            if not txt:
                continue
            if tag.name == "li":
                txt = "• " + txt
            if not txt.lower().startswith(("image:", "credit:")):
                paragraphs.append(txt)
        text = "\n\n".join(paragraphs)

        if not text.strip():
            return None

        media_urls = extract_media(soup)
        slug = urlparse(url).path.rstrip("/").split("/")[-1]
        doc_id = f"{category}_{slug}"

        return {
            "id": doc_id,
            "title": title,
            "subtitle": subtitle,
            "text": text,
            "date": date,
            "category": category,
            "media_urls": media_urls,
            "source_url": url
        }

    except Exception as e:
        print(f"Failed to extract article: {e}")
        return None


def main():
    out_dir = Path("../output")
    out_dir.mkdir(exist_ok=True)
    driver = driver_setup(headless=True)

    all_docs = []
    # iterate over categories and scrape articles
    for cat, url in CATEGORIES.items():
        print(f"\n=== {cat.upper()} ===")
        driver.get(url)
        time.sleep(3)
        load_all(driver)

        links = article_links_from_tagpage(driver.page_source)
        print(f" {len(links)} article links found")

        for link in links:
            doc = get_article(driver, link, cat)
            if not doc:
                print(f" *[skip] {link}")
                continue
            all_docs.append(doc)
            print(f"  *scraped {doc['id']}")

    driver.quit()
    # write all articles to JSONL file
    with open(out_dir / "the_batch_articles.jsonl", "w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n Parsing completed – {len(all_docs)} articles saved in output/the_batch_articles.jsonl")


if __name__ == "__main__":
    main()
