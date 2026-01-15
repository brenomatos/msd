
# `crawl/aos-fatos`

This subfolder of the `crawl` module contains the code and data used to **scrape the Aos Fatos fact‑checking website**.  
## Directory contents

| File/dir | Purpose |
|---|---|
| `crawler.py` | Loops through pages of the Aos Fatos claim index and downloads each HTML page.  The script currently targets the “Todas as declarações de Bolsonaro” section used for the BOL4Y dataset.  You can modify the base URL and page range to scrape different topics. |
| `pages/` | Raw HTML pages saved from the claim index.  To reduce size the pages are bundled into a single ZIP archive (`aos_fatos_pages.zip`).  Each file is named `page_N.html`, where `N` corresponds to the page number. |
| `parser.py` | Parses the downloaded HTML pages with BeautifulSoup and extracts structured information.  For each claim the parser records fields such as title, date, link to the fact‑check page, etc.  The parser aggregates these dictionaries and writes them to `dump.json`. |
| `dump.json` | JSON file containing a list of lists of claim dictionaries produced by `parser.py`.  It is large (~13 MB) and therefore stored in the repository.  Each entry corresponds to one claim extracted from a page. |
| `download-videos.py` | Reads `dump.json`, extracts all unique `origem_links` URLs (excluding YouTube channel pages), and downloads each video using the `yt_dlp` library.  The script can be configured to download audio only (commented example) or full video.  Downloaded files are saved in the current working directory with the YouTube ID as filename; errors are logged to `log_videos.txt`. |

## Usage

1. **Crawl the claim pages**

   ```bash
   # Download the first 500 pages (modify the range as needed)
   python3 crawler.py
   ```

   The crawler writes each HTML file to `pages/page_N.html`.  You may want to insert a delay between requests (as in the script) to avoid overloading the server.

2. **Parse pages into JSON**

   ```bash
   python3 parser.py
   ```

   This produces `dump.json` which contains a Python list of lists.  Each inner list corresponds to one HTML page and contains dictionaries with the fields described above.

3. **Download videos**

   ```bash
   python3 download-videos.py
   ```

   The downloader reads `dump.json`, extracts every unique `origem_links` URL (excluding channel pages), and downloads the corresponding video using `yt_dlp`.  The default options download the full MP4; the commented `ydl_opts` in the script show how to download audio only.

4. **Manage pages**

   The `pages/` directory stores the raw HTML pages.  To keep the repository lightweight it is compressed into `aos_fatos_pages.zip`.  Extract it when needed:

   ```bash
   unzip aos_fatos_pages.zip -d pages
   ```
