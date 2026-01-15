
# Crawl module

This directory contains the scripts used to **collect raw data** for the misinformation span detection datasets described in the ICWSM 2026 paper.  The paper introduces two new datasets (BOL4Y and EI22) which jointly cover more than 500 videos and around 2,400 annotated segments of misinformative claims.
## Purpose

The `crawl` module automates the collection of:

* **Claim metadata** – Titles, publication dates, topics and links to video sources for fact‑checked claims on the Aos Fatos website.  This metadata is necessary to locate the exact segments in which each misinformation claim appears.
* **HTML pages** – Local copies of every claim listing page, stored in the `pages/` subfolder of `aos‑fatos` (compressed as a ZIP file to reduce size).  Capturing the pages allows reproducible parsing without hitting the site repeatedly.
* **Video files** – Downloaded via `yt_dlp` for later transcription with the Whisper model, as described in the paper’s transcript extraction step.
* **Transcripts** – For sources where Aos Fatos already provided a transcript through their Escriba service, the `escriba-crawler` retrieves those transcripts directly.  These transcripts complement the videos; in total the authors obtained 525 videos and 121 textual transcripts
## Contents

| Item | Description |
|---|---|
| `aos‑fatos/` | Scripts and data to scrape claim listings and download videos from the Aos Fatos website.  See its own README for usage details. |
| `escriba‑crawler/` | Scripts to download CSV transcripts from the Aos Fatos Escriba service for videos where a transcript is already available. |
| `README.md` | Placeholder file from the original repository (empty).  You are reading an improved README in the `.txt` format. |

### `aos‑fatos` subfolder

This subfolder holds three main scripts:

* **`crawler.py`** – Loops through pages of Aos Fatos’s claim listings and downloads each HTML page.  The default implementation fetches the declarations of Jair Bolsonaro for the BOL4Y dataset.  Modify the base URL or range to crawl other index pages.
* **`parser.py`** – Uses BeautifulSoup to extract structured information (title, date, topic, origin link, repetition statistics, etc.) from the downloaded HTML pages.  It writes the resulting list of dictionaries to `dump.json`.
* **`download‑videos.py`** – Reads `dump.json`, collects all unique source URLs and downloads each video using `yt_dlp`.  Audio‐only downloads can be configured by adjusting the `ydl_opts` dictionary.  Errors are logged to `log_videos.txt`.

The `pages/` directory inside `aos‑fatos` contains the raw HTML pages used by the parser.  To reduce repository size, the pages are stored in a single ZIP archive.

### `escriba‑crawler` subfolder

This subfolder focuses on downloading transcripts from **Aos Fatos’ Escriba** service.  When a claim page lacks a link to the original video but includes an Escriba transcript, the `escriba.py` script automatically triggers the download from the Escriba export function.  Each transcript is stored as a CSV file in the `data/` subfolder.  These transcripts form part of the 121 textual transcripts mentioned in the paperfile:///home/oai/share/ICWSM_2026___misinformation_span_detection.pdf#:~:text=we%20note%20that%20for%20121,scription%20service.

## Usage

1. **Install dependencies** – The crawler scripts rely on `requests`, `beautifulsoup4`, `yt_dlp` and `playwright`.  Create a virtual environment and install the packages. Example:

```bash
pip install pandas etc.
playwright install  # required for the Escriba crawler
```

2. **Crawl the claim listings** – Navigate to `crawl/aos‑fatos` and run the crawler:

```bash
cd crawl/aos-fatos
python3 crawler.py  # downloads pages/page_0.html … pages/page_499.html
```

3. **Parse the pages** – Convert the HTML pages into structured JSON:

```bash
python3 parser.py  # produces dump.json
```

4. **Download videos** – Use the JSON file to download all source videos:

```bash
python3 download-videos.py  # downloads each video to the current directory
```

5. **Download Escriba transcripts** – If you wish to obtain Escriba transcripts, run the script in `escriba‑crawler`:

```bash
cd ../escriba-crawler
python3 escriba.py  # downloads CSV transcripts into data/
```

## Relation to the paper

The code in this directory underpins the **dataset construction** section of the paper.  The authors built the BOL4Y dataset by scraping Aos Fatos to collect 6,685 claims and 1,595 unique sources.  The `aos‑fatos` crawler replicates the scraping process, while `download‑videos.py` gathers the raw videos for transcription.  The `escriba‑crawler` complements this by fetching ready‑made transcripts when available.  Together, these scripts enable researchers to reproduce the data acquisition pipeline and extend the datasets for new domains.
