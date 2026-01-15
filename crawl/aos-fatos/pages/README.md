
# `crawl/aos-fatos/pages`

This directory stores **local copies of the Aos Fatos claim listing pages** that were downloaded during data collection for the BOL4Y dataset.  Keeping a local cache of pages allows the parsing step to run offline and ensures that the dataset can be reproduced even if the website layout changes.

## Contents

The pages are packaged as a single archive named `aos_fatos_pages.zip`.  Extract the archive to obtain individual HTML files named `page_0.html`, `page_1.html`, …, `page_499.html`.  Each file corresponds to a paginated listing on the Aos Fatos website.

To extract the archive and inspect the pages, run:

```bash
cd crawl/aos-fatos/pages
unzip aos_fatos_pages.zip
ls page_*.html  # see all saved pages
```

Once extracted, you can parse these pages with the `parser.py` script in
the parent folder.  The parser uses BeautifulSoup to extract claim
titles, descriptions, dates, sources and links to Escriba transcripts.
The resulting structured data are saved into a CSV/JSON file (`dump.csv`),
which is then used by the transcription downloader and the training
pipeline.

