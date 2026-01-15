
# Escriba transcripts (`data/` directory)

This directory contains the raw transcription files downloaded
from the **Escriba** portal of Aos Fatos.  These CSV files were
produced by the `escriba.py` script in the parent folder
(`crawl/escriba-crawler`), which uses Playwright to automate the
export of transcripts from each Escriba page.

## File naming

Each file in this directory is named `<page>-<fact_check_id>.csv`,
where:

- **page** – the numeric index of the claim page in the Aos Fatos dump; and
- **fact_check_id** – the unique identifier of the fact‑check claim.

For example, `11-161.csv` contains the transcript associated with page
`11` and fact‑check `161`.  The mapping between `page`/`fact_check_id`
and the original fact‑check article is stored in `crawl/aos-fatos/dump.csv`.


The raw CSV files are included here for completeness.
