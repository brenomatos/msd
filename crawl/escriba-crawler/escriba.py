from playwright.sync_api import sync_playwright
import time 
import pandas as pd
import os


def donwload_page(url,dump_page,fact_check_id,sleep_time=5):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        
        with page.expect_download() as download_info:
            page.evaluate("exportCSV")

        download = download_info.value
        # Save downloaded file somewhere
        download.save_as(str(dump_page)+"-"+str(fact_check_id)+".csv")
        print(page.title())
        browser.close()
        time.sleep(sleep_time)


downloaded_files = os.listdir("./")
downloaded_files = [x for x in downloaded_files if x.endswith(".csv")]
downloaded_files = [x[:-4] for x in downloaded_files]

df = pd.read_csv("../dump.csv")
df = df[df["origem_links"].str.contains("escriba")]
# df = df.drop_duplicates(subset=["origem_links"]) # i will not drop duplicates for now. I need the reference to page and fact_check_id to be unique
for index, row in df.iterrows():
    page = row["page"]
    fact_check_id = row["fact_check_id"] 
    url = row["origem_links"]
    if(str(page)+"-"+str(fact_check_id) not in downloaded_files):
        try:
            print(url)
            donwload_page(url,page,fact_check_id)
        except Exception as e:
            print(e)

