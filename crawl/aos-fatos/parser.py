from bs4 import BeautifulSoup
import json



def parse_page(page_path):
    list_of_dicts = []
    with open(page_path, "r") as f:
        soup = BeautifulSoup(f, 'html.parser')

    facts_list = soup.find("div", class_="facts-list") 
    facts = soup.find_all("div", class_="fact") 
    for item in facts[3:]:
        title = item.div.h4.string
        date = item.p.text
        aos_fatos_link = item.p.a
        if(aos_fatos_link):
            aos_fatos_link = aos_fatos_link["href"]

        fact_check = item.div.p.text 

        ## metatags
        metatags = item.find_all("p", class_="metatags upper")
        if(len(metatags)>1):
            print("More fields than expected in metatags")
        if(metatags):
            metatags = metatags[0]
            all_tags = metatags.find_all("span")
            if("Tema" in all_tags[0].text):
                tema = all_tags[0].text
                origem = all_tags[1].text
            else:
                tema = all_tags[1].text
                origem = all_tags[0].text

        links = item.find_all("a",class_="btn btn-text highlight")
        for l in links:
            if("LEIA MAIS" in l.text):
                leia_mais = l["href"]
            elif("ORIGEM" in l.text):
                origem_links = l["href"]
            elif("FONTE" in l.text):
                fonte = l["href"]

        date_list = item.find_all("div",class_="date-list")
        if(len(date_list)!= 1):
            if(len(date_list)>1):
                print("Error in date list, double check")
            else:
                print("NULL date list, proceed")
            repetition_count = "null"
            year_days_pair = []
        else:
            date_list = date_list[0]
            repetition_count = date_list.find("span",class_="repetitions highlight").text 
            years = date_list.find_all("span",class_="year")
            days = date_list.find_all("span",class_="days")
            year_days_pair = []
            if(len(years) == len(days)):
                for i in range(len(years)):
                    year_days_pair.append((years[i].text, days[i].text))
            else:
                print("error in days and years")

        ## adding it all to a dictionary
        data_dict = {
            "title":title,
            "date":date,
            "aos_fatos_link":aos_fatos_link,
            "fact_check": fact_check,
            "tema": tema,
            "origem": origem,
            "origem_links": origem_links,
            "repetition_count": repetition_count,
            "year_days_pair": year_days_pair
        }

        list_of_dicts.append(data_dict)
    return list_of_dicts 



data_dict_list = []
for i in range(1,500):
    try:
        data_dict = parse_page(page_path="pages/page_"+str(i)+".html")
        data_dict_list.append(data_dict)
    except:
        pass

final = json.dumps(data_dict_list, indent=2)

with open("dump.json","w") as f:
    f.write(final)