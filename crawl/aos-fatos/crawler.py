import requests
import time
import re

for i in range(500):
    url = 'https://www.aosfatos.org/todas-as-declara%C3%A7%C3%B5es-de-bolsonaro/?page='+str(i)+'#i'
    r = requests.get(url, allow_redirects=True)

    open('page_'+str(i)+'.html', 'wb').write(r.content)
    time.sleep(5)