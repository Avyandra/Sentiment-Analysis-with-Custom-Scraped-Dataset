from bs4 import BeautifulSoup as BS
#to handle dynamically loaded content in websites using javascript
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

url = "https://teslamotorsclub.com/tmc/threads/my-fsd-11-4-4-review.316063/"


#initialize our headless chrome to be loaded by selenium
driver = webdriver.Chrome()

#get the url and wait 10 secs to load any dynamically loaded content
driver.get(url)

#wait 10 seconds, can use webdriver wait and class condition for more robust result
time.sleep(10)

#save the html in page
page = driver.page_source

soup = BS(page, 'lxml')
scraped_content = []
container = soup.find('div', class_='p-body-inner')
if container:
    wrapper = container.find_all("div", class_="bbWrapper")
    for w in wrapper:
        text = w.get_text(strip=True, separator ='\n')
        if text:
            scraped_content.append(text)
            print(text)
            print("-"*80)

#content = soup.find(tag, id=class_name)

driver.quit()

df = pd.DataFrame(scraped_content)

#save as csv
df.to_csv("teslamotorsmyfsd.csv", index=False)
