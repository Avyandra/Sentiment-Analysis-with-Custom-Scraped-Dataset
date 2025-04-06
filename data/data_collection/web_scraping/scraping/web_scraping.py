from bs4 import BeautifulSoup as BS
#to handle dynamically loaded content in websites using javascript
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

def main():
    url = "https://www.forbes.com/sites/bradtempleton/2022/01/13/a-robocar-specialist-gives-tesla-full-self-drive-an-f/"
    get_webpage_text(url, 'div', 'field-items even', filename="forbes.csv")

def get_webpage_text(url,tag,class_name,filename="scraped_content.csv"):
    #initialize our headless chrome to be loaded by selenium
    driver = webdriver.Chrome()

    #get the url and wait 10 secs to load any dynamically loaded content
    driver.get(url)

    #wait 10 seconds, can use webdriver wait and class condition for more robust result
    time.sleep(10)

    #save the html in page
    page = driver.page_source

    soup = BS(page, 'lxml')

    content = soup.find(tag, class_=lambda x: x and "field-item" in x and "even" in x)
    #content = soup.find(tag, id=class_name)
    scraped_content = []
    if content:
        paragraphs = content.find_all('p')
        for p in paragraphs:
            print(p.text) # Extract and print text from each paragraph
            scraped_content.append(p.text)
    else:
        print("Div not found")

    driver.quit()

    df = pd.DataFrame(scraped_content)

    #save as csv
    #df.to_csv(filename, index=False)


if __name__ == '__main__':
    main()