from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
# from selenium.webdriver.firefox.service import Service
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import pypdf
import nltk
from tqdm import tqdm
import re
import io
from typing import List, Dict, Any, Tuple

nltk.download('punkt')

class SeleniumService:
    def __init__(self, 
                 service: Service, 
                 args: List[str] = None) -> None:
        self.service = service
        self.args = args or []
        self.options = webdriver.FirefoxOptions()
        for option in self.args:
            self.options.add_argument(option)
        self.service = Service()
        self.driver = webdriver.Chrome(service=self.service, options=self.options)

class Searcher:
    def __init__(self, 
                 query: str, 
                 selenium_service: SeleniumService, 
                 max_results=10, 
                 search_engine='congress') -> None: 
        self.query = query
        self.selenium_service = selenium_service
        self.webdriver = self.selenium_service.driver
        self.max_results = max_results
        self.search_engine = search_engine
    
    def _ddg_search(self) -> List[Dict[str, Any]]:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(self.query, max_results=self.max_results):
                results.append(r)
        self.results = results
        return self.results

    def _congress_search(self) -> List[Dict[str, Any]]:
        if 'OR' in self.query:
            parts = []
            for i, q in enumerate(self.query.split('OR')):
                part = q.strip().replace('"', '')
                if ' ' in part:
                    part = part.replace(' ', '+')
                part = "%5C%22" + part + "%5C%22+"
                if i != len(self.query.split('OR'))-1:
                    part = part + "OR+"
                parts.append(part)
            search = ''.join(parts)
            url = f"https://www.congress.gov/search?q=%7B%22congress%22%3A%22all%22%2C%22search%22%3A%22{search}%22%2C%22bill-status%22%3A%22law%22%7D"
        else:
            if len(self.query.split(' ')) > 1:
                url = f"https://www.congress.gov/search?q=%7B%22congress%22%3A%22all%22%2C%22source%22%3A%22all%22%2C%22search%22%3A%22{'+'.join(self.query.split(' '))}%22%2C%22bill-status%22%3A%22law%22%7D"
            else:
                url = f"https://www.congress.gov/search?q=%7B%22congress%22%3A%22all%22%2C%22source%22%3A%22all%22%2C%22search%22%3A%22{self.query}%22%2C%22bill-status%22%3A%22law%22%7D"
        # time.sleep(5)
        self.webdriver.get(url)
        # time.sleep(10)
        
        more_pages = True
        self.results = []
        current_page = 1
        max_pages = int(self.webdriver.find_elements(By.CLASS_NAME, "results-number")[-1].text.split('of')[-1].strip().replace(',',''))//100 + 1
        while more_pages:
            ol = self.webdriver.find_element(By.TAG_NAME, "ol")
            lis = ol.find_elements(By.XPATH, "//li[@class='expanded']")
            for li in lis:
                soup = BeautifulSoup(li.get_attribute('innerHTML'), "html.parser")
                bicam = soup.find('span', attrs={'class':'result-heading'}).get_text().split('.')
                if len(bicam) > 2:
                    bicam = 'hr'
                else:
                    bicam = 's'
                raw = soup.a['href']
                title = soup.find('span', class_="result-title").get_text().strip()
                congress = raw.split('/')[2][:3]
                bill_no = raw.split('?')[0].split('/')[-1]
                self.results.append((title, congress, bill_no, bicam))
            current_page+=1
            if current_page > max_pages:
                more_pages = False
            else:
                self.webdriver.get(url+f"&page={current_page}")
        self.webdriver.close()    
        return self.results
    
    def _openalex_search(self) -> List[Dict[str, Any]]:
        base_url = self.open_alex_base_url.format(self.query) + '&page={}'
                
        page = 1
        has_more_pages = True
        fewer_than_10k_results = True
        full_results = []
        print("Retrieving results from OpenAlex")
        while has_more_pages and fewer_than_10k_results:
            print(f"Reading OpenAlex Page {page}", end='\r')
            url = base_url.format(page)
            page_with_results = requests.get(url).json()
            
            results = page_with_results['results']
            full_results.extend(results)
            if len(full_results) > self.max_results*20:
                break
    
            page += 1
            per_page = page_with_results['meta']['per_page']
            has_more_pages = len(results) == per_page
            fewer_than_10k_results = per_page * page <= 10000

        self.results = [{'title':r['title'], 'href':r['locations'][0]['landing_page_url']} for r in full_results] 
        return self.results   
    
    def search(self) -> None:
        if self.search_engine == "ddg":
            self._ddg_search()
        elif self.search_engine == "congress":
            self._congress_search()
        elif self.search_engine == 'openalex':
            self.open_alex_base_url = "https://api.openalex.org/works?filter=default.search:{},open_access.is_oa:true&sort=relevance_score:desc"
            self._openalex_search()

    def scrape_from_url(self, url) -> Tuple[str, str]:
        try:
            self.webdriver.get(url)
        except TimeoutException:
            return url, '' 
        except WebDriverException:
            return url, ''
        # for crs pdfs, could be others without this format....
        if 'pdf' in url:
            pages = self.webdriver.find_elements(By.CLASS_NAME, 'page')
            full_text = ''
            for page in pages:
                full_text += page.text
                self.webdriver.find_element(By.ID, 'next').click()
        else:
            try:
                body_html = self.webdriver.find_element(By.TAG_NAME, 'body').get_attribute('innerHTML')
            except NoSuchElementException:
                return url, ''
            soup = BeautifulSoup(body_html, 'html.parser')
            full_text = ' '.join([d.get_text().replace('\n', '') for d in soup.find_all('div', {'class':'section'})])
            if full_text == '':
                full_text = ' '.join([p.get_text().replace('\n', '') for p in soup.find_all('p')]) 
        self.webdriver.close()
        return url, full_text

    def _congress_scrape(self, tup) -> Tuple[str, str, str, str]:
        congress, bill_no, bicam = tup
        if not congress[-1].isdigit():
            congress = congress[:-1]
        url = f"https://api.congress.gov/v3/bill/{congress}/{bicam}/{bill_no}/text"
        session = requests.Session()
        session.params = {"format": 'json'}
        session.headers.update({"x-api-key": '5VhOEr0OcuyhGgZlGRQX26b0av7Jp5JE8qDeStCb'})
        
        res = session.get(url)
        if res:
            data = res.json()
            dict_list = [d for d in data['textVersions'] if len(d['formats']) > 0] 
            res_ex_poss = [d for d in dict_list[-1]['formats'] if d['url'].endswith('xml') or d['url'].endswith('htm')]
            if len(res_ex_poss) > 0:
                res_ex = session.get(res_ex_poss[-1]['url'])
                soup = BeautifulSoup(res_ex.text, features='xml') 
                text = soup.get_text()
                if 'Page Not Found' in text:
                    res_ex = session.get(res_ex_poss[0]['url'])
                    soup = BeautifulSoup(res_ex.text, features='xml')
                    text = soup.get_text()
                poss_title = soup.find("dc:title")
            else:
                res_ex = dict_list[-1]['formats'][-1]
                raw_pdf = session.get(res_ex['url']).content
                pdf_file = io.BytesIO(raw_pdf)
                pdf = pypdf.PdfReader(pdf_file)
                text = '\n'.join([page.extract_text().replace('\n', ' ') for page in pdf.pages])
                poss_title = None
            if poss_title:
                title = poss_title.get_text()
            elif ('xml' in res_ex.url) and not poss_title: 
                try:
                    title = soup.find('official-title').get_text().replace('\n\t\t', '').replace('  ', ' ')
                except Exception as e:
                    # error from api
                    print(e)
                    print(tup, 'law not found. Skipping...')
            else:
                try:
                    title = re.search(r'An Act(.*?\.)', text, re.DOTALL).group(1).strip().replace('\n', '').replace('     ', '')
                except Exception as e:
                    print(e)
                    title = re.search(r'H\. R\.\s+\d+(.*?\.)', text, re.DOTALL).group(1).strip().replace('\n', '').replace('     ', '').replace('   ', '')
            return dict_list[-1]['formats'][-1]['url'], text, title, dict_list[-1]['date'].split('-')[0]
        else:
            # error from api
            print(tup, "law not found. Skipping...")
            return '', '', '', ''    
    
    def __call__(self, cb: callable = None) -> List[Dict[str, Any]]:
        self.search()
        self.results_list = []
        res_to_loop = self.results[:self.max_results] if len(self.results) > self.max_results else self.results
        with tqdm(total=self.max_results) as pbar:
            for i, result in enumerate(res_to_loop):
                if cb:
                    cb(i, len(res_to_loop))
                if len(self.results_list) >= self.max_results:
                    break

                if self.search_engine == 'congress':
                    try:
                        title = result[0]
                        url, text, alt_title, date = self._congress_scrape(result[1:])
                        if text == '':
                            continue
                        self.results_list.append((url, title, text, date, alt_title))
                    except Exception as e:
                        print(e)
                        pass
                else:
                    url, text = self.scrape_from_url(result['href'])
                    
                    title = result['title']

                    if text == '':
                        print('No text found. Skipping...')
                        for r in self.results:
                            if r not in res_to_loop:
                                res_to_loop.append(r)
                                print('New result added.')
                                break
                    else:
                        self.results_list.append((url, title, text, 'No date found', ''))
                pbar.update(1)
        return [{'url':r[0], 'title':r[1], 'text':r[2], 'year':r[3], 'alt_title':r[4]} for r in self.results_list]