from openai import OpenAI
from functools import partial
import pandas as pd
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from .searcher import Searcher, SeleniumService
from .rag import Embedder, RAG  
from typing import List, Dict, Any, Tuple
import re
from datetime import datetime
import os
from ast import literal_eval
import platform

class LLMSurfer:
    def __init__(self, 
                 client: OpenAI,
                 llm_name: str, 
                 research_goal: str, 
                 base_prompt: str, 
                 json_schema: dict, 
                 query: str, 
                 args: List[str] = None,
                 embedding_model: str = "text-embedding-3-small",
                 max_results: int = 5, 
                 search_engine: str = 'congress',
                 searcher_cb: callable = None,
                 embedder_cb: callable = None,
                 surfer_cb: callable = None) -> None:
        self.client = client
        self.llm_name = llm_name
        self.embedding_model = embedding_model
        self.research_goal = research_goal
        self.base_prompt = base_prompt
        self.json_schema = json_schema
        self.query = query
        self.args = args or []
        self.max_results = max_results
        self.search_engine = search_engine
        self.searcher_cb = searcher_cb
        self.embedder_cb = embedder_cb
        self.surfer_cb = surfer_cb

        if not platform.processor():
            self.service = Service(GeckoDriverManager().install())
        else:
            self.service = Service()

        self.selenium_service = SeleniumService(
            service = self.service, 
            args=self.args
        )   
        self.searcher = Searcher(
            query=self.query,
            selenium_service=self.selenium_service,
            max_results=self.max_results,
            search_engine=self.search_engine
        )
    
    def get_results(self) -> None:
        self.results = self.searcher(cb=self.searcher_cb)
    
    def process_one(self, result: Dict[str, Any]):
        url = result['url']
        title = result['title']
        print(f"Reading url: {url}")

        embedder = Embedder(
            client=self.client,
            result=result,
            embedding_model=self.embedding_model
        )
        context_str, context = RAG(client=self.client, embedder=embedder, embedder_cb=self.embedder_cb)(self.query)
        if isinstance(title, type(None)):
            title = url
        filled_prompt = self.base_prompt.format(research_goal=self.research_goal, url=re.escape(url), title=re.escape(title), text=re.escape(context_str))
        filled_prompt_list = [{"role":"user", "content":filled_prompt}]
        
        def get_response(self):
            return self.client.chat.completions.create(
                model=self.llm_name,
                messages=filled_prompt_list,
                response_format={
                    "type": "json_schema",
                    "json_schema": self.json_schema
                }
            )
        return partial(get_response, self), context

    def __call__(self, to_excel: bool = True, num_rel_chunks: int = 5) -> List[Tuple[str, Any]]:
        print(f"Collecting links from {self.searcher.search_engine}\r", flush=True)
        self.get_results()
        print(f"{len(self.results)} links collected out of {self.max_results}. The rest are unreachable")
        rel_docs = {}
        print('--'*50)
        for i, result in enumerate(self.results):
            if result['url'] not in rel_docs:
                print(f"Webpage #{i+1}")
                date = result['year']
                alt_title = result['alt_title']
                get_response, context = self.process_one(result)
                try:
                    res = get_response()
                except Exception as e:
                    print(f"Error processing {result['url']}: {e}")
                    continue
                out = literal_eval(res.choices[0].message.content)
                if not out: 
                    print(f"No output from {result['url']}")
                    continue
                relevancy = out['relevancy']
                print(f"Result {i+1}: {relevancy}, {result['title']} because: {out['comment']}")
                if self.search_engine == 'congress':
                    rel_docs[result['title']] = {'title':result['title'], 'url':result['url'], 'relevancy':relevancy, 'llm_comment':out['comment'], 'year': date, 'alternative_title':alt_title}
                else:
                    rel_docs[result['title']] = {'title':result['title'], 'url':result['url'], 'relevancy':relevancy, 'llm_comment':out['comment']}
                for key, value in out.items():
                    if key not in ['title', 'url', 'relevancy', 'comment']:
                        rel_docs[result['title']][key] = value
                
                if num_rel_chunks <= len(context):
                    for i in range(num_rel_chunks):
                        rel_docs[result['title']][f"Most Relevant Chunk {i+1}"] = context[i]
                else:
                    for i in range(len(context)):
                        if i < len(context):
                            rel_docs[result['title']][f"Most Relevant Chunk {i+1}"] = context[i]
                        else:
                            rel_docs[result['title']][f"Most Relevant Chunk {i+1}"] = "No more chunks available."
                
                if self.surfer_cb:
                    self.surfer_cb(i=i, length=len(self.results), result=result, out=out)
                print('--'*50)  
            else:
                print(f"Skipping duplicate URL: {result['title']}")
                continue

        self.rel_docs = rel_docs
        self.df = pd.DataFrame.from_dict(rel_docs, orient='index').reset_index()
        self.df = self.df[self.df.columns[1:]]

        if (to_excel) and (len(self.df) > 0):
            now = datetime.now()
            dt_string = now.strftime("%m-%d-%Y")
            os.makedirs('./saved_searches', exist_ok=True)
            output_path = f'./saved_searches/{self.query}_{self.max_results}_{dt_string}_results.xlsx'
            self.df.to_excel(output_path, index=False)
        return self.df, output_path
