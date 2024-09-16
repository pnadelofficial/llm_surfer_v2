from langchain.text_splitter import TokenTextSplitter
from openai import OpenAI
import nltk
import numpy as np
import os
from typing import Tuple

class Embedder:
    def __init__(self,
                 client: OpenAI,
                 embedding_model: str,
                 result: list,
                 chunk_size=200, 
                 chunk_overlap=50) -> None:
        self.client = client
        self.embedding_model = embedding_model
        self.result = result
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        os.makedirs("./data", exist_ok=True)    
    
    def _chunk_one(self, text: str, max_chunk_size=200) -> list:
        docs = self.text_splitter.split_text(text)
        sentences = nltk.sent_tokenize(text)
        chunks = []
        
        for doc in docs:
            chunk = []
            doc_sents = nltk.sent_tokenize(doc)
            
            if len(doc_sents) > 0:
                for s in sentences:
                    if doc_sents[0] in s:
                        chunk.append(s)
                        break
                chunk.extend(doc_sents[1:-1])  
                for s in sentences:
                    if doc_sents[-1] in s and s not in chunk:
                        chunk.append(s)
                        break
                full_chunk = ' '.join(chunk)
                if len(full_chunk.split()) > max_chunk_size:
                    trimmed_chunk = []
                    word_count = 0
                    for sent in chunk:
                        sent_words = sent.split()
                        if word_count + len(sent_words) <= max_chunk_size:
                            trimmed_chunk.append(sent)
                            word_count += len(sent_words)
                        else:
                            break
                    full_chunk = ' '.join(trimmed_chunk)
                chunks.append(full_chunk)
            else:
                chunks.append(doc)
        return chunks
    
    def _chunk(self) -> None:
        self.chunked_texts = []
        chunks = self._chunk_one(self.result['text'])
        for i, chunk in enumerate(chunks):
            if chunk == '':
                continue
            self.chunked_texts.append({'title': self.result['title'], 'text': chunk, 'chunk_id': i})
        self.all_chunks = [doc['text'] for doc in self.chunked_texts]

    def _embed_single(self, text: str) -> np.ndarray:
        res = self.client.embeddings.create(input=text, model=self.embedding_model)
        return np.array(res.data[0].embedding)
    
    def _embed(self, texts: list) -> np.ndarray:
        res = self.client.embeddings.create(input=texts, model=self.embedding_model)
        return np.array([r.embedding for r in res.data])
    
    def _embed_docs(self, cb: callable = None, bs=2048) -> None:
        self.batches = [self.all_chunks[i:i + bs] for i in range(0, len(self.all_chunks), bs)] # 2048 limit for oai embeddings
        for i, batch in enumerate(self.batches):
            if cb:
                cb(i, len(self.batches))
            self.embedded_docs = np.vstack((self.embedded_docs, self._embed(batch))) if hasattr(self, 'embedded_docs') else self._embed(batch)
    
    def __call__(self, cb: callable = None) -> np.ndarray:
        self._chunk()
        if os.path.exists(f"./data/{self.result['title']}/embeddings.npy"):
            if cb:
                cb(0, 1)
            self.embedded_docs = np.load(f"./data/{self.result['title']}/embeddings.npy")
            return self.embedded_docs
        else:        
            self._embed_docs(cb=cb)
            os.makedirs(f"./data/{self.result['title']}/", exist_ok=True)
            np.save(f"./data/{self.result['title']}/embeddings.npy", self.embedded_docs)
            return self.embedded_docs

class RAG:
    def __init__(self, 
                 client: OpenAI, 
                 embedder: Embedder,
                 embedder_cb: callable = None) -> None:      
        self.client = client
        self.embedder = embedder
        self.embedder(cb=embedder_cb)

        self.embedded_docs = self.embedder.embedded_docs
        self.embedding_model = self.embedder.embedding_model
        self.chunks = self.embedder.chunked_texts
        
    def __call__(self, query: str, k: int=5) -> Tuple[str, list]:
        query_embedding = self.embedder._embed_single(query)
        chunk_ids = (self.embedded_docs @ query_embedding).argsort()[::-1][:k]
        self.context = [self.embedder.all_chunks[i] for i in chunk_ids]
        return "\n".join(self.context), self.context
