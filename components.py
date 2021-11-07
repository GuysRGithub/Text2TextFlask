import concurrent.futures
import itertools
import operator
import re

from transformers import pipeline, set_seed
from gensim.summarization.bm25 import BM25
import requests
import wikipedia
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline


class QueryProcessor:
    def __init__(self, nlp, keep=None):
        self.nlp = nlp
        self.keep = keep or {'PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ'}

    def generate_query(self, text):
        doc = self.nlp(text)
        query = ' '.join(token.text for token in doc if token.pos_ in self.keep)
        return query


class DocumentRetrieval:
    def __init__(self, url='https://en.wikipedia.org/w/api.php'):
        self.url = url

    def search_pages(self, query):
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json'
        }
        res = requests.get(url=self.url, params=params)
        return res.json()

    @staticmethod
    def search_page(page_id):
        res = wikipedia.page(pageid=page_id)
        return res.content

    def search(self, query):
        pages = self.search_pages(query)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            process_list = [executor.submit(self.search_page, page['pageid']) for page in pages['query']['search']]
            docs = [self.post_process(doc.result()) for doc in process_list]
        return docs

    @staticmethod
    def post_process(doc):
        pattern = '|'.join([
            '== References ==',
            '== Further reading ==',
            '== External links',
            '== See also ==',
            '== Sources ==',
            '== Notes ==',
            '== Further references ==',
            '== Footnotes ==',
            '=== Notes ===',
            '=== Sources ===',
            '=== Citations ===',
        ])
        p = re.compile(pattern)
        indices = [m.start() for m in p.finditer(doc)]
        min_idx = min(*indices, len(doc))
        return doc[:min_idx]


class PassageRetrieval:
    def __init__(self, nlp):
        self.tokenize = lambda text: [token.lemma_ for token in nlp(text)]
        self.bm25 = None
        self.passages = None

    @staticmethod
    def preprocess(doc):
        passages = [p for p in doc.split('\n') if doc and not doc.startswith('=')]
        return passages

    def fit(self, docs):
        passages = list(itertools.chain(*map(self.preprocess, docs)))
        corpus = [self.tokenize(p) for p in passages]
        self.bm25 = BM25(corpus)
        self.passages = passages

    def most_similar(self, question, top_n=10):
        tokens = self.tokenize(question)
        scores = self.bm25.get_scores(tokens)
        scores_indexed = [(s, i) for i, s in enumerate(scores)]
        scores_indexed.sort(reverse=True)
        passages = [self.passages[i] for _, i in scores_indexed[:top_n]]
        return passages


class AnswerExtractor:
    def __init__(self, tokenizer, model):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)

    def extract(self, question, passages):
        answers = []
        for passage in passages:
            try:
                answer = self.nlp(question=question, context=passage)
                answer['text'] = passage
                answers.append(answer)
            except KeyError:
                pass
        answers.sort(key=operator.itemgetter('score'), reverse=True)
        return answers


class TextSuggestor:
    def __init__(self, model_name="gpt2"):
        self.generator = pipeline('text-generation', model=model_name)

    def generate(self, prompt, max_length=30, num_return_sequences=5):
        return self.generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
