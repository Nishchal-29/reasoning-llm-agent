# ----------------- Test snippet (KEEP COMMENTED) -----------------
# from duckduckgo_search import DDGS
# import requests
# from bs4 import BeautifulSoup
# import wikipediaapi as wiki
# import spacy
# import nltk
# from nltk.corpus import stopwords
#
# results = DDGS().text("Recipe for making Coffee", max_results=1)
# links = [r["href"] for r in results]
# print("Links found:")
# for link in links:
#     print(link)
#
# for url in links:
#     try:
#         print("\nScraping:", url)
#         res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
#         soup = BeautifulSoup(res.text, "html.parser")
#         paragraphs = [p.get_text() for p in soup.find_all("p")]
#         clean_text = " ".join(paragraphs)
#         print(clean_text[:500], "...\n")
#     except Exception as e:
#         print("Failed to scrape:", url, "Error:", e)


# ----------------- Working Class Version -----------------
# from duckduckgo_search import DDGS
# import requests
# from bs4 import BeautifulSoup
# import wikipediaapi as wiki
# import spacy
# import nltk
# from nltk.corpus import stopwords
# import re

# # ----------------- Setup NLP -----------------
# # nltk.download("stopwords")  # Run once
# STOPWORDS = set(stopwords.words("english"))
# nlp = spacy.load("en_core_web_sm")


# class WebSearchExecutor:
#     def __init__(self):
#         self.ddg = DDGS()
#         self.wiki = wiki.Wikipedia(
#             user_agent="SIH-ThreatClassification/1.0 (contact: teamemail@example.com)",
#             language="en"
#         )

#     # ----------------- Text Processing -----------------
#     def clean_text(self, text: str) -> str:
#         """Normalize and lemmatize text without removing punctuation."""
#         text = text.lower()
#         doc = nlp(text)
#         tokens = [token.lemma_ for token in doc if token.text not in STOPWORDS and not token.is_space]
#         return " ".join(tokens)

#     def extract_entities(self, text: str) -> list:
#         """Extract named entities using SpaCy."""
#         doc = nlp(text)
#         return [(ent.text, ent.label_) for ent in doc.ents]

#     def text_to_readable_paragraph(self, text: str) -> str:
#         """Convert raw text into readable sentences with only meaningful inline entities."""
#         # Remove bullets and excessive whitespace
#         text = re.sub(r"[\u2022•]", "", text)
#         text = re.sub(r"\s+", " ", text).strip()

#         doc = nlp(text)
#         sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]

#         readable_sentences = []
#         meaningful_labels = {"PERSON", "ORG", "GPE", "DATE", "NORP", "LOC", "EVENT", "TIME"}

#         for sent in sentences:
#             sent_doc = nlp(sent)
#             highlighted = sent
#             for ent in sent_doc.ents:
#                 if ent.label_ in meaningful_labels:
#                     highlighted = highlighted.replace(ent.text, f"{ent.text} ({ent.label_})")
#             readable_sentences.append(highlighted)

#         return " ".join(readable_sentences)

#     # ----------------- Web Search -----------------
#     def search_duckduckgo(self, query: str, max_results: int = 5) -> list:
#         results = []
#         try:
#             for r in self.ddg.text(query, max_results=max_results):
#                 if "href" in r:
#                     results.append({
#                         "title": r.get("title", ""),
#                         "snippet": r.get("body", ""),
#                         "url": r["href"]
#                     })
#         except Exception as e:
#             results.append({"error": str(e)})
#         return results

#     # ----------------- Wikipedia -----------------
#     def search_wikipedia(self, query: str) -> list:
#         results = []
#         try:
#             page = self.wiki.page(query)
#             if page.exists():
#                 results.append({
#                     "title": page.title,
#                     "summary": page.summary
#                 })
#             else:
#                 results.append({"error": "Page not found"})
#         except Exception as e:
#             results.append({"error": str(e)})
#         return results

#     # ----------------- Page Scraper -----------------
#     def scrape_page(self, url: str) -> str | None:
#         try:
#             res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
#             soup = BeautifulSoup(res.text, "html.parser")
#             return " ".join([p.get_text() for p in soup.find_all("p")])
#         except Exception:
#             return None

#     # ----------------- Query Classification -----------------
#     def classify_query(self, query: str) -> str:
#         """Classify query to Wikipedia or general web search."""
#         wiki_keywords = ["who", "what", "when", "where", "definition", "history"]
#         if any(k in query.lower() for k in wiki_keywords):
#             return "wikipedia"
#         return "websearch"

#     # ----------------- Execute Search Pipeline (JSON) -----------------
#     def execute(self, query: str) -> dict:
#         classification = self.classify_query(query)
#         response = {"query": query, "classification": classification, "results": []}

#         if classification == "wikipedia":
#             wiki_results = self.search_wikipedia(query)
#             for r in wiki_results:
#                 if "summary" in r:
#                     r["cleaned_text"] = self.clean_text(r["summary"])
#                     r["entities"] = self.extract_entities(r["summary"])
#                 response["results"].append(r)

#         else:  # websearch
#             ddg_results = self.search_duckduckgo(query)
#             if not ddg_results:
#                 wiki_results = self.search_wikipedia(query)
#                 for r in wiki_results:
#                     if "summary" in r:
#                         r["cleaned_text"] = self.clean_text(r["summary"])
#                         r["entities"] = self.extract_entities(r["summary"])
#                     response["results"].append(r)
#                 if not response["results"]:
#                     response["results"].append({"error": "No results found"})
#                 return response

#             for r in ddg_results:
#                 if "url" in r:
#                     page_text = self.scrape_page(r["url"])
#                     if page_text:
#                         r["scraped_text_preview"] = page_text[:500] + "..."
#                         r["cleaned_text"] = self.clean_text(page_text)
#                         r["entities"] = self.extract_entities(page_text)
#                 response["results"].append(r)

#         return response

#     # ----------------- Execute Search Pipeline (Text) -----------------
#     def execute_as_text(self, query: str) -> str:
#         """Return the search result as a readable paragraph instead of JSON."""
#         classification = self.classify_query(query)
#         response_texts = [f"Query: {query}."]

#         if classification == "wikipedia":
#             wiki_results = self.search_wikipedia(query)
#             for r in wiki_results:
#                 if "summary" in r:
#                     readable = self.text_to_readable_paragraph(r["summary"])
#                     response_texts.append(f"Wikipedia says: {readable}")

#         else:  # websearch
#             ddg_results = self.search_duckduckgo(query)
#             if not ddg_results:
#                 wiki_results = self.search_wikipedia(query)
#                 for r in wiki_results:
#                     if "summary" in r:
#                         readable = self.text_to_readable_paragraph(r["summary"])
#                         response_texts.append(f"Wikipedia says: {readable}")
#             else:
#                 for r in ddg_results:
#                     if r.get("title"):
#                         response_texts.append(f"Title: {r['title']}")
#                     if r.get("snippet"):
#                         response_texts.append(f"Snippet: {r['snippet']}")
#                     if "url" in r:
#                         page_text = self.scrape_page(r["url"])
#                         if page_text:
#                             readable = self.text_to_readable_paragraph(page_text[:2000])
#                             response_texts.append(f"Page preview: {readable}")

#         return "\n\n".join(response_texts)


# # ----------------- Example Usage -----------------
# if __name__ == "__main__":
#     web = WebSearchExecutor()
#     query = "coffee recipe"

#     # JSON style output
#     result_json = web.execute(query)
#     import json
#     print("=== JSON Output ===")
#     print(json.dumps(result_json, indent=4))

#     # ChatGPT style text output
#     result_text = web.execute_as_text(query)
#     print("\n=== Text Output ===")
#     print(result_text)


from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import wikipediaapi

class WebSearchExecutorSimple:
    def __init__(self):
        self.ddg = DDGS()
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="SIH-ThreatClassification/1.0 (contact: teamemail@example.com)",
            language="en"
        )

    # ----------------- Page Scraper -----------------
    def scrape_page(self, url: str) -> str | None:
        try:
            res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(res.text, "html.parser")
            paragraphs = [p.get_text() for p in soup.find_all("p")]
            text = " ".join(paragraphs).strip()
            return text if text else None
        except Exception:
            return None

    # ----------------- DuckDuckGo Search -----------------
    def search_duckduckgo(self, query: str, max_results: int = 5) -> list:
        results = []
        try:
            for r in self.ddg.text(query, max_results=max_results):
                if "href" in r:
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r["href"]
                    })
        except Exception as e:
            results.append({"error": str(e)})
        return results

    # ----------------- Wikipedia Search -----------------
    def search_wikipedia(self, query: str) -> list:
        results = []
        page = self.wiki.page(query)
        if page.exists():
            results.append({
                "title": page.title,
                "text": page.summary[:1000],  # Limit summary length
                "url": f"https://en.wikipedia.org/wiki/{page.title.replace(' ', '_')}"
            })
        return results

    # ----------------- Execute Search Pipeline -----------------
    def execute(self, query: str) -> dict:
        response = {"query": query, "results": []}

        # Wikipedia results first
        wiki_results = self.search_wikipedia(query)
        response["results"].extend(wiki_results)

        # DuckDuckGo results
        ddg_results = self.search_duckduckgo(query)
        for r in ddg_results:
            result_item = {
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "url": r.get("url", "")
            }
            if r.get("url"):
                page_text = self.scrape_page(r["url"])
                if page_text:
                    result_item["text"] = page_text[:1000]  # Limit preview
            response["results"].append(result_item)

        if not response["results"]:
            response["results"].append({"error": "No results found"})

        return response


# ----------------- Example Usage -----------------
if __name__ == "__main__":
    web = WebSearchExecutorSimple()
    query = "coffee recipe"

    result_json = web.execute(query)

    import json
    print(json.dumps(result_json, indent=4))
