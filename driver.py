import requests
import json

from crawl_website import download_and_clean_html
from extract_relations_spanbert import ExtractRelationsSpanbert
from extract_relations_gemini import ExtractRelationsGemini

RELATION_MAP = {
    1: "per:schools_attended",
    2: "per:employee_of",
    3: "per:cities_of_residence",
    4: "org:top_members/employees"
}

class InfoExtraction:
    def __init__(self, model, google_api_key, google_engine_id, google_gemini_api_key, r, t, q, k):
        """Recieve the target precision and user's query. """
        self.model = model
        self.google_api_key = google_api_key
        self.google_engine_id = google_engine_id
        self.google_gemini_api_key = google_gemini_api_key

        self.threshold = t 
        self.query = q 
        self.tuple_num = k
        self.r = r 
        self.X = set()
        self.iteration = 0
        self.used_queries = set()
        self.seen_keys = set()

        self.chosen_tuples = {} # {key: <subj, obj>, value: {subj: "", obj:"", confidence:""}}

  
    def start(self):
        """Start the searching process. """
        # keep iteratign until number of tuples reached k
        print(f"""____
Parameters:
Client key      = {self.google_api_key}
Engine key      = {self.google_engine_id}
Gemini key      = {self.google_gemini_api_key}
Method          = {self.model}
Relation        = {self.r}
Threshold       = {self.threshold}
Query           = {self.query}
# of Tuples     = {self.tuple_num}
Loading necessary libraries; This should take a minute or so ...)
""")
        while len(self.chosen_tuples) < self.tuple_num:
            if self.iteration > 0:
                self.update_query()

            print(f"""
========== Iteration: {self.iteration} - Query: {self.query} ==========
""")
            unique_tuples = set()
            
            # Step 1: Get Top 10 URLs from Google Custom Search
            urls = self.google_search()
            if not urls:
                print("No results retrieved. Exiting...")
                return
            
            all_tuples = []

            for idx, result in enumerate(urls):
                # if idx > 0:
                #     break
                url = result["url"]
                print(f"\nURL ({idx+1} / {len(urls)}): {url}")

                # Step 2: Fetch and clean webpage content
                webpage_text = download_and_clean_html(url)
                if not webpage_text:
                    print("Unable to fetch URL. Skipping...")
                    continue

                if len(webpage_text) > 10000:
                    webpage_text = webpage_text[:10000]
                    print(f"Truncated to 10,000 characters")
                else:
                    print(f"Webpage length (num characters): {len(webpage_text)}")

                if self.model == "-spanbert":
                    er = ExtractRelationsSpanbert(self.r, self.threshold, self.seen_keys, self.chosen_tuples)
                    chosen_tuples, seen_keys = er.extract_relations_spanbert(webpage_text)
                    self.chosen_tuples = chosen_tuples
                    self.seen_keys.update(seen_keys)
                
                if self.model == "-gemini":
                    er_gemini = ExtractRelationsGemini(self.r, self.threshold, self.google_gemini_api_key, self.seen_keys, self.chosen_tuples)
                    chosen_tuples, seen_keys = er_gemini.extract_relations_gemini(webpage_text)
                    self.chosen_tuples = chosen_tuples
                    self.seen_keys.update(seen_keys)

            self.print_final_output()

            self.iteration += 1

    def google_search(self):
        """Query the Google API to get the top 10 result. """
        num_results=10

        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": self.google_api_key, "cx": self.google_engine_id, "q": self.query, "num": num_results}

        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            results = response.json()
            search_results = []
            
            idx = 0
            for item in results.get("items", []):
                url = item.get("link", "")
                search_results.append({
                    "url": item.get("link", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", "")
                })

                idx = idx + 1
                    
            return search_results
        else:
            print("API Error:", response.status_code, response.text)
            return None


    def print_final_output(self):
        """Print the final output, confidence in descending order. """
        print(f"\n{'='*18} ALL RELATIONS for {RELATION_MAP[self.r]} ( {len(self.chosen_tuples)} ) {'='*18}")
        
        if self.model == "-spanbert":
            # Sort by confidence descending
            for tup in sorted(self.chosen_tuples.values(), key=lambda x: x["confidence"], reverse=True):
                print(f"Confidence: {tup['confidence']:.7f} \t| Subject: {tup['subject']} \t| Object: {tup['object']}")
                
        if self.model == "-gemini":
            for tup in self.chosen_tuples.values():
                print(f"Subject: {tup['subject']} \t| Object: {tup['object']}")
        
        if len(self.chosen_tuples) >= self.tuple_num:
            print(f"Total # of iterations = {self.iteration + 1}")

    def update_query(self):
        """Choose new queries from chosen tuple to append old query. """

        if self.model == "-spanbert":
            # Pick the highest-confidence unused tuple
            best_key, best_tuple = max(
                (
                    (key, val)
                    for key, val in self.chosen_tuples.items()
                    if key not in self.used_queries
                ),
                key=lambda x: x[1]["confidence"],
                default=(None, None)
            )
        else:
            # For Gemini, pick any unused tuple
            best_key, best_tuple = next(
                (
                    (key, val)
                    for key, val in self.chosen_tuples.items()
                    if key not in self.used_queries
                ),
                (None, None)
            )

        if best_tuple:
            self.used_queries.add(best_key)
            self.query = " ".join(best_key)  # best_key is (subj, obj)
        else:
            print("No new unused tuples available. Stopping.")
            exit()

