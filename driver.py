import requests
import json

from crawl_website import download_and_clean_html
from extract_relations_spanbert import ExtractRelationsSpanbert
from extract_relations_gemini import ExtractRelationsGemini

import spacy
import time
import google.generativeai as palm

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
        self.X = set()
        self.iteration = 0
        self.used_queries = set()

        self.chosen_tuples = []
        relation_map = {
            1: "Schools_Attended",
            2: "Work_For",
            3: "Live_In",
            4: "Top_Member_Employees"
        }

        if r not in relation_map:
            raise ValueError("r must be an integer between 1 and 4: "
                            "1 for Schools_Attended, 2 for Work_For, 3 for Live_In, and 4 for Top_Member_Employees.")
        
        self.r = r 
        self.relation = relation_map[r]

        self.relation_requirements = {
            "Schools_Attended": ("PERSON", "ORG"),
            "Work_For": ("PERSON", "ORG"),
            "Live_In": ("PERSON", "GPE"),
            "Top_Member_Employees": ("PERSON", "ORG")
        }

        self.nlp = spacy.load("en_core_web_lg") 
        # self.spanbert = SpanBERT("SpanBERT/pretrained_spanbert")
        self.entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
        self.target_relation = RELATION_MAP[self.r]
        
  
    def start(self):
        """Start the searching process. """
        # keep iteratign until number of tuples reached k
        while len(self.chosen_tuples) < self.tuple_num:
            if self.iteration > 0:
                self.update_query()
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

    Loading necessary libraries; This should take a minute or so ...
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
                # print(webpage_text)
                if self.model == "-spanbert":
                    # self.use_spanbert()
                    er = ExtractRelationsSpanbert(self.r, self.threshold)
                    self.chosen_tuples += er.extract_relations_spanbert(webpage_text)
                
                if self.model == "-gemini":
                    # Ensure the text is a string and not a list
                    er_gemini = ExtractRelationsGemini(self.r, self.threshold, self.google_gemini_api_key)
                    self.chosen_tuples += er_gemini.extract_relations_gemini(webpage_text)

                    # if isinstance(webpage_text, list):
                    #     webpage_text = ' '.join(webpage_text)
                    
                    # webpage_tuples = self.extract_relations_gemini(webpage_text)
                
                    # print(f"Tuples found for this URL: {len(webpage_tuples)}")
                    # all_tuples.extend(webpage_tuples)

                    # # Add only unique tuples
                    # for tuple_item in webpage_tuples:
                    #     # Convert tuple to a hashable type (lowercase for case-insensitive comparison)
                    #     unique_tuple = tuple(str(item).lower() for item in tuple_item)
                    #     unique_tuples.add(unique_tuple)
                
            if len(self.chosen_tuples) > self.tuple_num:
                self.print_final_output()

            self.iteration += 1


            
        #     if len(all_tuples) >= self.tuple_num:
        #         break
        
        # print("\nExtracted Tuples:")
        # final_tuples = []
        # for unique_tuple in unique_tuples:
        #     # Convert back to original representation
        #     original_tuple = tuple(unique_tuple)
        #     final_tuples.append(original_tuple)
        #     print(original_tuple)
            
        #     # Stop if we've reached the desired number of tuples
        #     if len(final_tuples) == self.tuple_num:
        #         break

        # return final_tuples

    def google_search(self):
        """Query the Google API to get the top 10 result. """
        num_results=10

        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": self.google_api_key, "cx": self.google_engine_id, "q": self.query, "num": num_results}

        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            results = response.json()
            search_results = []

            print("Google Search Results:\n======================")
            
            idx = 0
            for item in results.get("items", []):
                url = item.get("link", "")
                search_results.append({
                    "url": item.get("link", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", "")
                })

                idx = idx + 1
                    
            print("======================")
            return search_results
        else:
            print("API Error:", response.status_code, response.text)
            return None


    def print_final_output(self):
        """Print the final output, confidence in descending order. """
        print(f"\n{'='*18} ALL RELATIONS for {RELATION_MAP[self.r]} ( {len(self.chosen_tuples)} ) {'='*18}")
        
        if self.model == "-spanbert":
            # Sort by confidence descending
            for tup in sorted(self.chosen_tuples, key=lambda x: x["confidence"], reverse=True):
                print(f"Confidence: {tup['confidence']:.7f} \t| Subject: {tup['subject']} \t| Object: {tup['object']}")
                
        if self.model == "-gemini":
            for tup in self.chosen_tuples:
                print(f"Subject: {tup['subject']} \t| Object: {tup['object']}")

        print(f"Total # of iterations = {self.iteration + 1}")

    def update_query(self):
        """Choose new queries from chosen tuple to append old query. """

        if self.model == "-spanbert":
            # Pick the highest-confidence unused tuple
            best_tuple = max(
                (tup for tup in self.chosen_tuples if tup["key"] not in self.used_queries),
                key=lambda x: x["confidence"],
                default=None
            )
        else:
            # For Gemini, pick any unused tuple
            best_tuple = next(
                (tup for tup in self.chosen_tuples if tup["key"] not in self.used_queries),
                None
            )

        if best_tuple:
            self.used_queries.add(best_tuple["key"])
            self.query = best_tuple["key"]
        else:
            print("No new unused tuples available. Stopping.")
            exit()

