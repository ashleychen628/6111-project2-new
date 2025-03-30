import requests
import json

from crawl_website import download_and_clean_html
from extract_relations import ExtractRelations

# import spacy
# from SpanBERT.spanbert import SpanBERT
# from SpanBERT.spacy_help_functions import get_entities, create_entity_pairs, extract_relations

# import sys
# import os

# Add the project root (6111-project2) to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

RELATION_MAP = {
    1: "per:schools_attended",
    2: "per:employee_of",
    3: "per:cities_of_residence",
    4: "org:top_members/employees"
}

# spanbert = SpanBERT("SpanBERT/pretrained_spanbert")

import time
import google.generativeai as palm

class InfoExtraction:
    def __init__(self, model, google_api_key, google_engine_id, google_gemini_api_key, r, t, q, k):
        """Recieve the target precision and user's query. """
        self.model = model
        self.google_api_key = google_api_key
        self.google_engine_id = google_engine_id
        self.google_gemini_api_key = google_gemini_api_key

        self.t = t
        self.q = q
        self.k = k
        self.X = set()
        self.iteration = 0
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

        # self.nlp = spacy.load("en_core_web_lg") 
        # self.spanbert = SpanBERT("SpanBERT/pretrained_spanbert")
        self.entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
        self.target_relation = RELATION_MAP[self.r]
  
    def start(self):
        """Start the searching process. """
        print(f"""____
Parameters:
Client key      = {self.google_api_key}
Engine key      = {self.google_engine_id}
Gemini key      = {self.google_gemini_api_key}
Method          = {self.model}
Relation        = {self.r}
Threshold       = {self.t}
Query           = {self.q}
# of Tuples     = {self.k}

Loading necessary libraries; This should take a minute or so ...
========== Iteration: {self.iteration} - Query: {self.q} ==========
""")
        unique_tuples = set()
        
        # print(f"=========== Iteration: {self.iteration} - Query: {self.q} ===========")
        # Step 1: Get Top 10 URLs from Google Custom Search
        urls = self.google_search()
        if not urls:
            print("No results retrieved. Exiting...")
            return
        
        all_tuples = []

        for idx, result in enumerate(urls):
            if idx > 0:
                break
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
                er = ExtractRelations(self.r)
                er.extract_entities_spacy(webpage_text)
                

            if self.model == "-gemini":
                # Ensure the text is a string and not a list
                if isinstance(webpage_text, list):
                    webpage_text = ' '.join(webpage_text)
                
                webpage_tuples = self.extract_relations_gemini(webpage_text)
            
                print(f"Tuples found for this URL: {len(webpage_tuples)}")
                all_tuples.extend(webpage_tuples)

                # Add only unique tuples
                for tuple_item in webpage_tuples:
                    # Convert tuple to a hashable type (lowercase for case-insensitive comparison)
                    unique_tuple = tuple(str(item).lower() for item in tuple_item)
                    unique_tuples.add(unique_tuple)
            
            if len(all_tuples) >= self.k:
                break
        
        print("\nExtracted Tuples:")
        final_tuples = []
        for unique_tuple in unique_tuples:
            # Convert back to original representation
            original_tuple = tuple(unique_tuple)
            final_tuples.append(original_tuple)
            print(original_tuple)
            
            # Stop if we've reached the desired number of tuples
            if len(final_tuples) == self.k:
                break

        return final_tuples

    def google_search(self):
        """Query the Google API to get the top 10 result. """
        num_results=10

        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": self.google_api_key, "cx": self.google_engine_id, "q": self.q, "num": num_results}

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

    def extract_sentences(self, text):
        """Process webpage text and extract sentences using spaCy."""
        # Make sure it's a string
        if not isinstance(text, str):
            text = str(text)

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        print(f"Extracted {len(sentences)} sentences from webpage.")

        named_entities = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:  # Organizations & Locations
                named_entities.append((ent.text, ent.label_))

        print(f"Extracted {len(named_entities)} named entities.")
        return [(sentence, 
                 [ent for ent in named_entities if ent[0] in sentence]) 
                for sentence in sentences]
    
    def sentence_has_required_entities(self, entities):
        """
        Check if the sentence has the required entity types for the current relation.
        """
        req_subject, req_object = self.relation_requirements[self.relation]
        
        # Check if we have both subject and object types
        has_subject = any(ent[1] == req_subject for ent in entities)
        has_object = any(ent[1] == req_object for ent in entities)
        
        return has_subject and has_object
    
    def call_gemini_api(self, sentence):
        """
        Construct a prompt with the sentence and call the Gemini API.
        Returns a tuple (subject, relation, object, confidence) if successful, else None.
        """
        palm.configure(api_key=self.google_gemini_api_key)
        
        req_subject, req_object = self.relation_requirements[self.relation]
        prompt = (
            f"Extract the relation '{self.relation}' from the following sentence. "
            f"The subject should be of type {req_subject} and the object should be of type {req_object}. "
            f"If the relation is present, return the result in JSON format with keys 'subject', 'relation', and 'object'. "
            f"If the relation is not present or there is not enough information, return an empty JSON object. "
            f"Sentence: \"{sentence}\""
        )
        
        # To avoid overloading the API, sleep for a short period
        time.sleep(1)
        
        try:
            model = palm.GenerativeModel('models/gemini-2.0-flash')
            response = model.generate_content(prompt)
            
            # Try to parse the response
            try:
                text = response.text.strip('```json\n').strip('```')
                result = json.loads(text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to manually extract
                print(f"JSON parsing failed. Raw response: {response.text}")
                return None

            # Validate the result
            if result and "subject" in result and "object" in result:
                return (result["subject"], self.relation, result["object"], 1.0)
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
        
        return None

    def extract_relations_gemini(self, text):
        """
        Process the webpage text:
         1. Split the text into sentences and get entities per sentence.
         2. For each sentence that contains the required entities, call Gemini to extract the relation.
         3. Return a list of extracted tuples.
        """
        if not isinstance(text, str):
            text = ' '.join(text)
        
        sentences_with_entities = self.extract_sentences(text)
        extracted_tuples = []
        
        print(f"Processing {len(sentences_with_entities)} sentences with entities")
        
        for sentence, entities in sentences_with_entities:
            # Ensure sentence is a string
            if isinstance(sentence, list):
                sentence = ' '.join(sentence)
            
            # print(f"\nProcessing sentence: {sentence}")
            # print(f"Entities in sentence: {entities}")
            
            # Skip sentences that don't have the required entity types
            if not self.sentence_has_required_entities(entities):
                print("Sentence does not have required entities. Skipping.")
                continue
            
            # Call Gemini on the sentence
            relation_tuple = self.call_gemini_api(sentence)
            
            if relation_tuple is not None:
                print(f"Extracted tuple: {relation_tuple}")
                
                # Avoid duplicate tuples
                if relation_tuple not in extracted_tuples:
                    extracted_tuples.append(relation_tuple)
                
                # Stop if we've reached the desired number of tuples
                if len(extracted_tuples) >= self.k:
                    break
        
        print(f"Total extracted tuples: {len(extracted_tuples)}")
        return extracted_tuples
