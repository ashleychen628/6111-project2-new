import spacy
import time
import json
from spacy_help_functions import get_entities, create_entity_pairs, extract_relations
import google.generativeai as palm

nlp = spacy.load("en_core_web_lg") 

class ExtractRelationsGemini:
    def __init__(self, r, t, google_gemini_api_key):
        self.relation = r
        self.threshold = t
        self.google_gemini_api_key = google_gemini_api_key
        self.candidate_pairs = []
        self.chosen_tuples = []
        self.relation_map = {}
        self.seen_token_spans = set()

        relation_map = {
            1: "Schools_Attended",
            2: "Work_For",
            3: "Live_In",
            4: "Top_Member_Employees"
        }
        self.relation_name = relation_map[r]

        self.relation_requirements = {
            "Schools_Attended": ("PERSON", "ORGANIZATION"),
            "Work_For": ("PERSON", "ORGANIZATION"),
            "Live_In": ("PERSON", ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]),
            "Top_Member_Employees": ("PERSON", "ORGANIZATION")
        }
        self.entities_of_interest = {
          1: ["PERSON", "ORGANIZATION"], # Schools_Attended
          2: ["PERSON", "ORGANIZATION"], # Work_For 
          3: ["PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"], # Live_In
          4: ["ORGANIZATION", "PERSON"] # Top_Member_Employees
        }
                  
    
    def extract_relations_gemini(self, raw_text):
        """Process webpage text and extract sentences using spaCy."""
        doc = nlp(raw_text)

        print("Annotating the webpage using spacy...")

        sentences = list(doc.sents)
        extracted_annotations = 0

        print(f"Extracted {len(list(doc.sents))} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
        
        for idx, sentence in enumerate(sentences):
            if (idx + 1) % 5 == 0:
                print(f"\n\tProcessed {idx + 1} / {len(sentences)} sentences")

            ents = get_entities(sentence, self.entities_of_interest[self.relation])
            
            # create entity pairs
            sentence_entity_pairs = create_entity_pairs(sentence, self.entities_of_interest[self.relation])

            for ep in sentence_entity_pairs:
                subj, obj = ep[1], ep[2]

                if self.relation == 1 and subj[1] == "PERSON" and obj[1] == "ORGANIZATION":
                    # Schools_Attended
                    self.candidate_pairs.append({"tokens": ep[0], "subj": subj, "obj": obj})
                
                elif self.relation == 2 and subj[1] == "PERSON" and obj[1] == "ORGANIZATION":
                    # Work_For
                    self.candidate_pairs.append({"tokens": ep[0], "subj": subj, "obj": obj})
                
                elif self.relation == 3 and subj[1] == "PERSON" and obj[1] in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]:
                    # Live_In
                    self.candidate_pairs.append({"tokens": ep[0], "subj": subj, "obj": obj})
                
                elif self.relation == 4 and subj[1] == "ORGANIZATION" and obj[1] == "PERSON":
                    # Top_Member_Employees
                    self.candidate_pairs.append({"tokens": ep[0], "subj": subj, "obj": obj})
            
            if len(self.candidate_pairs) == 0:
                continue
            else:
                relation_tuple = self.call_gemini_api(sentence)

        print(f"\n\tExtracted annotations for  {extracted_annotations}  out of total  {len(sentences)}  sentences")
        print(f"\n\tRelations extracted from this website: {len(self.chosen_tuples)} (Overall: {len(self.relation_map)})")
        return self.chosen_tuples
    

    def call_gemini_api(self, sentence):
        """
        Construct a prompt with the sentence and call the Gemini API.
        Returns a tuple (subject, relation, object, confidence) if successful, else None.
        """
        palm.configure(api_key=self.google_gemini_api_key)
        
        req_subject, req_object = self.relation_requirements[self.relation_name]

        # If the object is a list, join them into a string
        if isinstance(req_object, list):
            req_object_str = " or ".join(req_object)
        else:
            req_object_str = req_object

        prompt = (
            f"Extract the relation '{self.relation_name}' from the following sentence. "
            f"The subject should be of type {req_subject} and the object should be of type {req_object}. "
            f"If the relation is present, return the result in JSON format with keys 'subject', 'relation', and 'object'. "
            f"If the relation is not present or there is not enough information, return an empty JSON object. "
            f"Sentence: \"{sentence}\""
        )

        # To avoid overloading the API, sleep for a short period
        time.sleep(5)
        
        try:
            model = palm.GenerativeModel('models/gemini-2.0-flash')
            generation_config = palm.types.GenerationConfig(
                max_output_tokens=100,
                temperature=0.2,
                top_p=1,
                top_k=32
            )
            response = model.generate_content(prompt, generation_config=generation_config)
            
            # Try to parse the response
            try:
                # text = response.text.strip('```json\n').strip('```')
                json_match = re.search(r"\{.*?\}", response.text, re.DOTALL)
                # result = json.loads(text)
                result = json.loads(json_match.group())
                # print(response.text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to manually extract
                # print(f"JSON parsing failed. Raw response: {response.text}")
                return None

            # Validate the result
            if result and "subject" in result and "object" in result:
                print(f"[✓] Extracted: {result['subject']} — {self.relation} — {result['object']}")
                return (result["subject"], self.relation, result["object"], 1.0)
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # If 429 is in error, apply retry delay
            if "429" in str(e):
                retry_match = re.search(r'retry_delay \{\s*seconds: (\d+)', str(e))
                if retry_match:
                    delay = int(retry_match.group(1))
                    print(f"Sleeping for {delay} seconds due to rate limit...")
                    time.sleep(delay)
        
        return None