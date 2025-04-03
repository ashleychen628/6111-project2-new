import spacy
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs, extract_relations

spanbert = SpanBERT("./pretrained_spanbert")
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
        self.relation_requirements = {
            "Schools_Attended": ("PERSON", "ORG"),
            "Work_For": ("PERSON", "ORG"),
            "Live_In": ("PERSON", "GPE"),
            "Top_Member_Employees": ("PERSON", "ORG")
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

            ents = get_entities(sentence, self.entities_of_interest)
            
            # create entity pairs
            sentence_entity_pairs = create_entity_pairs(sentence, self.entities_of_interest)

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
            #     relation_predictions = spanbert.predict(self.candidate_pairs)  # get predictions: list of (relation, confidence) pairs

            # relation_mapping = {
            #     1: ("per:schools_attended", "PERSON", "ORGANIZATION"),
            #     2: ("per:employee_of", "PERSON", "ORGANIZATION"),
            #     3: ("per:cities_of_residence", "PERSON", ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]),
            #     4: ("org:top_members/employees", "ORGANIZATION", "PERSON")
            # }

            # expected_label, expected_subj_type, expected_obj_type = relation_mapping[self.relation]

            # for ex, pred in zip(self.candidate_pairs, relation_predictions):
            #     relation_label, confidence = pred
            #     subj, obj = ex['subj'], ex['obj']
            #     tokens = ex['tokens']
            #     key = (subj[0], obj[0])
                
            #     if relation_label == expected_label and \
            #     (subj[1] == expected_subj_type and (obj[1] in expected_obj_type if isinstance(expected_obj_type, list) else obj[1] == expected_obj_type)):
            #         if key not in self.relation_map or confidence > self.relation_map[key][0]:
            #             self.relation_map[key] = (confidence, relation_label, tokens)
            #         else:
            #             continue
                    
            #         print("\n\t\t=== Extracted Relation ===")
            #         print(f"the relation_label is: {relation_label}")
            #         print(f"\t\tInput tokens: {tokens}")
            #         print(f"\t\tOutput Confidence: {confidence:.7f} ; Subject: {subj[0]} ; Object: {obj[0]} ;")
            #         self.relation_map[key] = (confidence, relation_label, tokens)
            #         if confidence >= self.threshold:
            #                 print("\t\tAdding to set of extracted relations")
            #                 print(relation_label)
            #                 print("\t\t==========")
                            
            #                 token_tuple = tuple(tokens)
            #                 if token_tuple not in self.seen_token_spans:
            #                     self.seen_token_spans.add(token_tuple)
            #                     extracted_annotations += 1
            #                 query_key = f"{subj} {obj}"
            #                 self.chosen_tuples.append({
            #                     "subject": subj[0],
            #                     "object": obj[0],
            #                     "confidence": confidence,
            #                     "key": key
            #                 })
            #         else:
            #             print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
            #             print("\t\t==========")

            # self.candidate_pairs = []

        print(f"\n\tExtracted annotations for  {extracted_annotations}  out of total  {len(sentences)}  sentences")
        print(f"\n\tRelations extracted from this website: {len(self.chosen_tuples)} (Overall: {len(self.relation_map)})")
        return self.chosen_tuples
    

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