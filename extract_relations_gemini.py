import spacy
import re
import time
import json
from spacy_help_functions import get_entities, create_entity_pairs, extract_relations
import google.generativeai as palm

nlp = spacy.load("en_core_web_lg") 

EXAMPLES = {
    "Schools_Attended": (
        "Sentence: \"Jeff Bezos graduated from Princeton University.\" → "
        "{\"subject\": \"Jeff Bezos\", \"relation\": \"Schools_Attended\", \"object\": \"Princeton University\"}"
    ),
    "Work_For": (
        "Sentence: \"Alec Radford works at OpenAI.\" → "
        "{\"subject\": \"Alec Radford\", \"relation\": \"Work_For\", \"object\": \"OpenAI\"}"
    ),
    "Live_In": (
        "Sentence: \"Mariah Carey lives in New York City.\" → "
        "{\"subject\": \"Mariah Carey\", \"relation\": \"Live_In\", \"object\": \"New York City\"}"
    ),
    "Top_Member_Employees": (
        "Sentence: \"Jensen Huang is the CEO of Nvidia.\" → "
        "{\"subject\": \"Nvidia\", \"relation\": \"Top_Member_Employees\", \"object\": \"Jensen Huang\"}"
    )
}

RELATION_SPECS = {
        "Schools_Attended": {
            "subject_type": "a person\'s actual name, not a title, role, or pronoun (e.g., not \"scientist\", \"he\", \"entrepreneur\").",
            "object_type": "an educational organization (e.g., university, college, school).",
            "object_constraints": "NOT be vague, relative, or deictic (e.g., not 'here', 'there', 'this place').",
            "relation_description": "indicates that the person attended school at the organization"
        },
        "Work_For": {
            "subject_type": "a person\'s actual name, not a title, role, or pronoun (e.g., not \"scientist\", \"he\", \"entrepreneur\").",
            "object_type": "an organization (e.g., 'Google', 'microsoft')",
            "object_constraints": "be a formal organization name, not vague terms like 'company' or 'here'.",
            "relation_description": "indicates that the person worked for the organization"
        },
        "Live_In": {
            "subject_type": "a person\'s actual name, not a title, role, or pronoun (e.g., not \"scientist\", \"he\", \"entrepreneur\").",
            "object_type": "a named location such as a city, state, province, or country",
            "object_constraints": "be a valid geographic location, not vague terms like 'here' or 'there'.",
            "relation_description": "indicates that the person resides in the location"
        },
        "Top_Member_Employees": {
            "subject_type": "an organization's name (e.g., 'Google', 'microsoft')",
            "object_type": "a person\'s actual name, not a title, role, or pronoun (e.g., not \"scientist\", \"he\", \"entrepreneur\").",
            "object_constraints": "be a person's full name, not a title or role (e.g., not 'CEO').",
            "relation_description": "indicates that the person is a top employee or executive of the organization"
        }
}

RELATION_DESCRIPTIONS = {
    "Schools_Attended": (
        "if the sentence describes a PERSON who attended, studied at, graduated from, or was educated at an ORGANIZATION (e.g., a school or university). "
        "The subject must be the actual name of the person (avoid pronouns), and the object must be the actual name of the school or university."
    ),
    "Work_For": (
        "if the sentence describes a PERSON who works or worked at an ORGANIZATION. "
        "Use the actual name of the person and the company or organization mentioned."
    ),
    "Live_In": (
        "if the sentence describes a PERSON who lives, lived, resided, or moved to a LOCATION, CITY, STATE_OR_PROVINCE, or COUNTRY. "
        "The subject must be the actual name of the person, and the object must be the actual place name."
    ),
    "Top_Member_Employees": (
        "if the sentence describes a PERSON who is a top member of an ORGANIZATION (e.g., CEO, founder, executive). "
        "The subject must be the name of the organization and the object must be the name of the person (top member)."
    )
}

class ExtractRelationsGemini:
    def __init__(self, r, t, google_gemini_api_key, seen_keys, chosen_tuples):
        self.relation = r
        self.threshold = t
        self.google_gemini_api_key = google_gemini_api_key
        self.candidate_pairs = []
        self.chosen_tuples = chosen_tuples
        self.relation_map = {}
        self.seen_sentence = set()
        self.seen_keys = seen_keys
        self.possible_tuples_num = 0
        self.duplicate = 0

        
        relation_map = {
            1: "Schools_Attended",
            2: "Work_For",
            3: "Live_In",
            4: "Top_Member_Employees"
        }
        self.relation_name = relation_map[r]

        relation_string = {
            1: "attended school(s) at",
            2: "work for",
            3: "live in",
            4: "is (a) senior employee at"
        }
        self.relation_str = relation_string[r]

        self.relation_requirements = {
            "Schools_Attended": ("person", "organization"),
            "Work_For": ("person", "organization"),
            "Live_In": ("person", ["location", "city", "state or province", "country"]),
            "Top_Member_Employees": ("person", "organization")
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
            self.candidate_pairs = []
            if (idx + 1) % 5 == 0:
                print(f"\n\tProcessed {idx + 1} / {len(sentences)} sentences")
            
            ents = get_entities(sentence, self.entities_of_interest[self.relation])
            
            # create entity pairs
            sentence_entity_pairs = create_entity_pairs(sentence, self.entities_of_interest[self.relation])
        
            for ep in sentence_entity_pairs:
                tokens, subj, obj = ep[0], ep[1], ep[2]
                if not tokens or not subj[0].strip() or not obj[0].strip():
                    continue

                if self.relation == 1:  # Schools_Attended
                    if (subj[1], obj[1]) == ("PERSON", "ORGANIZATION"):
                        self.candidate_pairs.append({"tokens": tokens, "subj": subj, "obj": obj})
                    elif (obj[1], subj[1]) == ("PERSON", "ORGANIZATION"):
                        self.candidate_pairs.append({"tokens": tokens, "subj": obj, "obj": subj})

                elif self.relation == 2:  # Work_For
                    if (subj[1], obj[1]) == ("PERSON", "ORGANIZATION"):
                        self.candidate_pairs.append({"tokens": tokens, "subj": subj, "obj": obj})
                    elif (obj[1], subj[1]) == ("PERSON", "ORGANIZATION"):
                        self.candidate_pairs.append({"tokens": tokens, "subj": obj, "obj": subj})

                elif self.relation == 3:  # Live_In
                    valid_locs = {"LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"}
                    if subj[1] == "PERSON" and obj[1] in valid_locs:
                        self.candidate_pairs.append({"tokens": tokens, "subj": subj, "obj": obj})
                    elif obj[1] == "PERSON" and subj[1] in valid_locs:
                        self.candidate_pairs.append({"tokens": tokens, "subj": obj, "obj": subj})

                elif self.relation == 4:  # Top_Member_Employees
                    if (subj[1], obj[1]) == ("ORGANIZATION", "PERSON"):
                        self.candidate_pairs.append({"tokens": tokens, "subj": subj, "obj": obj})
                    elif (obj[1], subj[1]) == ("ORGANIZATION", "PERSON"):
                        self.candidate_pairs.append({"tokens": tokens, "subj": obj, "obj": subj})
            
            if len(self.candidate_pairs) == 0:
                continue
            else:
                for ex in self.candidate_pairs:
                    subj, obj = ex['subj'], ex['obj']
                    tokens = ex['tokens']
                    relation_answer = self.call_gemini_api(sentence, subj, obj, tokens)

                    if relation_answer and 'subject' in relation_answer and 'object' in relation_answer:
                        subj = relation_answer['subject']
                        obj = relation_answer['object']
                        key = (subj, obj)
                        self.possible_tuples_num += 1
                        print("\n\t\t=== Extracted Relation ===")
                        print(f"\t\tSentence: {sentence}")
                        print(f"\t\tSubject: {subj} ; Object: {obj} ;")

                        if key not in self.seen_keys:
                            if sentence not in self.seen_sentence:
                                self.seen_sentence.add(sentence)
                                extracted_annotations += 1

                            self.chosen_tuples[key] = {
                                "subject": subj,
                                "object": obj,
                                "confidence": 1,
                                "key": key
                            }
                            self.seen_keys.add(key)
                            print(f"\t\tAdding to set of extracted relations")
                        else:
                            self.duplicate += 1
                            print(f"\t\tDuplicate. Ignoring this.")

                        print("\t\t==========")

        print(f"\tExtracted annotations for  {extracted_annotations}  out of total  {len(sentences)}  sentences \n")
        print(f"\tRelations extracted from this website: {(self.possible_tuples_num - self.duplicate)} (Overall: {self.possible_tuples_num}) \n")
        return self.chosen_tuples, self.seen_keys
    

    def call_gemini_api(self, sentence, subj, obj, token):
        """
        Construct a prompt with the sentence and call the Gemini API.
        Returns a tuple (subject, relation, object, confidence) if successful, else None.
        """
        subj_name = subj[0]
        subj_type = subj[1].lower()

        obj_name = obj[0]
        obj_type = obj[1].lower()

        req_subject, req_object = self.relation_requirements[self.relation_name]

        if(subj_type == req_subject and (obj_type in req_object if isinstance(req_object, list) else obj_type == req_object)):
            palm.configure(api_key=self.google_gemini_api_key)
           
            # If the object is a list, join them into a string
            if isinstance(req_object, list):
                req_object_str = " or ".join(req_object)
            else:
                req_object_str = req_object

            example = EXAMPLES[self.relation_name]
            description = RELATION_DESCRIPTIONS[self.relation_name]

            spec = RELATION_SPECS[self.relation_name]
           
            prompt = (
                f'Given a sentence: "{sentence}".\n'
                f'Your task is to extract if there is a relation of {self.relation_name} in this sentence'
                f'between the subject name: {subj_name} of type {subj_type} and '
                f'the object name: {obj_name} of type {req_object_str}.\n'
                f'Such that "{subj_name} {self.relation_str} {obj_name}" can be referred from this sentence. \n'
                'You must extract the relation based solely on the content of the provided sentence.'
                f'You should answer strictly in the following JSON format:\n\n'
                '{\n'
                '  "subject": "<subject name>",\n '
                '  "object": "<object name>",\n '
                '}\n\n'
                'If the subject or object includes extra descriptive text or non-name elements, '
                'refine it to only the proper noun representing the actual person or organization name.' 
                'Otherwise, preserve the original value. For example, "Bill Gates Personal Awards" should be refined to "Bill Gates"."'
                'If such a relation exists between the subject and object, return the json as instructed.\n'
                f'If any condition is not met, the value is just return an empty json: {{}}\n'
                'Please make sure the subject and object are both valid.'
                'return an empty json if the sentence involves non-english words.'
                f'For example: {EXAMPLES[self.relation_name]}'
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
                json_match = re.search(r"\{.*?\}", response.text, re.DOTALL)
                if not json_match:
                    return None

                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return None

                return result
                
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
        return None