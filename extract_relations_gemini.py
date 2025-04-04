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
    def __init__(self, r, t, google_gemini_api_key, seen_keys):
        self.relation = r
        self.threshold = t
        self.google_gemini_api_key = google_gemini_api_key
        self.candidate_pairs = []
        self.chosen_tuples = []
        self.relation_map = {}
        self.seen_sentence = set()
        self.seen_keys = seen_keys

        
        relation_map = {
            1: "Schools_Attended",
            2: "Work_For",
            3: "Live_In",
            4: "Top_Member_Employees"
        }
        self.relation_name = relation_map[r]

        self.relation_requirements = {
            "Schools_Attended": ("person", "organization"),
            "Work_For": ("oerson", "organization"),
            "Live_In": ("oerson", ["location", "city", "state or province", "country"]),
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
                for ex in self.candidate_pairs:
                    subj, obj = ex['subj'], ex['obj']
                    tokens = ex['tokens']
                    sentence = " ".join(tokens)
                    relation_tuple = self.call_gemini_api(sentence)

                    if relation_tuple is not None:
                        subj, obj = relation_tuple
                        key = (subj, obj)

                        if key in self.seen_keys:
                            continue

                        if sentence not in self.seen_sentence:
                            self.seen_sentence.add(sentence)
                            extracted_annotations += 1

                        self.chosen_tuples.append({
                            "subject": subj,
                            "object": obj,
                            "confidence": 1,
                            "key": key
                        })
                        self.seen_keys.add(key)

                        print("\n\t\t=== Extracted Relation ===")
                        print(f"\t\tSentence:: {sentence}")
                        print(f"\t\tSubject: {subj} ; Object: {obj} ;")
                        print(f"\t\tAdding to set of extracted relations")
                        print("\t\t==========")

            self.candidate_pairs = []

        print(f"\tExtracted annotations for  {extracted_annotations}  out of total  {len(sentences)}  sentences \n")
        print(f"\tRelations extracted from this website: {len(self.chosen_tuples)} (Overall: {len(self.relation_map)}) \n")
        return self.chosen_tuples, self.seen_keys
    

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

        example = EXAMPLES[self.relation_name]
        description = RELATION_DESCRIPTIONS[self.relation_name]
# f"For this step, extract the relation '{self.relation_name}' from the following sentence: \"{sentence}\" \n"
        # prompt = (
        #     f"I am implementing an Iterative Set Expansion (ISE) algorithm. My goal is to extract factual relations from natural language sentences.\n"
            
        #     f"Your task is to extract the '{self.relation_name}' relation from the following sentence: \"{sentence}\" \n"
        #     f"For this relation '{self.relation_name}', the subject must be of type {req_subject} and the object must be of type {req_object}.\n"
        #     f"For type person, use the actual names as they appear in the sentence. **Avoid pronouns** like 'he', 'she', or 'they'.\n"
        #     f"If the relation is present, return the result in JSON format with keys 'subject', 'relation', and 'object'.\n"
        #     f"If the relation is not present or there is not enough information, return an empty JSON object{{}}.\n"
        #     f"If the sentence contains non-English words, skip these words."
        #     f"Ignore sentences that are unclear, vague, or refer to lists of languages, dates, or unrelated facts.\n"
        #     f"Example: {example}"
        # )
        
        spec = RELATION_SPECS[self.relation_name]

        prompt = (
            f'Given a sentence: "{sentence}", extract a relation in the following JSON format:\n\n'
            '{\n'
            '  "subject": "<subject>",\n'
            '  "object": "<object>"\n'
            '}\n\n'
            'The value of this json has to be exactly the same as it appears in the sentence.\n'
            'Only extract if all of the following conditions are satisfied:\n\n'
            f'- The relation is "{self.relation_name}".\n'
            f'- The subject is {spec["subject_type"]}.\n'
            f'- The object is a proper noun that refers to {spec["object_type"]}.\n'
            f'- The object must {spec["object_constraints"]}.\n'
            f'- Both subject and object must be nouns, and clearly connected by the "{self.relation_name}" relation.\n'
            '- Both subject and object must appear explicitly and unambiguously in the sentence.\n'
            f'- The relation must clearly indicate that {spec["relation_description"]}.'
            '- The sentence must provide a clear, unambiguous match for the relation — no inferred subjects or vague contexts.\n\n'
            'If any condition is not met, return this exact output:\n'
            '{}'
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

            # Validate the result
            if result and "subject" in result and "object" in result:
                return (result["subject"], result["object"])
            
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