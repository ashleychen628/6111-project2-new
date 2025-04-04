import spacy
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs, extract_relations

spanbert = SpanBERT("./pretrained_spanbert")
nlp = spacy.load("en_core_web_lg") 

class ExtractRelationsSpanbert:
    def __init__(self, r, t, seen_keys):
        self.relation = r
        self.threshold = t
        self.candidate_pairs = []
        self.chosen_tuples = []
        self.relation_map = {}
        self.seen_keys = seen_keys
        self.seen_token_spans = set()

        self.entities_of_interest = {
          1: ["PERSON", "ORGANIZATION"], # Schools_Attended
          2: ["PERSON", "ORGANIZATION"], # Work_For 
          3: ["PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"], # Live_In
          4: ["ORGANIZATION", "PERSON"] # Top_Member_Employees
        }

        self.relation_mapping = {
                1: ("per:schools_attended", "PERSON", "ORGANIZATION"),
                2: ("per:employee_of", "PERSON", "ORGANIZATION"),
                3: ("per:cities_of_residence", "PERSON", ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]),
                4: ("org:top_members/employees", "ORGANIZATION", "PERSON")
            }
                  
    
    def extract_relations_spanbert(self, raw_text):
        """Process webpage text and extract sentences using spaCy."""
        doc = nlp(raw_text)

        print("Annotating the webpage using spacy...")

        sentences = list(doc.sents)
        # print(sentences)
        extracted_annotations = 0

        print(f"Extracted {len(list(doc.sents))} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
        
        for idx, sentence in enumerate(sentences):
            self.candidate_pairs = []
            if (idx + 1) % 5 == 0:
                print(f"\n\tProcessed {idx + 1} / {len(sentences)} sentences")
            # print(f"index: {idx + 1}, sentence: {sentence} \n")
            ents = get_entities(sentence, self.entities_of_interest[self.relation])
            
            # create entity pairs
            sentence_entity_pairs = create_entity_pairs(sentence, self.entities_of_interest[self.relation])

            for ep in sentence_entity_pairs:
                tokens, subj, obj = ep[0], ep[1], ep[2]

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
                relation_predictions = spanbert.predict(self.candidate_pairs)  # get predictions: list of (relation, confidence) pairs

            expected_label, expected_subj_type, expected_obj_type = self.relation_mapping[self.relation]

            for ex, pred in zip(self.candidate_pairs, relation_predictions):

                relation_label, confidence = pred
                subj, obj = ex['subj'], ex['obj']
                tokens = ex['tokens']
                key = (subj[0], obj[0])

                if relation_label == expected_label and \
                (subj[1] == expected_subj_type and (obj[1] in expected_obj_type if isinstance(expected_obj_type, list) else obj[1] == expected_obj_type)):
                    if key not in self.relation_map or confidence > self.relation_map[key][0]:
                        self.relation_map[key] = (confidence, relation_label, tokens)
                    else:
                        continue
                    
                    print("\n\t\t=== Extracted Relation ===")
                    print(f"\t\tInput tokens: {tokens}")
                    print(f"\t\tOutput Confidence: {confidence:.7f} ; Subject: {subj[0]} ; Object: {obj[0]} ;")

                    if confidence >= self.threshold:
                        token_tuple = tuple(tokens)
                        if token_tuple not in self.seen_token_spans:
                            self.seen_token_spans.add(token_tuple)
                            extracted_annotations += 1

                        if key not in self.seen_keys:
                            print("\t\tAdding to set of extracted relations")
                            self.chosen_tuples.append({
                                "subject": subj[0],
                                "object": obj[0],
                                "confidence": confidence,
                                "key": key
                            })
                            self.seen_keys.add(key)
                        else:
                            print("\t\tDuplicate with lower confidence than existing record. Ignoring this.")
                        print("\t\t==========")
                    else:
                        print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
                        print("\t\t==========")

        print(f"\n\tExtracted annotations for  {extracted_annotations}  out of total  {len(sentences)}  sentences")
        print(f"\n\tRelations extracted from this website: {len(self.chosen_tuples)} (Overall: {len(self.relation_map)})")

        return self.chosen_tuples, self.seen_keys
