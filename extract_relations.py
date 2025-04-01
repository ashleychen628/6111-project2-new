import spacy
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs, extract_relations

spanbert = SpanBERT("./pretrained_spanbert")
nlp = spacy.load("en_core_web_lg") 

class ExtractRelations:
    def __init__(self, r, t):
        self.relation = r
        self.threshold = t
        self.candidate_pairs = []
        self.chosen_tuples = []
        self.entities_of_interest = [
                                        "PERSON", 
                                        "ORGANIZATION", 
                                        "LOCATION", 
                                        "CITY", 
                                        "STATE_OR_PROVINCE", 
                                        "COUNTRY"
                                    ]
                  

    def extract_relations_spanbert(self, raw_text):
        if self.candidate_pairs != 0:


         
            print("Candidate entity pairs:")
            for p in candidate_pairs:
                print("Subject: {}\tObject: {}".format(p["subj"][0:2], p["obj"][0:2]))
            print("Applying SpanBERT for each of the {} candidate pairs. This should take some time...".format(len(candidate_pairs)))

            # if len(candidate_pairs) == 0:
            #     continue
            
            # relation_preds = spanbert.predict(candidate_pairs)  # get predictions: list of (relation, confidence) pairs

            # Print Extracted Relations
            # print("\nExtracted relations:")
            # for ex, pred in list(zip(candidate_pairs, relation_preds)):
            #     print("\tSubject: {}\tObject: {}\tRelation: {}\tConfidence: {:.2f}".format(ex["subj"][0], ex["obj"][0], pred[0], pred[1]))

            #     if self.relation == 1:  # Schools_Attended
            #         subject_type, object_type = "PERSON", "ORGANIZATION"
            #     elif self.relation == 2:  # Work_For
            #         subject_type, object_type = "PERSON", "ORGANIZATION"
            #     elif self.relation == 3:  # Live_In
            #         subject_type, object_type = "PERSON", ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
            #     elif self.relation == 4:  # Top_Member_Employees
            #         subject_type, object_type = "ORGANIZATION", "PERSON"
    
    def extract_entities_spacy(self, raw_text):
        """Process webpage text and extract sentences using spaCy."""
        doc = nlp(raw_text)
        print("Annotating the webpage using spacy...")
        sentences = list(doc.sents)
        print(f"Extracted {len(list(doc.sents))} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
        for idx, sentence in enumerate(sentences):
            if (idx + 1) % 5 == 0:
                print(f"\n\tProcessed {idx + 1} / {len(sentences)} sentences")
            # print("\n\nProcessing sentence: {}".format(sentence))
            # print("Tokenized sentence: {}".format([token.text for token in sentence]))
            ents = get_entities(sentence, self.entities_of_interest)
            # print("spaCy extracted entities: {}".format(ents))
            
            # create entity pairs
            sentence_entity_pairs = create_entity_pairs(sentence, self.entities_of_interest)
            # Inside your class method
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
                        
            # print("Candidate entity pairs:")
            # for p in self.candidate_pairs:
            #     print("Subject: {}\tObject: {}".format(p["subj"][0:2], p["obj"][0:2]))
            
            if len(self.candidate_pairs) == 0:
                continue
            else:
                relation_predictions = spanbert.predict(self.candidate_pairs)  # get predictions: list of (relation, confidence) pairs

            relation_mapping = {
                1: ("per:schools_attended", "PERSON", "ORGANIZATION"),
                2: ("per:employee_of", "PERSON", "ORGANIZATION"),
                3: ("per:cities_of_residence", "PERSON", ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]),
                4: ("org:top_members/employees", "ORGANIZATION", "PERSON")
            }

            expected_label, expected_subj_type, expected_obj_type = relation_mapping[self.relation]

            for ex, pred in zip(self.candidate_pairs, relation_predictions):
                relation_label, confidence = pred
                subj, obj = ex['subj'], ex['obj']
                tokens = ex['tokens']

                if relation_label == expected_label and (subj[1] == expected_subj_type and (obj[1] in expected_obj_type if isinstance(expected_obj_type, list) else obj[1] == expected_obj_type)):
                    print("\n\t\t=== Extracted Relation ===")
                    print(f"\t\tInput tokens: {tokens}")
                    print(f"\t\tOutput Confidence: {confidence:.7f} ; Subject: {subj[0]} ; Object: {obj[0]} ;")
                    
                    if confidence >= self.threshold:
                        print("\t\tAdding to set of extracted relations")
                        print("\t\t==========")
                        self.chosen_tuples.append({
                            "subject": subj[0],
                            "object": obj[0],
                        })
                    else:
                        print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
                        print("\t\t==========")

            # # Print Extracted Relations
            # print("\nExtracted relations:")
            # for ex, pred in list(zip(self.candidate_pairs, relation_preds)):
            #     print("\tSubject: {}\tObject: {}\tRelation: {}\tConfidence: {:.2f}".format(ex["subj"][0], ex["obj"][0], pred[0], pred[1]))

            #     # TODO: focus on target relations
            #     # '1':"per:schools_attended"
            #     # '2':"per:employee_of"
            #     # '3':"per:cities_of_residence"
            #     # '4':"org:top_members/employees"