import spacy
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs, extract_relations

spanbert = SpanBERT("./pretrained_spanbert")
nlp = spacy.load("en_core_web_lg") 

class ExtractRelations:
    def __init__(self, relation):
        self.relation = relation
        self.candidate_pairs = []
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
        print(f"Extracted {len(list(doc.sents))} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
        for sentence in doc.sents:  
            # print("\n\nProcessing sentence: {}".format(sentence))
            # print("Tokenized sentence: {}".format([token.text for token in sentence]))
            ents = get_entities(sentence, self.entities_of_interest)
            # print("spaCy extracted entities: {}".format(ents))
            
            # create entity pairs
            sentence_entity_pairs = create_entity_pairs(sentence, self.entities_of_interest)
            for ep in sentence_entity_pairs:
                if (
                    (ep[2][1] == 'PERSON' and ep[1][1] == 'ORGANIZATION') or # Top_Member_Employees 
                    (ep[1][1] == 'PERSON' and ep[2][1] == 'ORGANIZATION') or # Schools_Attended / Work_For
                    (ep[1][1] == 'PERSON' and ep[2][1] in ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]) # Live_In
                ):
                    self.candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]}) # e1=Subject, e2=Object
            
            # Classify Relations for all Candidate Entity Pairs using SpanBERT
            self.candidate_pairs = [p for p in self.candidate_pairs if p["subj"][1] in ["PERSON", "ORGANIZATION"]]
            self.candidate_pairs = [p for p in self.candidate_pairs if p["obj"][1] in ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]]

            # print("Candidate entity pairs:")
            # for p in self.candidate_pairs:
            #     print("Subject: {}\tObject: {}".format(p["subj"][0:2], p["obj"][0:2]))
            print("Applying SpanBERT for each of the {} candidate pairs. This should take some time...".format(len(self.candidate_pairs)))

            if len(self.candidate_pairs) == 0:
                continue
            else:
                relation_predictions = spanbert.predict(self.candidate_pairs)  # get predictions: list of (relation, confidence) pairs

            # # Print Extracted Relations
            # print("\nExtracted relations:")
            # for ex, pred in list(zip(self.candidate_pairs, relation_preds)):
            #     print("\tSubject: {}\tObject: {}\tRelation: {}\tConfidence: {:.2f}".format(ex["subj"][0], ex["obj"][0], pred[0], pred[1]))

            #     # TODO: focus on target relations
            #     # '1':"per:schools_attended"
            #     # '2':"per:employee_of"
            #     # '3':"per:cities_of_residence"
            #     # '4':"org:top_members/employees"