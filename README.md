# 6111-project2

# Query Expansion with Relevance Feedback

#### Names: Danyao Chen
#### UNI: dc3861

## **Project Overview**
This project implements a relation extraction system that identifies structured facts from unstructured natural language text on the web. Specifically, it focuses on extracting high-confidence relations of the following types:
 * Schools_Attended (e.g., Jeff Bezos → Schools_Attended → Princeton University)
 * Work_For (e.g., Alec Radford → Work_For → OpenAI)
 * Live_In (e.g., Mariah Carey → Live_In → New York City)
 * Top_Member_Employees (e.g., Nvidia → Top_Member_Employees → Jensen Huang)

The system supports two relation extraction modes:
 1. SpanBERT-based Extraction (Traditional IE Pipeline)
    - Uses named entity recognition (NER) from spaCy to identify relevant entities.
    - Filters entity pairs based on the expected types for the relation of interest.
    - Uses a fine-tuned SpanBERT model to predict whether a relation exists between the entities.
    - Returns high-confidence predictions (above a configurable threshold).
 2. Gemini-based Extraction (Few-shot Prompting)
    - Uses prompt engineering to call the Gemini large language model API.
    - Provides a one-shot in-context example for each relation type.
    - Returns the extracted subject–relation–object triple as structured JSON if present.

The system starts with a user-defined seed set of examples for a given relation and expands it iteratively using query-based web search. The text content of top-k search results is fetched and cleaned. Sentences are parsed and analyzed to extract additional relation tuples, which are added to the seed set in subsequent iterations.

This project demonstrates the paradigm shift from traditional supervised pipelines (SpanBERT) to modern LLM-powered approaches (Gemini) and allows side-by-side comparison of both techniques in an iterative extraction framework.

## **Code Structure**
```
|-- crawl_website.py
|-- driver.py
|-- extract_relations_gemini.py
|-- extract_relations_spanbert.py
|-- project2.py
```

## **Run the Project**
```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo

// after creating your virtual environment
python3 -m venv env
source env/bin/activate
(env) pip install -r requirements.txt

// after activate your env
(env) python3 project2.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>
// example
(env) python3 project2.py -gemini AIzaSyCeF-LloN0i8IJe0HZ8gDLnRTmxSjDpKvw 84f98f408aba949e8 AIzaSyBBtuMNtsm3bQ-5nlV21PCJ89xZHTRvwao 1 0.7 "sergey brin stanford" 10
(env) python3 project2.py -spanbert AIzaSyCeF-LloN0i8IJe0HZ8gDLnRTmxSjDpKvw 84f98f408aba949e8 AIzaSyBBtuMNtsm3bQ-5nlV21PCJ89xZHTRvwao 1 0.7 "sergey brin stanford" 10
```
## **My Google API key and Google Engine ID**
- "api_key": "AIzaSyCeF-LloN0i8IJe0HZ8gDLnRTmxSjDpKvw"
- "cx_id": "84f98f408aba949e8"
- "google gemini api key": "AIzaSyBBtuMNtsm3bQ-5nlV21PCJ89xZHTRvwao"
Internal Design and Project Structure
This project is organized to implement relation extraction using two different approaches: a traditional pipeline using SpanBERT and a modern few-shot prompting pipeline using Gemini. The internal design reflects a modular structure that separates crawling, sentence annotation, entity pair construction, and relation prediction.

## Code Structure
#### project2.py
* The entry point of the project. It parses command-line arguments, handles configuration parameters (like query, top-k URLs, and thresholds), and initiates the extraction process by calling functions in driver.py.
#### driver.py
* Coordinates the end-to-end information extraction process. It:
  * Iteratively expands seed sets using extracted relations.
  * Calls the appropriate extractor (SpanBERT or Gemini) based on command-line input.
  * Stores and deduplicates relation tuples.
  * Manages the control flow for multiple query iterations.
#### crawl_website.py
* Responsible for downloading and cleaning webpage content.
* Key operations:
  * Uses requests and BeautifulSoup to fetch and parse HTML content.
  * Extracts textual content, handles special formatting issues.
  * Cleans and normalizes the raw HTML to plain text, which is later tokenized.
#### spanbert.py
* Loads the pretrained SpanBERT model and defines a SpanBERT class wrapper.
* It provides a predict() method that takes in tokenized sentences with candidate entity spans and outputs the predicted relation label and its confidence score.
#### extract_relations.py
* Contains the base classes and logic to:
  * Annotate sentences using spaCy.
  * Filter entity pairs based on expected types for a given relation (e.g., PERSON–ORGANIZATION).
  * Use either SpanBERT or Gemini to perform relation classification.
  * Track seen sentences and entity pairs to avoid duplication.
#### spacy_help_functions.py
* Provides helper utilities for entity extraction and token span handling using spaCy.
* Functions include:
  * get_entities(): returns named entities of interest.
  * create_entity_pairs(): constructs all candidate (subj, obj) entity pairs from a sentence.
  * extract_relations(): helper to format and filter relation outputs.

## External Libraries
* spaCy (en_core_web_lg)
  - Used for sentence segmentation and named entity recognition (NER). This provides typed entities required for constructing valid input pairs for SpanBERT.
* Transformers / PyTorch (for SpanBERT)
  - The pretrained SpanBERT model is loaded and run using PyTorch, typically via the HuggingFace transformers interface. The model predicts relation types and outputs a confidence score.
* BeautifulSoup (bs4)
  - Used to parse and clean HTML from crawled web pages. Handles structure-aware parsing and text extraction from specific tags.
* Google GenerativeAI (Gemini API)
  - Used to extract relations through few-shot prompting. The API is used in the alternative Gemini-based pipeline that queries the model for JSON-formatted relation triples.
* Standard Python Libraries
  - re: for regex-based text cleaning.
  - requests: for HTTP requests.
  - logging: for error logging and diagnostics.
  - os, io, json, and others for utility purposes.

## Implementation Details of Step 3
#### Step 3: Information Extraction
This step implements the core relation extraction logic using two different approaches: SpanBERT and Google's Gemini API. Both methods are wrapped in their own classes—ExtractRelationsSpanbert and ExtractRelationsGemini, respectively—and can be selected at runtime using a command-line flag.

#### 1. Using SpanBERT (Traditional IE Pipeline)
The SpanBERT-based extractor is implemented in extract_relations_spanbert.py. Its workflow follows a multi-step annotation process:
* Named Entity Recognition: spaCy (en_core_web_lg) is used to segment webpage text into sentences and annotate named entities.
* Entity Pair Filtering: Based on the relation of interest (e.g., per:employee_of), entity pairs are filtered to match the required types. For instance, Work_For only runs on (PERSON, ORGANIZATION) or (ORGANIZATION, PERSON) pairs.
* SpanBERT Inference: Valid candidate pairs are passed to a fine-tuned SpanBERT model (spanbert.py) for classification and confidence scoring.
* Tuple Selection: Only predictions that match the expected relation label and exceed a user-defined confidence threshold are kept. Duplicates are filtered based on (subject, object) keys with preference for higher-confidence matches.
* Output: Extracted subject-object pairs along with relation and confidence are appended to chosen_tuples.
This implementation is optimized for efficiency by only applying SpanBERT on semantically valid entity pairs and skipping unrelated sentences early in the pipeline.

#### 2. Using Gemini (Few-Shot Prompting)
The Gemini-based approach, implemented in extract_relations_gemini.py, uses few-shot prompting for relation extraction via Google's generativeai API:
* spaCy Preprocessing: The text is first segmented into sentences using spaCy, and named entities are extracted.
* Entity Pair Identification: Entity pairs are filtered similarly to the SpanBERT pipeline, ensuring only valid combinations (e.g., PERSON and ORGANIZATION for Schools_Attended) are considered.
* Prompt Construction: For each valid sentence, a structured natural language prompt is created that specifies:
  * The relation of interest
  * Entity type constraints
  * Requirements to avoid pronouns or inferred subjects
  * An example in JSON format (e.g., {"subject": "Jeff Bezos", "relation": "Schools_Attended", "object": "Princeton University"})
* Model Inference: The prompt is passed to Gemini via the GenerativeModel API (models/gemini-2.0-flash) with deterministic generation parameters (temperature=0.2, top_p=1).
* JSON Extraction: The response is parsed for valid triples; malformed or empty responses are ignored. Duplicate keys are filtered using a seen_keys set.
* Output: Valid extracted triples are added to chosen_tuples.
This approach minimizes cost and API calls by pre-filtering entity pairs and only invoking Gemini when entity types are appropriate.
