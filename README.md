# 6111-project1

# Query Expansion with Relevance Feedback

#### Names: Danyao Chen
#### UNI: dc3861

## **Project Overview**
This project implements an **information retrieval system** that improves Google search results using **relevance feedback and query expansion**. The system refines user queries iteratively by incorporating **TF-IDF-based keyword selection** and **Vector Space Model (VSM) ordering** to enhance search precision.

The method follows an **interactive relevance feedback loop**, where:
1. The user issues an **initial query**.
2. The system retrieves the **top 10 results** using the **Google Custom Search API**.
3. The user **marks relevant results**.
4. The system **expands the query** by selecting **two important words** from relevant results using **TF-IDF ranking**.
5. The **expanded query is reordered** using **cosine similarity** to prioritize relevant terms.
6. The process repeats until the **desired precision@10** is met or no further improvements can be made.

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
### **Our Google API key and Google Engine ID**
- "api_key": "AIzaSyCeF-LloN0i8IJe0HZ8gDLnRTmxSjDpKvw"
- "cx_id": "84f98f408aba949e8"
- "google gemini api key": "AIzaSyBBtuMNtsm3bQ-5nlV21PCJ89xZHTRvwao"

## **Methodology**
This project implements **two key query expansion techniques**:

### **1. TF-IDF-Based Keyword Selection**
- Extracts **important words** from the **user-marked relevant search results**.
- Uses **TF-IDF scores** to rank words by importance.
- Selects **top 2 words** that are not already in the query.

#### Libraries Used for TF-IDF calculation
- sklearn.feature_extraction.text.TfidfVectorizer – Converts text into TF-IDF vectors for information retrieval.
- sklearn.metrics.pairwise.cosine_similarity – Computes cosine similarity to reorder query terms based on relevance.
  
### **2. Query Reordering using Vector Space Model (VSM)**
- Computes **cosine similarity** between query words and relevant document vectors.
- Reorders the **expanded query** to **prioritize more relevant terms**.
- Ensures the **original query words always remain at the front** to preserve intent.

#### Other External Libraries Used
The following external libraries are used in this project:
- collections (Standard Library)
  Used for efficient data structures such as defaultdict, Counter, and deque, which help in handling data organization and frequency counting.
- re (Standard Library)
  Used for regular expression operations, enabling pattern matching and text manipulation.
- glob (Standard Library)
  Used for file path matching with wildcard patterns, facilitating batch processing of files in directories.
- numpy (Third-Party Library)
  A powerful numerical computing library used for array manipulation, mathematical operations, and efficient data processing.

## **Query-Modification Method**

Our query-modification method is designed to iteratively improve search queries by identifying and incorporating the most relevant keywords while preserving an optimal query structure. This ensures that each refinement leads to better search results, increasing precision with every round.

### 1. Selecting New Keywords

Each round of query expansion selects two new words that are highly relevant to the search intent. The process is as follows:

#### Step 1: Extracting Key Terms from Relevant Documents

- The system collects snippets from the search results that the user has marked as **relevant**.
- These snippets are **cleaned and preprocessed**:
  - Removing special characters, punctuation, and numbers.
  - Converting text to lowercase (`casefold()`).
  - Removing **stopwords** from a predefined list (`proj1-stop.txt`).
- After preprocessing, each snippet is treated as a **document** for term analysis.

#### Step 2: Computing Word Importance Using TF-IDF

- A **TF-IDF vectorizer** is used to **compute the importance** of words in the cleaned snippets.
- The **TF-IDF score** represents how important a word is in the given context.
- Words with **higher TF-IDF scores** across all relevant snippets are considered **more informative**.

#### Step 3: Filtering and Selecting the Top 2 Words

- The **top TF-IDF words** are sorted in **descending order of importance**.
- Words **already present** in the original query are ignored to avoid redundancy.
- The system selects the **first two words** that are **not already in the query**.
- If there are fewer than two valid words, it selects as many as possible.

---

### 2. Determining Query Word Order

After selecting the most relevant words, the query is **expanded and reordered** for better ranking and retrieval performance. The following steps are used:

#### Step 1: Expanding the Query

- The new keywords are added to the current query to form an **expanded query**.
- The updated query consists of:

  ```
  original query + new top 2 words
  ```

- This ensures that the expanded query retains its **original intent** while incorporating relevant new terms.

#### Step 2: Reordering Using Cosine Similarity

- The expanded query is **vectorized** using the same **TF-IDF model**.
- The **cosine similarity** between the query vector and the TF-IDF vectors of the relevant documents is computed.
- This similarity score determines how well each word in the query **aligns with the relevant results**.

#### Step 3: Sorting Words Based on Relevance

- Words in the expanded query are **reordered** based on their average cosine similarity scores.
- Words that have **higher similarity** to relevant documents are placed **earlier in the query**.
- This ensures that **more relevant terms appear first**, improving search engine ranking.

---

### 3. Summary

- **TF-IDF scoring** is used to extract **the two most relevant** new words.
- The **original query is preserved**, and new words are **added in a structured way**.
- **Cosine similarity reordering** ensures that **more relevant terms appear first**.
- This method ensures that the query **gradually improves over multiple iterations**, leading to higher precision and better retrieval results.

## **Future Improvements**
- Implement Rocchio Algorithm to weight relevant and non-relevant terms.
- Experiment with Word Embeddings (Word2Vec/BERT) for better query expansion.
- Integrate stopword removal & stemming for improved term selection.
