# Content-Based Recommendation System

## Overview
This project is a simple content-based recommendation system that suggests items (e.g., movies) based on a user's textual preferences. It uses TF-IDF vectorization and cosine similarity to find the top matches.

## Dataset
- **Source:** [https://github.com/LearnDataSci/articles/blob/master/Python%20Pandas%20Tutorial%20A%20Complete%20Introduction%20for%20Beginners/IMDB-Movie-Data.csv]
- **Description:** A CSV file containing movie titles and plot summaries.
- **Location:** The dataset is included in the as `IMDB-Movie-Data.csv`.

## Setup
- **Python Version:** Python 3.13.2
- **Virtual Environment:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
- **How to install dependencies:**
  
  pip install -r requirements.txt
  
- **Running:**
  
  python recommend.py "Some user description"
  
- **Resuts:**

  When the system is run with a sample query, such as: ``` python3 recommend.py "I love thrilling action movies set in space, with a comedic twist." ``` The system processes the description and recommends the top 5 movies with the closest match based on cosine similarity.

  For instance, with this query, the output might look like the following: ``` Recommendations: Title similarity 509 Gravity 0.146386 112 The Bad Batch 0.142542 511 Shooter 0.135688 564 Snow White and the Huntsman 0.128283 28 Bad Moms 0.120557 ```

  Similarity: The cosine similarity score between the user's query and the movie descriptions. Higher values indicate better matches. These recommendations are derived by comparing the user's query with the movie descriptions using the TF-IDF vectorization technique, followed by calculation of cosine similarity. ``` This provides a concise example of the output when the system is run with a sample query.

- **Salary Expectation per month**
  
  Atleast $25 an hour so $4000 per month. 

