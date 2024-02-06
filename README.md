# Simple Persian IR Engine
This information retrival engine was developed for Information Retrival course (CE421) fall 2023

Elements implemented in this project include:
- Persian Text Normalization
- Persian Text tokenizing
- Stop word removal
- Positional Inverted Index generation
- Champions list generation
- Similarity calculated using **DAAT Cosine Similarity** (qf x tf x idf)
- Query Search using **Index Elimination** based on **Term Frequency** metric

&ast; Word stemming was performed using <a href="https://github.com/roshan-research/hazm">Hazm</a> toolkit 
## Input Corpora
The input corpora json file should contain the following fields for each document. In the following, 0 is document id which should be unique.
```json
{
  "0":
  {
    "title":"",
    "content":"",
    "url":""
  }
}
```
A small example of an acceptable corpora is provided: <a href="data_50.json">data_50</a>