from elasticsearch import Elasticsearch


es = Elasticsearch(
  "https://0347233eb5574af1b83750c10fdb1982.us-central1.gcp.cloud.es.io:443",
  api_key="myelakey"
)



documents = [
  { "index": { "_index": "image-index", "_id": "97805533519997"}},
  {"name": "Snow Crash", "author": "Neal Stephenson", "release_date": "1992-06-01", "page_count": 470, "_extract_binary_content": True, "_reduce_whitespace": True, "_run_ml_inference": True},
  { "index": { "_index": "image-index", "_id": "9780441017225"}},
  {"name": "Revelation Space", "author": "Alastair Reynolds", "release_date": "2000-03-15", "page_count": 585, "_extract_binary_content": True, "_reduce_whitespace": True, "_run_ml_inference": True},
  { "index": { "_index": "image-index", "_id": "9780451524935"}},
  {"name": "1984", "author": "George Orwell", "release_date": "1985-06-01", "page_count": 328, "_extract_binary_content": True, "_reduce_whitespace": True, "_run_ml_inference": True},
  { "index": { "_index": "image-index", "_id": "9781451673319"}},
  {"name": "Fahrenheit 451", "author": "Ray Bradbury", "release_date": "1953-10-15", "page_count": 227, "_extract_binary_content": True, "_reduce_whitespace": True, "_run_ml_inference": True},
  { "index": { "_index": "image-index", "_id": "9780060850524"}},
  {"name": "Brave New World", "author": "Aldous Huxley", "release_date": "1932-06-01", "page_count": 268, "_extract_binary_content": True, "_reduce_whitespace": True, "_run_ml_inference": True},
  { "index": { "_index": "image-index", "_id": "9780385490818"}},
  {"name": "The Handmaid's Tale", "author": "Margaret Atwood", "release_date": "1985-06-01", "page_count": 311, "_extract_binary_content": True, "_reduce_whitespace": True, "_run_ml_inference": True},
]

es.bulk(operations=documents, pipeline="ent-search-generic-ingestion")
# Perform operations on Elasticsearch
# For example, create an index
es.search(index="image-index", q="snow") 