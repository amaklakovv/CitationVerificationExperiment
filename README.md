# Citation Checker (AIML430 Capstone Project)

Checks the case citations in a document against a small corpus of real court opinions: is the cited case in the corpus, and does it contain the quote attributed to it?

## Setup

Requires Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pdfplumber rake-nltk
python -m nltk.downloader stopwords punkt punkt_tab
```

## Run

```bash
# check the Mata brief
python citation_check.py

# check any .txt or .pdf
python citation_check.py test_docs/park_v_kim.txt

# compare against eval/expected.csv
python citation_check.py eval

# results at different thresholds                            
python citation_check.py sweep

# build ui/results.html
python citation_check.py html
```

Open `ui/results.html` in a browser to view the results without running anything.
