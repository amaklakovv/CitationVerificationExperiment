import csv
import html
import re
import sys
from pathlib import Path

import pdfplumber
from rake_nltk import Rake

def read_document(path):
    path = Path(path)
    if path.suffix == ".pdf":
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if len(text.split()) < 50 * len(pdf.pages):  # a normal page has hundreds of words
                print(f"Warning: only {len(text.split())} words in {len(pdf.pages)} pages - {path} may be a scan, so citations will be missed")
    else:
        text = path.read_text(encoding="utf-8")
    return text

# A party name: capitalised words, allowing "of", "the", "and" in between (e.g. "Estate of Durden")
PARTY = r"[A-Z][\w.&'’-]*(?:\s+(?:of|the|and|&|[A-Z][\w.&'’-]*))*"

# A case citation: "Party v. Party, <numbers...> (<court> <year>)"
CITATION = re.compile(PARTY + r"\s+v\.\s+.{1,200}?\([^()]*\d{4}\)")

def find_claim(text, start, end):
    before = text[max(0, start - 600):start]
    closing = re.search(r"[”\"][^“”\"]{0,5}$", before)  # a quote ending right before the citation
    if closing:
        body = before[:closing.start()]
        openers = [m.end() for m in re.finditer(r"(?<!\w)[“\"'‘](?=\w)", body)]  # opening quote mark: not after a letter, before a word
        if openers and len(body) - openers[-1] >= 20:
            return body[openers[-1]:]
    sentence_start = before.rfind(". ") + 2 if ". " in before else 0  # otherwise, the sentence around the citation
    after = text[end:end + 300]
    sentence_end = after.find(". ") if ". " in after else len(after)
    return (before[sentence_start:] + after[:sentence_end]).strip()

def find_citations(text):
    text = " ".join(text.split())  # undo line wrapping, so citations split across lines join back up
    citations = []
    for match in CITATION.finditer(text):
        first_party, rest = match.group().split(" v. ", 1)
        first_party = first_party.split(". ")[-1]  # drop the end of a previous sentence caught at the start
        first_party = re.sub(r"^(In|See|See also|Cf\.)\s+", "", first_party)  # drop a leading "In"/"See"
        claim = find_claim(text, match.start(), match.end())
        citations.append((first_party + " v. " + rest, claim))
    return citations

def load_corpus(index_path="corpus/index.csv"):
    with open(index_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def squash(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())  # "678 F. Supp. 3d 443" -> "678fsupp3d443"

def name_key(citation):
    first_party, second_party = re.split(r"\s+v\.?\s+", citation, maxsplit=1)  # US "v." or Australian/NZ "v"
    return squash(first_party.split()[-1]), squash(second_party.split()[0])  # "Mata v. Avianca, Inc." -> ("mata", "avianca")

def check_existence(citation, corpus):
    for case in corpus:
        numbers = case["citation"].split("(")[0].strip()  # "678 F. Supp. 3d 443 (S.D.N.Y. 2023)" -> "678 F. Supp. 3d 443"
        if squash(numbers) in squash(citation):
            return case, f"matched citation {numbers}"
        if name_key(case["name"]) == name_key(citation):
            return case, f"matched case name {case['name']}"
    return None, "no matching name or citation in corpus"

def plain(text):
    text = re.sub(r"\*\d+", " ", text)  # drop star-page markers like *461
    return " " + " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split()) + " "  # "“Existing law”" -> " existing law "

def keyword_overlap(claim, source_text):
    rake = Rake()
    rake.extract_keywords_from_text(claim)
    phrases = list(dict.fromkeys(plain(p).strip() for p in rake.get_ranked_phrases()))  # strip quote marks, drop duplicates
    phrases = [p for p in phrases if p]
    source = plain(source_text)
    matched = [p for p in phrases if f" {p} " in source]
    missing = [p for p in phrases if f" {p} " not in source]
    exact_quote = plain(claim) in source
    return matched, missing, exact_quote

SUPPORTED = 0.8  # share of keywords that must appear in the source for a claim to count as supported

def bucket(case, matched, missing, exact_quote, threshold=SUPPORTED):
    if case is None:
        return "not_found"
    total = len(matched) + len(missing)
    if exact_quote or (total and len(matched) / total >= threshold):
        return "verified"
    return "unverifiable_quote"

def check_document(path, corpus):
    results = []
    for citation, claim in find_citations(read_document(path)):
        case, reason = check_existence(citation, corpus)
        matched, missing, exact_quote = [], [], False
        if case:
            source_text = read_document(Path("corpus") / case["file"])
            matched, missing, exact_quote = keyword_overlap(claim, source_text)
        results.append({
            "citation": citation, "claim": claim, "reason": reason,
            "matched": matched, "missing": missing, "exact_quote": exact_quote,
            "bucket": bucket(case, matched, missing, exact_quote),
        })
    return results

LABELS = {"verified": "VERIFIED", "not_found": "NOT FOUND", "unverifiable_quote": "CAN'T VERIFY"}

def print_results(path, results):
    print(path)
    for i, r in enumerate(results, 1):
        print(f"{i:2}. {LABELS[r['bucket']]}  {r['citation']}")
        print(f"      ({r['reason']})")
        if r["bucket"] != "not_found":
            print(f"      claim: {r['claim'][:150]}{'...' if len(r['claim']) > 150 else ''}")
            print(f"      exact text in source: {'yes' if r['exact_quote'] else 'no'}")
            print(f"      keywords matched {len(r['matched'])}/{len(r['matched']) + len(r['missing'])}: {r['matched']}")
            print(f"      keywords missing: {r['missing']}")

def document_path(test_doc):
    return Path("test_docs") / f"{test_doc}.txt"

def same_citation(expected, extracted):
    numbers = re.search(r"\d+ [A-Za-z0-9. ']+? \d+\b", expected)  # e.g. "925 F.3d 1339"
    if numbers and squash(numbers.group()) in squash(extracted):
        return True
    return " v. " in expected and name_key(expected) == name_key(extracted)

def match_expected(corpus, expected_path="eval/expected.csv"):
    with open(expected_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    results_by_doc = {}
    pairs = []
    for row in rows:
        doc = row["test_doc"]
        if doc not in results_by_doc:
            results_by_doc[doc] = check_document(document_path(doc), corpus)
        match = next((r for r in results_by_doc[doc] if same_citation(row["citation_text"], r["citation"])), None)
        pairs.append((row, match))
    return pairs

def evaluate(corpus):
    pairs = match_expected(corpus)
    correct = 0
    for row, match in pairs:
        got = match["bucket"] if match else "(not extracted)"
        ok = got == row["expected_bucket"]
        correct += ok
        print(f"{'PASS' if ok else 'MISS'}  expected {row['expected_bucket']:18} got {got:18} {row['citation_text'][:60]}")
        if not ok:
            print(f"      why it matters: {row['reason']}")
    print(f"\n{correct}/{len(pairs)} citations given the expected result")

def bucket_at(match, threshold):
    if match is None:
        return "(not extracted)"
    if match["bucket"] == "not_found":
        return "not_found"  # not in the corpus, whatever the threshold
    return bucket(True, match["matched"], match["missing"], match["exact_quote"], threshold)

def sweep(corpus, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    pairs = match_expected(corpus)
    found = [(row, m) for row, m in pairs if m and m["bucket"] != "not_found"]  # only these depend on the threshold
    print("Citations found in the corpus:")
    for row, m in found:
        total = len(m["matched"]) + len(m["missing"])
        print(f"  {row['test_doc']:20} {row['citation_text'][:45]:45}  keywords {len(m['matched'])}/{total}"
              f" ({len(m['matched']) / total:.0%}), exact text: {'yes' if m['exact_quote'] else 'no'}, expected {row['expected_bucket']}")
    names = [f"{row['test_doc'].split('_')[0]}→{row['citation_text'].split(' v. ')[0][:8]}" for row, _ in found]
    print(f"\n{'threshold':10} {'correct':8} " + " ".join(f"{n:20}" for n in names))
    for t in thresholds:
        correct = sum(bucket_at(m, t) == row["expected_bucket"] for row, m in pairs)
        cells = []
        for row, m in found:
            got = bucket_at(m, t)
            cells.append(f"{got + ('' if got == row['expected_bucket'] else ' x'):20}")
        print(f"{t:<10.0%} {correct}/{len(pairs):<6} " + " ".join(cells))

def esc(text):
    return html.escape(str(text))

def keywords(r):
    if r["bucket"] == "not_found":
        return ""
    return f"matched: {', '.join(r['matched']) or 'none'} | missing: {', '.join(r['missing']) or 'none'}"

def doc_title(path):
    return path.stem.replace("_", " ").title().replace(" V ", " v. ")  # "park_v_kim" -> "Park v. Kim"

def write_html(corpus):
    sources = esc(" and ".join(case["name"] for case in corpus))
    options, outputs = [], []
    for i, path in enumerate(sorted(Path("test_docs").glob("*.txt"))):
        options.append(f'<option value="doc-{i}">{esc(doc_title(path))}</option>')
        rows = "".join(f"<tr><td>{LABELS[r['bucket']]}</td><td>{esc(r['citation'])}</td><td>{esc(keywords(r))}</td></tr>"
                       for r in check_document(path, corpus))
        outputs.append(f'<div class="output" id="doc-{i}" hidden><h2>Citations found in {esc(doc_title(path))}</h2>'
                       f"<p>Each citation was looked up in the corpus of real cases ({sources}). "
                       f"NOT FOUND means it isn't in the corpus, not that it's proven fake. "
                       f"For cases that are found, the keywords show which parts of the quote appear in the real case.</p>"
                       f"<table><tr><th>Result</th><th>Citation</th><th>Keywords</th></tr>{rows}</table></div>")

    eval_rows, correct = [], 0
    pairs = match_expected(corpus)
    for row, match in pairs:
        got = match["bucket"] if match else "(not extracted)"
        ok = got == row["expected_bucket"]
        correct += ok
        eval_rows.append(f"<tr><td>{'PASS' if ok else 'MISS'}</td><td>{esc(row['citation_text'])}</td>"
                         f"<td>{LABELS[row['expected_bucket']]}</td><td>{LABELS.get(got, got)}</td></tr>")

    page = Path("ui/template.html").read_text(encoding="utf-8")
    page = page.replace("{{OPTIONS}}", "".join(options)).replace("{{OUTPUTS}}", "".join(outputs))
    page = page.replace("{{SCORE}}", f"{correct}/{len(pairs)} citations given the expected result")
    page = page.replace("{{EVAL_ROWS}}", "".join(eval_rows))
    Path("ui/results.html").write_text(page, encoding="utf-8")
    print("Wrote ui/results.html - open it in a browser")

if __name__ == "__main__":
    corpus = load_corpus()
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate(corpus)
    elif len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep(corpus)
    elif len(sys.argv) > 1 and sys.argv[1] == "html":
        write_html(corpus)
    else:
        path = sys.argv[1] if len(sys.argv) > 1 else "test_docs/mata_brief.txt"
        print_results(path, check_document(path, corpus))
