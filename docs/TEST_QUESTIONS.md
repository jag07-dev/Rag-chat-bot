# Long-document test: NorthStar Logistics Operations Manual

**File:** `data/northstar_operations_manual.pdf`. It has 105 pages and about 57,000 words. With the app's default settings (60 words, 15 overlap) it becomes 1,268 chunks.

**This document is fictional.** Every company, hub, figure, and policy was invented for testing. Most of the text is generic filler without any facts, which is deliberate: it makes the real facts hard to find, as in a long real-world PDF.

**Deliberate traps**
- 12 hub profiles with the same wording but different numbers (retrieval must pick the right hub).
- Three incident-reporting rules that differ only in the deadline (4, 2, and 24 hours).
- Two vehicle servicing intervals and two stacking heights.
- Two tables (leave by country, purchase approval limits).
- Questions about things that are not in the document, including near-misses (a hub and countries that don't exist).

## How to test

**In the app:** run `streamlit run app.py`, upload the manual, and ask the questions below. A long PDF may hit the memory limit of the free cloud tier, so testing locally is more reliable.

**With the eval harness:** from the `eval` folder, run the commands in the README's Evaluation section. The question set is `eval_set_long.json`.

## Answerable questions

| ID | Question | Expected answer | PDF page |
|---|---|---|---|
| L01 | In which year was NorthStar Logistics founded? | 2011 | 2 |
| L02 | Where is the company headquartered? | Pune | 2 |
| H01 | How many dock doors does the Kochi hub have? | 20 | 16 |
| H02 | In which year did the Bhopal hub open? | 2020 | 17 |
| H03 | How many pallet positions does the Guwahati hub have? | 19,900 | 19 |
| H04 | How many people work at the Surat hub? | 505 | 19 |
| P01 | How many days of annual leave do employees in Germany get? | 30 | 21 |
| P02 | How many days of sick leave do employees in Singapore get? | 14 | 21 |
| P03 | How long is the probation period for operations staff? | 6 months | 22 |
| S01 | Within how many hours must a warehouse injury be reported? | 4 hours | 24 |
| S02 | Within how many hours must a vehicle accident be reported? | 2 hours | 25 |
| S03 | Within how many hours must a data security incident be reported? | 24 hours | 25 |
| S04 | How often must forklift operators renew their licence? | Every 18 months | 26 |
| S05 | What is the maximum stacking height in the cold chain zone? | 3 levels | 27 |
| F01 | How often are light vans serviced? | Every 10,000 km | 29 |
| F02 | How quickly are express deliveries completed? | Within 24 hours | 30 |
| F03 | How long must drivers break, and after how many hours of driving? | 30 minutes after 4.5 hours | 30 |
| R01 | Who must approve a purchase of INR 3,000,000? | The Chief Financial Officer | 32 |
| R02 | How many written quotations are needed for purchases above INR 100,000? | At least 3 | 33 |
| M01 | What is the combined pallet capacity of the Pune and Nagpur hubs? | 42,000 + 31,500 = 73,500 | 12 |

Page numbers are PDF pages (page 1 is the cover). Use Ctrl+F in the PDF to check any answer.

## Questions the bot should refuse

| ID | Question | Why |
|---|---|---|
| U01 | How many dock doors does the Kolkata hub have? | There is no Kolkata hub (near-miss of H01) |
| U02 | How many days of annual leave do employees in Brazil get? | Brazil is not in the leave table |
| U03 | Who is the CEO of NorthStar Logistics? | No CEO is named |
| U04 | What is the company's policy on paying suppliers in cryptocurrency? | Not covered |
| U05 | How many days of sick leave do employees in Canada get? | Canada is not in the leave table |
| U06 | What is the fuel price per litre used for budgeting? | Not covered |

## Broad questions to try by hand (not in the eval file)

- "Summarize this document."
- "What are the main topics covered in the manual?"
- "List all the hubs and the cities they are in."
- "Which hub has the most dock doors?" (needs comparing 12 profiles)
- "What is the difference between the vehicle accident and warehouse injury reporting rules?"

Broad and comparison questions are expected to be weak, because the app sends only the top 8 chunks (about 476 words) to the model, so it cannot see the whole document. Record what actually happens: it is useful for the README and for interviews.

## What to record

For each run, note:
1. Time for the first upload to finish embedding, and the time for the second question.
2. hit@1, hit@3, MRR, and answer correctness from the harness.
3. Which questions failed, grouped by type (hub, distractor, table, multi-part, broad).
4. One fix you tried and its effect on the numbers.
