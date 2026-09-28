# Lab 03 — What RAG Is Made Of

Lab session in week 5; content maps to week 4, Context Engineering (LO-03) — the course is
running one week behind its official calendar · Narxoz University

A retrieval-augmented generation system is four small, separately-testable pieces: **chunking**
(cut documents into retrievable units), **embedding** (turn text into vectors), **retrieval**
(find the vectors closest to a question), and **generation** (answer using only what retrieval
found). This lab builds all four by hand, on four short policy articles from a fictional bank —
same domain and same "facts are invented, not real banking rules" convention as Lab 01's support
queue.

**No API key. No installation. No cost.** Two small open models are downloaded inside the
notebook itself: an embedding model for retrieval, an instruction-following model for
generation. Neither needs a Hugging Face login.

## Setup

Open the notebook in Google Colab — one click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unreal-kz/lab-03-AI-course/blob/main/lab03_what_is_rag.ipynb)

Run the first cell (`pip install`) with *Runtime → Run all* if you like — it only installs a
package. From Part 1 on, run cells one at a time: write your prediction down before you run the
cell that reveals the answer.

## Why these two models

- **Embedding: `intfloat/multilingual-e5-small`.** Small enough to embed everything in this lab
  in a couple of seconds on CPU, and multilingual enough to make the Advanced cross-language task
  possible. Its convention — prefix documents with `"passage: "` and questions with `"query: "` —
  is specific to the E5 model family; other embedding models have no such requirement, or a
  different one.
- **Generation: `Qwen/Qwen2.5-1.5B-Instruct`.** Not gated (no Hugging Face login), unlike several
  comparable small instruct models. It is instruction-tuned, unlike Lab 02's GPT-2, so it can
  actually attempt an answer instead of just continuing text.

## Part 0 — the corpus

Four articles from the bank's policy pages, all about payment cards — deliberately
near-neighbours, so retrieval has to tell them apart, not just notice the word "card". Each
article is a list of sentences; `needle_idx` marks the one sentence that answers that article's
two questions (`literal`, a direct restatement, and `paraphrase`, the same need in different
words). Read all four before you touch the code.

## Part 1 — chunking

Two ways to cut the four articles into retrievable chunks: **article-level** (one chunk per
article) and **sentence-level** (one chunk per sentence). Predict the two chunk counts from the
sentence totals printed in Part 0, then measure.

## Part 2 — embedding and retrieval

Embed every chunk, embed a question, rank chunks by cosine similarity. You predict the retrieved
article/sentence before running, for both a `literal` and a `paraphrase` version of the same
question.

## Part 3 — generation, without and with retrieval

`Qwen2.5-1.5B-Instruct` has never seen this fictional bank's policy. Ask it the question directly
and it can only guess; hand it the one retrieved sentence as its only allowed fact, and it can
answer correctly. Same question, two prompts — the difference between the two answers **is** the
point of RAG.

## What you hand in

One page. Download the notebook (*File → Download → .ipynb*) after running every cell and
answering every ✏️ prompt, and upload it to Canvas.

1. Your Part 1 prediction and the two measured chunk counts.
2. Your Part 2 predictions (article-level and sentence-level, both queries) next to what actually
   came back.
3. The two Part 3 answers (without and with context), and one sentence on what the no-context
   answer got wrong.
4. One sentence: for a corpus like this one, would you chunk by article or by sentence, and why?

## Extension tasks

Every number under "Expect" is from an actual run (`reference_run.py`, `extra_checks.py` —
`intfloat/multilingual-e5-small` + `Qwen/Qwen2.5-1.5B-Instruct`, greedy decoding, 2026-09-28), not
from memory. Greedy decoding is deterministic, so your own run should match closely; a different
`transformers` version may shift the last digit of a score but not which chunk wins.

### Core — no new tools

**1. Repeat the walkthrough on `cards-pin-block`'s literal query.**
Do: run Part 2 and Part 3 with `query = CORPUS[1]["queries"]["literal"]`
(*"How long does it take for the card to work again after I unblock it in the app?"*).
Hand in: the article-level and sentence-level top results, and whether either one is the needle
(`"Unblocking in the app is free and takes effect within 2 minutes."`).
Expect: article-level correctly retrieves `cards-pin-block` (score 0.908). Sentence-level
retrieves `cards-lost:1` — *"Blocking takes effect within 60 seconds, after which the card cannot
be used..."* (score 0.888) — a wrong **article**, not just a wrong sentence, and the actual needle
does not even appear in the sentence-level top 3.
Trap: "blocking" and "unblocking" are near-opposite actions described in almost the same sentence
shape ("X takes effect within Y, after which..."). A high cosine score tells you the phrasing is
close, not that the meaning is close — reversed actions with parallel grammar are exactly what
sentence-level chunking confuses most.

**2. Recall@1 across all eight queries.**
Do: loop Part 2's retrieval over all four articles × both query kinds, and count how often each
granularity's top-1 result is the needle.
Hand in: the two recall@1 fractions.
Expect: article-level 0.75 (6/8), sentence-level 0.375 (3/8) — sentence-level is **worse**, not
better.
Trap: don't assume finer-grained chunking is automatically more precise. A single short sentence
carries less surrounding context than a whole article, so its embedding has less to anchor on and
drifts toward whatever other short sentence happens to share vocabulary — even from the wrong
article. A whole-article embedding averages over enough content that the dominant topic usually
still wins.

**3. The near-tie in the main walkthrough.**
Do: look at the sentence-level top-3 for `cards-lost`'s literal query, printed in Part 2.
Hand in: the score gap between the top-ranked (wrong) sentence and the actually-correct needle
sentence, which is ranked second.
Expect: 0.8514 (`cards-lost:4`, wrong) vs 0.8476 (`cards-lost:5`, the needle) — a gap of 0.0038.
Trap: a cosine score is not a probability, and 0.85 is not "85% confident" — it is only useful for
ranking candidates against each other. A gap this small means the ranking is close to arbitrary;
retrieving only the top-1 (`k=1`) would have silently handed the generator the wrong fact.

### Advanced — extra credit, pick one

**4. Cross-language retrieval.**
Do: take the Kazakh and Russian versions of `cards-lost`'s literal query from
`Lectures/lecture-05-eval/corpus.py` (same `intfloat/multilingual-e5-small` model, same English
sentence-level index — no re-embedding of documents needed, only the query changes language).
Hand in: the top-3 for each language, and whether the needle is retrieved.
Expect: the **Russian** query correctly ranks the needle `cards-lost:5` first (score 0.757,
notably lower than the English-query score of 0.848 for the same needle). The **Kazakh** query
does not: it ranks `cards-expiry:0` first (0.828) and `cards-expiry:4` second (0.821); the needle
`cards-lost:5` is third (0.809).
Trap: don't conclude "the model handles multilingual retrieval." It handles Russian passably and
Kazakh worse, on this single query — consistent with the tokenizer disparity already measured in
Lecture 2 and Lecture 5, and worth stating explicitly rather than assuming a "multilingual" model
name means uniform quality across languages.

**5. Generation in Russian, and a translation check.**
Do: ask the generation model the Russian version of the `cards-lost` literal query
(`"Сколько стоит моментальная карта в отделении?"`), with the Russian needle sentence as its only
allowed fact.
Hand in: the answer, and — separately — one sentence on whether you found any awkward phrasing or
mistranslation in the Russian or Kazakh text in `corpus.py`. That file marks itself
`REVIEW_STATUS = "unreviewed"`: the Russian and especially Kazakh translations have not yet been
checked by a native speaker, and finding a real error is a valid, useful answer to this task.
Expect: `"3000 тенге."` — correct and grounded, same as the English case.
Trap: a correct one-line answer to a factual question does not mean the translation quality is
fine everywhere in the file — check a sentence you did not use in this task before concluding
anything about overall quality.

## Files

| File | What it is |
|---|---|
| `lab03_what_is_rag.ipynb` | The lab: 18 cells, Parts 0–3, no outputs saved |
| `README.md` | This file |
