# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {Dr. Priya Nair on the Science of Sleep Deprivation}
Description: {Dr. Priya Nair has spent fifteen years studying what happens to the brain when it doesn't sleep. In this episode, we talk through her landmark 2019 study on cumulative sleep debt, what the research says about weekend recovery sleep (spoiler: it doesn't work the way you think), and why she believes the eight-hour standard is more cultural myth than biological fact. She also shares what changed in her own sleep habits after spending a decade measuring everyone else's. If you've ever felt fine on five hours, this conversation will make you rethink that confidence.}
Label: {interview}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```

[blank — you need to parse the response in classify_episode(). What format
makes parsing reliable? Think about: a single label on its own line?
A structured format like "Label: X / Reasoning: Y"? JSON?
What are the tradeoffs?]

JSON — a single object: {"label": "<one of the four labels>", "reasoning": "<1-2 sentences>"}.

Why JSON over the alternatives:
- "Label: X / Reasoning: Y" is easy to read but brittle to parse — the model
  varies the prefix ("Label:", "**Label**", "The label is"), so split-on-colon
  logic breaks. Multi-line reasoning is also hard to capture.
- A bare label on its own line is trivial to parse but throws away the reasoning,
  and the model often adds a sentence anyway that then needs stripping.
- JSON has explicit named keys, so label and reasoning are unambiguous regardless
  of wording. json.loads() handles whitespace and multi-line values for free, and
  the keys map directly onto the return dict.

Tradeoff: the model can wrap JSON in ```json fences or add stray text, so the
parser must strip fences / slice out the {...} substring before json.loads(),
and fall back to "unknown" if that fails. The prompt asks for "ONLY a JSON
object, no markdown fences" to minimize this.
```

---

**Edge cases to handle in the prompt:**

```
[blank — what if labeled_examples is empty? What if the description is very
short? How does your prompt handle these?]

A/ If labeled_examples is empty: build a zero-shot prompt — include the task
instruction and the four label definitions, but omit the examples section
entirely (no empty "Examples:" header). The model classifies from the
definitions alone; accuracy is lower, so this is a fallback, not the norm.

B/ If the description is very short or empty: still send it in the normal
format. Don't fabricate detail. There may be too little signal to choose
confidently — that's fine, the model can reason briefly and the result may
fall through to "unknown" at validation. No special prompt branching needed
beyond guarding against an empty examples block.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name fro\\m config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
[blank — how do you extract the label and reasoning from the LLM's text output?
What string operations or parsing logic do you need?
This depends on the output format you chose in build_few_shot_prompt.]

The response is JSON, so:
1. Take response.choices[0].message.content and .strip() it.
2. If it's wrapped in a ```json ... ``` fence or has stray prose, slice from the
   first "{" to the last "}" so only the JSON object remains.
3. Call json.loads() on that substring to get a dict.
4. Read data["label"] and data["reasoning"]. Normalize the label with
   .strip().lower() before validating (handles "Interview", " interview ", etc.).
5. Wrap steps 2-4 in try/except (json.JSONDecodeError, KeyError, TypeError) so a
   malformed response falls through to label="unknown" rather than crashing.
```

---

**Step 4 — Validate the label:**

```
[blank — what do you do if the LLM returns a label that isn't in VALID_LABELS?
What should label be set to?]

After normalizing the parsed label (strip + lowercase), check whether it is in
VALID_LABELS. If it is, keep it. If it isn't — the model returned something like
"storytelling", an empty string, or extra words — set label to "unknown" and keep
whatever reasoning was returned. "unknown" is a value the UI and evaluation already
understand, so it never crashes; it just counts as an incorrect prediction.
```

---

**Step 5 — Handle errors gracefully:**

```
[blank — what could go wrong? (Network error? Unparseable response?)
What should the function return if something fails?
Hint: the evaluation loop runs 20 calls — one bad response shouldn't crash everything.]

Things that can fail:
- The API call itself: network error, timeout, rate limit, bad API key.
- The response isn't valid JSON, or is missing the "label"/"reasoning" keys.

Wrap the whole flow (LLM call + parse) in try/except. On any exception, return
a safe dict instead of raising:
    {"label": "unknown", "reasoning": "Error: <short message>"}

This guarantees classify_episode() always returns a dict with the right keys,
so the evaluation loop's 20 calls keep going even if one episode fails — a single
bad response can't crash the whole run.
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: [title]
Raw response text: [paste it here]
```

**How did you parse the label out of the response?**

```
[describe the string operations — strip, split, lower, etc.]
```

**Did any episodes return `"unknown"`? If so, why?**

```
[yes / no — if yes, what did the raw response look like?]
```

**One thing about the output format that surprised you:**

```
[your answer here]
```
