# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
[blank — write out the accuracy formula in plain English.
 What counts as "correct"? What do you divide by?]

accuracy = (number of correct predictions) / (total number of predictions)

A prediction is "correct" when predictions[i] is exactly equal to
ground_truth[i] (same string, same position). We divide by the total number
of predictions (= len(predictions)). The result is a float between 0.0 and 1.0.
```

---

**Step-by-step logic:**

```
[blank — describe the steps your code will take.
 1. ...
 2. ...
 3. ...]

1. If predictions is empty, return 0.0 (avoid dividing by zero).
2. Count how many positions i have predictions[i] == ground_truth[i].
   (e.g. sum(p == t for p, t in zip(predictions, ground_truth)))
3. Divide that count by len(predictions) and return it as a float.
```

---

**Edge case — what if both lists are empty?**

```
[blank — what should the function return? Why?]

Return 0.0. With no predictions there are zero correct out of zero total, so
the fraction is undefined — dividing would raise ZeroDivisionError. Returning
0.0 is a safe, sensible default (nothing was classified correctly).
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

[blank — what does compute_accuracy() return for these inputs? Show your work.]

Compare position by position:
  i=0: interview == interview  ✓ correct
  i=1: solo      == solo       ✓ correct
  i=2: panel     != solo       ✗ wrong
  i=3: interview != narrative  ✗ wrong

correct = 2, total = 4
accuracy = 2 / 4 = 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
[blank — be precise. When does an episode count as correctly classified
 for the "interview" class, for example?]

"correct" for a class is counted only over episodes whose GROUND TRUTH is that
class. An episode counts as correct for "interview" when its ground_truth is
"interview" AND the prediction is also "interview".

Key point: group by ground truth, not by prediction. A panel episode wrongly
predicted as interview is a miss for panel — it does NOT count toward interview.
```

---

**What does "total" mean for a given class?**

```
[blank — is "total" the total number of predictions, or something more specific?]

"total" for a class is the number of episodes whose GROUND TRUTH is that class —
NOT the total number of predictions and NOT how many times the class was
predicted. So total for "interview" = how many test episodes are truly interview.
Summing "total" across all four classes equals len(ground_truth).
```

---

**Step-by-step logic:**

```
[blank — describe the steps your code will take.
 1. Initialize ...
 2. Loop over ...
 3. For each pair (predicted, truth) ...
 4. After the loop ...
 5. Return ...]

1. Initialize a stats dict for every label in VALID_LABELS:
   {label: {"correct": 0, "total": 0, "accuracy": 0.0}}
2. Loop over the (predicted, truth) pairs together — zip(predictions, ground_truth).
3. For each pair, use TRUTH to pick the bucket:
     - increment stats[truth]["total"] by 1
     - if predicted == truth, also increment stats[truth]["correct"] by 1
   (Guard: only do this if truth is in VALID_LABELS.)
4. After the loop, for each label compute accuracy = correct / total,
   but set accuracy = 0.0 when total == 0 (no division by zero).
5. Return the stats dict.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
[blank — what should accuracy be set to? Why?
 Hint: look at the docstring in evaluate.py.]

Set accuracy to 0.0 (the docstring in evaluate.py specifies "0.0 if total is 0").
With zero episodes of that class there's nothing to divide, and computing
correct / total would raise ZeroDivisionError. 0.0 keeps the return shape
consistent and signals "no data for this class" without crashing.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

Group each pair by its ground-truth label:
  i=0: truth=interview, pred=interview  → interview: total+1, correct+1
  i=1: truth=solo,      pred=interview  → solo:      total+1 (wrong)
  i=2: truth=solo,      pred=solo       → solo:      total+1, correct+1
  i=3: truth=panel,     pred=panel      → panel:     total+1, correct+1
  i=4: truth=narrative, pred=panel      → narrative: total+1 (wrong)

label       correct  total  accuracy
----------  -------  -----  --------
interview      1       1      1.0
solo           1       2      0.5
panel          1       1      1.0
narrative      0       1      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

Overall accuracy is an average that hides distribution — a model can score 75% while getting one entire class 0% right. Per-class accuracy shows you which class is failing, so you know exactly where to fix the prompt or labels.

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

The panel-vs-interview boundary is underspecified in your prompt or your panel examples are weak, so the model can't tell multi-guest roundtables from host+guest conversations. When uncertain, it defaults to the more familiar label (interview). Fix: sharpen the panel definition and pick clearer panel examples.

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?

More training examples (20→100) help only if they add diversity, with diminishing returns — and since all examples go in the prompt, it gets slower and can dilute the signal. More test episodes (200) don't change the classifier; they just make your accuracy numbers far more reliable and reveal real error patterns instead of noise from a tiny 5-per-class set.
