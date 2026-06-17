import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """
    # 1. Task instruction + the four label definitions (from the spec).
    instruction = (
        "You are classifying podcast episodes by their format. Classify the "
        "episode into exactly one of these four labels:\n\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — "
        "no guests, no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often "
        "debating or discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, "
        "archival audio, reporting — with a clear narrative arc"
    )

    # 2. The output format we ask for: a single JSON object, so classify_episode()
    #    can parse it reliably (see the spec's output-format decision).
    output_format = (
        "Respond with ONLY a JSON object, no markdown fences or extra text, "
        "in exactly this shape:\n"
        '{"label": "<one of: interview, solo, panel, narrative>", '
        '"reasoning": "<one or two sentences explaining the choice>"}'
    )

    parts = [instruction]

    # 3. The few-shot examples. If there are none, skip this block entirely
    #    (zero-shot fallback) instead of emitting an empty "Examples:" header.
    if labeled_examples:
        parts.append("Here are labeled examples. Study how each description maps to its label:")
        for ex in labeled_examples:
            parts.append(
                f"Title: {ex['title']}\n"
                f"Description: {ex['description']}\n"
                f"Label: {ex['label']}"
            )

    # 4. The new episode to classify — same format as the examples, but with the
    #    label left open instead of given.
    parts.append(
        "Now classify the following episode.\n\n"
        f"Description: {description}\n"
        "Label: ?"
    )

    # 5. Append the output-format instruction last, then join everything with a
    #    clear "---" delimiter between blocks.
    #  Resulting structure: instruction → --- → examples header → --- → example 1 → ... → --- → target episode → --- → output format
    parts.append(output_format)
    return "\n\n---\n\n".join(parts)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    TODO — Milestone 2 (complete after build_few_shot_prompt):

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.

    Before writing code, complete specs/classifier-spec.md.
    """
    try:
        # Step 1 — Build the prompt from the labeled examples + this description.
        prompt = build_few_shot_prompt(labeled_examples, description)

        # Step 2 — Send it to the LLM and pull out the response text.
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        response_text = response.choices[0].message.content

        # Step 3 — Parse the JSON out of the response. Slice from the first "{"
        # to the last "}" so any stray prose or ```json fences are dropped.
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}")
        data = json.loads(text[start:end + 1])

        label = str(data["label"]).strip().lower()
        reasoning = str(data.get("reasoning", "")).strip()

        # Step 4 — Validate the label. Anything not in VALID_LABELS becomes "unknown".
        if label not in VALID_LABELS:
            label = "unknown"

        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        # Step 5 — Any failure (network, bad JSON, missing keys) returns a safe
        # dict so one bad response can't crash the 20-call evaluation loop.
        return {"label": "unknown", "reasoning": f"Error: {e}"}
