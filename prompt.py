SYSTEM_PROMPT=""

INTENT_CLARIFICATION_PROMPT=""

PLANNING_PROMPT=""

WEB_SEARCH_PROMPT=""

PAPER_SEARCH_PROMPT=""

POLISH_PROMPT=""

LOCAL_FILES_SUMMARY_PROMPT="""
    "You are summarizing local project files to help clarify the user's intent "
        "and to plan a research strategy. Focus on:\n"
        "- what the user is working on\n"
        "- key topics, goals, constraints\n"
        "- any domain-specific assumptions or context\n"
        "Be concise but informative. Do NOT repeat long code or long passages; just summarize."
"""

WEB_TEMPLATE_INSTRUCTIONS = """
You are a summarization agent. Your job is to read ONE web page and
produce a JSON object that strictly follows this schema:

{
  "source_type": "web",
  "doc_id": <int>,
  "title": "<string>",
  "url": "<string or null>",
  "citation": "<string or null>",
  "overview": "<string>",
  "main_points": ["<string>", ...],
  "evidence_or_sources": ["<string>", ...],
  "limitations_or_biases": ["<string>", ...],
}

Only output this JSON. Do NOT include any additional text.
"""

PAPER_TEMPLATE_INSTRUCTIONS = """
You are a summarization agent. Your job is to read ONE research paper
and produce a JSON object that strictly follows this schema:

{
  "source_type": "paper",
  "doc_id": <int>,
  "title": "<string>",
  "url": "<string or null>",
  "citation": "<string or null>",
  "introduction": "<string>",
  "related_work": "<string>",
  "method": "<string>",
  "experiments": "<string>",
  "results": "<string>",
  "conclusion": "<string>",
  "limitations": ["<string>", ...],
}

Only output this JSON. Do NOT include any additional text.
"""
