"""
Workflow Parser for DRBench DRFlow Evaluation.

Extracts personalized workflow steps, flowcharts, and reference maps
from generated research report text.
"""

import json
import re
from typing import Dict, List, Optional

_JSON_BLOCK_PATTERNS = [
    r"```json\s*\n(.*?)\n\s*```",
    r"```\s*\n(\[.*?\])\n\s*```",
]


def _repair_json_common_issues(s: str) -> str:
    """
    Fix frequent LLM/typo issues that break strict JSON parsing.

    - Stray ')' immediately after a closing double-quote before , } or ]
      (e.g. ...\"meet.\")\\n  }).
    - Trailing commas before } or ].
    """
    s = re.sub(r'"\)\s*(?=[,}\]])', '"', s)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _try_parse_workflow_json_array(raw: str) -> Optional[List[Dict]]:
    """Parse JSON array of step dicts; try light repairs if the first load fails."""
    raw = raw.strip()
    for candidate in (raw, _repair_json_common_issues(raw)):
        try:
            steps = json.loads(candidate)
            if isinstance(steps, list):
                return steps
        except json.JSONDecodeError:
            continue
    return None


def _load_steps_from_section_text(
    section_text: str, raw_array_fallback: bool
) -> List[Dict]:
    """Extract first valid workflow JSON array from a markdown section."""
    for jp in _JSON_BLOCK_PATTERNS:
        json_match = re.search(jp, section_text, re.DOTALL)
        if json_match:
            steps = _try_parse_workflow_json_array(json_match.group(1))
            if steps is not None:
                return steps

    if raw_array_fallback:
        raw_json = re.search(r"(\[\s*\{.*?\}\s*\])", section_text, re.DOTALL)
        if raw_json:
            steps = _try_parse_workflow_json_array(raw_json.group(1))
            if steps is not None:
                return steps

    return []


def extract_personalized_workflow_steps(report_text: str) -> List[Dict]:
    """
    Extract personalized workflow steps JSON from report text.

    Looks for sections titled 'Personalized Workflow Steps' (with various heading levels)
    and parses the embedded JSON code block.

    Args:
        report_text: Full report text in markdown format.

    Returns:
        List of workflow step dictionaries, each containing keys like
        'item', 'title', 'task', 'task_description', 'subtask_description',
        'prerequisite', 'prerequisite_description'.
    """
    section_patterns = [
        r'#{2,4}\s*Personalized\s+Workflow\s+Steps',
        r'#{2,4}\s*Personalized\s+Recommended\s+Workflow\s+Steps',
        r'#{2,4}\s*Recommended\s+Workflow\s+Steps',
        r'#{2,4}\s*Workflow\s+Steps',
        r'#{2,4}\s*Personalized\s+Workflow\s+Recommendations',
    ]

    section_start = None
    for pattern in section_patterns:
        match = re.search(pattern, report_text, re.IGNORECASE)
        if match:
            section_start = match.end()
            break

    if section_start is None:
        return []

    remaining_text = report_text[section_start:]
    next_heading = re.search(r'\n#{2,4}\s+', remaining_text)
    if next_heading:
        section_text = remaining_text[:next_heading.start()]
    else:
        section_text = remaining_text

    return _load_steps_from_section_text(section_text, raw_array_fallback=True)


def extract_generic_workflow_steps(report_text: str) -> List[Dict]:
    """
    Extract generic (non-personalized) workflow steps JSON from report text.

    Args:
        report_text: Full report text in markdown format.

    Returns:
        List of generic workflow step dictionaries.
    """
    section_patterns = [
        r'#{2,4}\s*Generic\s+Workflow\s+Steps',
        r'#{2,4}\s*Enterprise\s+Workflow\s+Steps',
    ]

    section_start = None
    for pattern in section_patterns:
        match = re.search(pattern, report_text, re.IGNORECASE)
        if match:
            section_start = match.end()
            break

    if section_start is None:
        return []

    remaining_text = report_text[section_start:]
    next_heading = re.search(r'\n#{2,4}\s+', remaining_text)
    if next_heading:
        section_text = remaining_text[:next_heading.start()]
    else:
        section_text = remaining_text

    return _load_steps_from_section_text(section_text, raw_array_fallback=False)


def extract_personalized_flowchart(report_text: str) -> str:
    """
    Extract personalized flowchart mermaid code from report text.

    Args:
        report_text: Full report text in markdown format.

    Returns:
        Mermaid diagram code as a string, or empty string if not found.
    """
    section_patterns = [
        r'#{2,4}\s*Personalized\s+Flowchart',
        r'#{2,4}\s*Personalized\s+Recommended\s+Flowchart',
        r'#{2,4}\s*Recommended\s+Flowchart',
    ]

    section_start = None
    for pattern in section_patterns:
        match = re.search(pattern, report_text, re.IGNORECASE)
        if match:
            section_start = match.end()
            break

    if section_start is None:
        return ""

    remaining_text = report_text[section_start:]
    mermaid_match = re.search(
        r'```mermaid\s*\n(.*?)\n\s*```', remaining_text, re.DOTALL
    )

    if mermaid_match is None:
        return ""

    return mermaid_match.group(1).strip()


def extract_condition_resolution_table(report_text: str) -> List[Dict]:
    """
    Extract condition resolution table from report text.

    The table has columns: Generic Condition | User Evidence Found | Resolution

    Args:
        report_text: Full report text in markdown format.

    Returns:
        List of dicts with keys 'condition', 'evidence', 'resolution'.
    """
    section_patterns = [
        r'#{2,4}\s*Condition\s+Resolution\s+Table',
    ]

    section_start = None
    for pattern in section_patterns:
        match = re.search(pattern, report_text, re.IGNORECASE)
        if match:
            section_start = match.end()
            break

    if section_start is None:
        return []

    remaining_text = report_text[section_start:]
    next_heading = re.search(r'\n#{2,4}\s+', remaining_text)
    if next_heading:
        section_text = remaining_text[:next_heading.start()]
    else:
        section_text = remaining_text

    rows = []
    table_rows = re.findall(
        r'\|([^|]+)\|([^|]+)\|([^|]+)\|', section_text
    )
    for row in table_rows:
        condition = row[0].strip()
        evidence = row[1].strip()
        resolution = row[2].strip()
        if condition.startswith('-') or condition == 'Generic Condition':
            continue
        rows.append({
            'condition': condition,
            'evidence': evidence,
            'resolution': resolution,
        })

    return rows


def extract_references_map(report_text: str) -> Dict[str, str]:
    """
    Extract reference number to citation description mapping from report text.

    Extracts the full reference line (not just the bold title) so that the
    citation normalizer can resolve enterprise chat (User/Channel) and email
    (sender/date) citations for factuality scoring.

    Handles formats like:
        [^1]: **filename.txt** - Internal Document (path)
        [^2]: **Mattermost Message** - Enterprise Chat (User: jwang, Channel: ...)
        [^3]: **Email Subject** - Email from sender@domain.com on 2024-01-01

    Args:
        report_text: Full report text in markdown format.

    Returns:
        Dict mapping reference number string (e.g., "1") to full citation description.
    """
    references = {}

    ref_pattern = r'\[\^(\d+)\]:\s*(.+?)$'
    matches = re.finditer(ref_pattern, report_text, re.MULTILINE)

    for match in matches:
        ref_num = match.group(1)
        ref_text = match.group(2).strip()
        ref_text = re.sub(r'\*\*', '', ref_text)
        references[ref_num] = ref_text

    if not references:
        ref_pattern_alt = r'\[\^(\d+)\]:\s*([^\n]+)'
        matches = re.finditer(ref_pattern_alt, report_text, re.MULTILINE)
        for match in matches:
            ref_num = match.group(1)
            ref_text = match.group(2).strip()
            ref_text = re.sub(r'\*\*', '', ref_text)
            references[ref_num] = ref_text

    return references


def extract_step_citations(
    step_text: str, references_map: Dict[str, str]
) -> List[str]:
    """
    Extract citation file names from workflow step text using reference map.

    Finds [^N] style citation markers and resolves them to file names.

    Args:
        step_text: Text from a workflow step (task_description, etc.)
        references_map: Mapping from reference numbers to file names.

    Returns:
        Deduplicated list of cited file names.
    """
    citations = []

    ref_pattern = r'\[\^(\d+)\]'
    matches = re.finditer(ref_pattern, step_text)

    for match in matches:
        ref_num = match.group(1)
        if ref_num in references_map:
            citations.append(references_map[ref_num])

    ref_pattern_alt = r'\[(\d+)\]'
    matches_alt = re.finditer(ref_pattern_alt, step_text)
    for match in matches_alt:
        ref_num = match.group(1)
        if ref_num in references_map and references_map[ref_num] not in citations:
            citations.append(references_map[ref_num])

    return list(dict.fromkeys(citations))


def parse_workflow_from_report(
    report_text: str,
) -> Dict:
    """
    Parse all workflow-related data from a report text.

    This is the main entry point for extracting workflow data from a report.

    Args:
        report_text: Full report text in markdown format.

    Returns:
        Dict with keys:
            - 'workflow_steps': list of personalized workflow step dicts
            - 'generic_workflow_steps': list of generic workflow step dicts
            - 'workflow_flowchart': mermaid flowchart string
            - 'references_map': dict mapping ref numbers to file names
            - 'condition_resolution_table': list of condition resolution dicts
            - 'report_text': the original report text
    """
    workflow_steps = extract_personalized_workflow_steps(report_text)
    generic_steps = extract_generic_workflow_steps(report_text)
    flowchart = extract_personalized_flowchart(report_text)
    references_map = extract_references_map(report_text)
    condition_table = extract_condition_resolution_table(report_text)

    return {
        'workflow_steps': workflow_steps,
        'generic_workflow_steps': generic_steps,
        'workflow_flowchart': flowchart,
        'references_map': references_map,
        'condition_resolution_table': condition_table,
        'report_text': report_text,
    }


def format_predicted_steps_text(workflow_steps: List[Dict]) -> str:
    """
    Format predicted workflow steps as readable text for LLM prompts.

    Args:
        workflow_steps: List of workflow step dictionaries.

    Returns:
        Formatted text listing all steps.
    """
    if not workflow_steps:
        return "No predicted workflow steps found."

    lines = []
    for step in workflow_steps:
        item = step.get('item', '?')
        title = step.get('title', 'N/A')
        task = step.get('task', 'N/A')
        desc = step.get('task_description', 'N/A')
        subtask = step.get('subtask_description', '')
        prereq = step.get('prerequisite', '')

        lines.append(f"Step {item}: {title}")
        lines.append(f"  Task: {task}")
        lines.append(f"  Description: {desc}")
        if subtask:
            lines.append(f"  Subtask/Conditions: {subtask}")
        if prereq:
            lines.append(f"  Prerequisite: Step {prereq}")
        lines.append("")

    return "\n".join(lines)


def format_golden_steps_text(eval_data: Dict) -> str:
    """
    Format golden workflow steps from eval config as readable text for LLM prompts.

    Args:
        eval_data: Evaluation config with 'dr_report_evaluation_qa' key.

    Returns:
        Formatted text listing all golden steps.
    """
    qa_list = eval_data.get("dr_report_evaluation_qa", [])
    golden_steps = [
        qa for qa in qa_list if qa.get("type") == "workflow_step"
    ]

    if not golden_steps:
        return "No golden workflow steps found."

    lines = []
    for step in golden_steps:
        step_id = step.get('id', 'N/A')
        title = step.get('title', 'N/A')
        task = step.get('task', 'N/A')
        answer = step.get('answer', 'N/A')
        conditions = step.get('conditions', '')

        lines.append(f"{step_id}: {title}")
        lines.append(f"  Task: {task}")
        lines.append(f"  Expected Answer: {answer[:300]}...")
        if conditions:
            lines.append(f"  Conditions: {conditions[:200]}...")
        lines.append("")

    return "\n".join(lines)
