import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from drflow.agents.utils import prompt_llm
from drflow.agents.drflow_agent.vector_store import VectorStore
from .base import ResearchContext
from .citation_registry import UnifiedCitationRegistry

logger = logging.getLogger(__name__)


class FlowchartAssembler:
    """Assembler for personalized recommended flowcharts with comprehensive citation support"""

    def __init__(self, model: str, vector_store: Optional[VectorStore] = None):
        self.model = model
        self.vector_store = vector_store
        self.citation_registry = UnifiedCitationRegistry()
        self.is_local_files = False
        self.evidence_metadata = {
            "document_ids": [],
            "evidence_map": {"web_sources": [], "internal_sources": [], "research_findings": []},
        }

    def generate_personalized_flowchart(
        self,
        context: ResearchContext,
        action_plan=None,
        flowchart_search_scope: Literal["whole_vector_db", "action_plan_only"] = "whole_vector_db",
    ) -> str:
        """
        Generate a personalized flowchart using a three-phase approach:
        1. Collect ALL scenarios (generic rules + personal evidence)
        2. Build complete generic workflow with all conditional branches
        3. Prune/resolve the generic workflow based on personal information

        flowchart_search_scope:
            - "whole_vector_db": Search the entire vector store for content (default).
            - "action_plan_only": Use only documents/content from executed action plan results
              and context.findings; preserves references for citation without querying the
              full vector DB.
        """
        
        self.citation_registry = UnifiedCitationRegistry()
        
        if flowchart_search_scope == "action_plan_only":
            thematic_content = self._collect_content_from_action_plan(context, action_plan)
            logger.info(f"Phase 1 (action_plan_only): Collected {sum(len(items) for items in thematic_content.values())} items from action plan and findings")
        else:
            thematic_content = self._analyze_and_cluster_content(context)
            logger.info(f"Phase 1: Collected {sum(len(items) for items in thematic_content.values())} items across {len(thematic_content)} themes")
        
        all_docs = []
        global_sources = {}
        
        for theme, content_items in thematic_content.items():
            for item in content_items:
                doc_id = item.get("doc_id", "")
                if not doc_id:
                    continue
                
                source_info = self._extract_source_info(item)
                source_type = source_info.get("type", "unknown")
                title = source_info.get("title", "")
                
                # When action_plan_only, do not filter unknowns: use fallback so citations still work
                if flowchart_search_scope == "action_plan_only" and (
                    source_type in ["unknown", "research", "skip_citation"]
                    or title in ["Unknown", "Research Finding", "Unknown Source", "Untitled", ""]
                ):
                    source_info = self._source_info_for_action_plan_item(item, doc_id)
                    source_type = source_info.get("type", "internal")
                    title = source_info.get("title", doc_id)
                elif source_type in ["unknown", "research", "skip_citation"]:
                    logger.debug(f"Filtering out source with type '{source_type}': {doc_id}")
                    continue
                elif title in ["Unknown", "Research Finding", "Unknown Source", "Untitled", ""]:
                    logger.debug(f"Filtering out source with unknown title: {doc_id}")
                    continue
                
                self.citation_registry.register_document(
                    doc_id=doc_id,
                    source_info=source_info,
                    document_type="regular"
                )
                global_sources[doc_id] = source_info
                
                content = item.get("content", "")
                if len(content) > 3000:
                    content = content[:3000] + "..."
                
                all_docs.append({
                    "doc_id": doc_id,
                    "content": content,
                    "theme": theme,
                    "source_type": source_type,
                    "title": title
                })
        
        for key, finding in context.findings.items():
            if not isinstance(finding, dict):
                continue
            if not finding.get("success") and flowchart_search_scope != "action_plan_only":
                continue

            results = finding.get("results", [])
            first_nested_filename = None
            first_nested_path = None

            if isinstance(results, list):
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    rid = r.get("doc_id") or r.get("id") or ""
                    if not rid or rid in global_sources:
                        continue
                    r_meta = r.get("metadata", {})
                    r_path = r_meta.get("file_path") or r_meta.get("path") or ""
                    r_filename = (
                        r_meta.get("filename")
                        or r_meta.get("file_name")
                        or (Path(r_path).name if r_path else "")
                    )
                    if not first_nested_filename and r_filename and "." in r_filename and not r_filename.startswith("Search"):
                        first_nested_filename = r_filename
                        first_nested_path = r_path
            elif isinstance(results, dict):
                local_docs = results.get("local_documents") or []
                if isinstance(local_docs, list):
                    for doc in local_docs:
                        if not isinstance(doc, dict):
                            continue
                        fp = doc.get("file_path", "")
                        if fp and not first_nested_filename:
                            first_nested_filename = Path(fp).name
                            first_nested_path = fp

            doc_id = finding.get("doc_id", key)
            if doc_id in global_sources:
                continue

            tool_name = (finding.get("tool") or finding.get("metadata", {}).get("tool_used") or "").lower()
            unpacked_items = []
            if "enterprise" in tool_name:
                unpacked_items = self._unpack_enterprise_service_results(finding)

            if unpacked_items:
                for item in unpacked_items:
                    iid = item["doc_id"]
                    if iid in global_sources:
                        continue
                    item_source_info = self._extract_source_info({"metadata": item["metadata"]})
                    self.citation_registry.register_document(
                        doc_id=iid,
                        source_info=item_source_info,
                        document_type="regular",
                    )
                    global_sources[iid] = item_source_info
                    all_docs.append({
                        "doc_id": iid,
                        "content": item["content"][:3000],
                        "theme": "findings",
                        "source_type": item_source_info.get("type", "internal"),
                        "title": item_source_info.get("title", iid),
                    })
                continue

            source_info = self._extract_source_info_from_finding(finding)
            source_type = source_info.get("type", "unknown")
            title = source_info.get("title", "")

            if flowchart_search_scope == "action_plan_only" and (
                source_type in ["unknown", "research", "skip_citation"]
                or title in ["Unknown", "Research Finding", "Unknown Source", "Untitled", ""]
            ):
                source_info = self._source_info_from_finding_for_citation(finding, doc_id)
                source_type = source_info.get("type", "internal")
                title = source_info.get("title", doc_id)
            elif source_type in ["unknown", "research", "skip_citation"]:
                continue
            elif title in ["Unknown", "Research Finding", "Unknown Source", "Untitled", ""]:
                continue

            if first_nested_filename and (
                title.startswith("Search ") or ":" in title or len(title) > 80
            ):
                source_info = dict(source_info)
                source_info["title"] = first_nested_filename
                if first_nested_path:
                    source_info["path"] = first_nested_path
                    source_info["type"] = "internal"
                    source_type = "internal"
                title = first_nested_filename

            self.citation_registry.register_document(
                doc_id=doc_id,
                source_info=source_info,
                document_type="regular"
            )
            global_sources[doc_id] = source_info

            content = finding.get("synthesis", finding.get("summary", finding.get("description", "")))
            if not content:
                if isinstance(results, dict):
                    content = results.get("synthesis", str(results)[:800])
                elif isinstance(results, list) and results:
                    content = str(results[0])[:800]

            all_docs.append({
                "doc_id": doc_id,
                "content": str(content)[:3000] if content else "",
                "theme": "findings",
                "source_type": source_type,
                "title": title
            })
        
        logger.info(f"Phase 2: {len(all_docs)} valid documents for citation (after filtering unknowns)")
        logger.info(f"Registered {len(self.citation_registry.documents)} sources in citation registry")
        
        if not all_docs:
            logger.warning("No valid documents available for citations")
            return self._generate_fallback_report(context)
        
        id_map = {}
        for i, doc in enumerate(all_docs):
            simple_id = f"src_{i+1}"
            id_map[simple_id] = doc["doc_id"]
            doc["simple_id"] = simple_id

        findings_text = self._build_findings_text(all_docs)
        sources_reference = json.dumps(global_sources, indent=2, default=str)
        
        prompt = self._build_three_phase_prompt(
            context=context,
            all_docs=all_docs,
            findings_text=findings_text,
            sources_reference=sources_reference
        )

        try:
            flowchart_report = prompt_llm(model=self.model, prompt=prompt)
            flowchart_report = self._strip_existing_references(flowchart_report)
            flowchart_report = self._normalize_doc_citations(flowchart_report)
            flowchart_report = self._resolve_simplified_ids(flowchart_report, id_map)

            citation_count = self._count_doc_citations(flowchart_report)
            logger.info(f"Initial citation count: {citation_count}")
            
            if citation_count < 3:
                logger.info("Insufficient citations; attempting repair")
                flowchart_report = self._repair_citations(
                    flowchart_report=flowchart_report,
                    findings_text=findings_text,
                    available_doc_ids=[doc["doc_id"] for doc in all_docs],
                )
                logger.info(f"Post-repair citation count: {self._count_doc_citations(flowchart_report)}")

            flowchart_report = self._repair_personalized_citations(
                flowchart_report, all_docs
            )
            logger.info(
                f"Post-personalized-repair citation count: "
                f"{self._count_doc_citations(flowchart_report)}"
            )

            flowchart_report = self._validate_personalized_workflow(flowchart_report)

            final_report, citation_assignments = self.citation_registry.finalize_citations(flowchart_report)
            logger.info(f"Final citation assignments: {len(citation_assignments)}")

            if citation_assignments:
                references_section = self._generate_filtered_references()
                final_report += "\n" + references_section
            else:
                final_report += "\n\n## References\n\nNo references cited."

            return final_report
        except Exception as e:
            logger.error(f"Error generating flowchart: {e}")
            return f"Error generating personalized flowchart: {e}"

    def _build_findings_text(self, all_docs: List[Dict]) -> str:
        """Build formatted findings text for the prompt"""
        findings_text = ""
        for doc in all_docs:
            sid = doc.get("simple_id", doc["doc_id"])
            findings_text += f"--- START OF DOCUMENT {sid} ---\n"
            findings_text += f"SOURCE TYPE: {doc['source_type']}\n"
            findings_text += f"TITLE: {doc['title']}\n"
            findings_text += f"CONTENT:\n{doc['content']}\n"
            findings_text += f"--- END OF DOCUMENT {sid} ---\n\n"
        return findings_text

    def _build_three_phase_prompt(self, context: ResearchContext,
                                   all_docs: List[Dict], findings_text: str, sources_reference: str) -> str:
        """
        Build prompt that implements three-phase workflow generation:
        1. Collect all scenarios
        2. Build complete generic workflow
        3. Prune based on personalization
        """
        
        doc_id_lines = []
        for doc in all_docs:
            sid = doc.get("simple_id", doc["doc_id"])
            doc_id_lines.append(f"  {sid} = {doc['title']}")
        doc_ids_block = "\n".join(doc_id_lines)
        
        return f"""
You are an expert enterprise workflow architect. Generate a COMPREHENSIVE workflow report with **10-15** detailed, granular steps.

DR Question: "{context.original_question}"
User situation and persona: Infer from the DR question and from the retrieved documents (emails, chats, internal files) below.

### AVAILABLE SOURCES (use these IDs for citations):
{doc_ids_block}

### DOCUMENT CONTENTS:
{findings_text}

### SOURCE REFERENCE:
{sources_reference}

### CITATION FORMAT:
Use `[DOC:src_N]` format where src_N is a source ID from the list above (e.g., [DOC:src_1], [DOC:src_3]).
Cite policy documents for workflow rules and personal documents for user-specific facts.

**FACTUALITY REQUIREMENT**: In the Generic Workflow Steps and Personalized Workflow Steps (Section 1 and 2), cite ONLY src_N IDs that correspond to **internal sources** (policy documents, local files, emails, internal documents). Do NOT cite IDs that correspond to web or external search results in workflow step content.

---

# Research Report: Personalized Workflow Analysis

## Executive Summary
Summarize the workflow analysis citing key policy and personal evidence sources [DOC:src_N].

## 1. Generic Enterprise Workflow

### Overview
Extract a COMPREHENSIVE workflow from the policy documents. You MUST include EVERY distinct procedural step - aim for **10-15 detailed steps** covering the COMPLETE process from start to finish. Each step should represent a SINGLE discrete action (e.g., "Complete intake form" is one step, "Run gate check" is another, "Configure settings" is another). Do NOT merge multiple actions into one step.

### Generic Workflow Steps
```json
[
  {{
    "item": 1,
    "title": "Step title (e.g., 'Create X record', 'Complete intake form XYZ', 'Run gate check')",
    "task": "Brief task (e.g., 'Create a new onboarding workspace and attach handoff notes') [DOC:src_N]",
    "task_description": "DETAILED description (3-5 sentences) explaining exactly what to do, what specific fields to fill, what systems/tools/forms to use, what data to capture. Include ALL necessary details: form names or codes, system/tool names, specific fields, required attachments, who needs to be involved. [DOC:src_N]",
    "subtask_description": "For ANY step that has a decision/branch/conditional in the policy (if X then A, else B): you MUST put the FULL condition text HERE with ALL branches explicitly listed (e.g., 'If X → select Y; else select Z.'). For steps with no branch, use empty string.",
    "prerequisite": "Item number of prerequisite step (e.g., '1') or empty string if this is the first step or has no dependency",
    "prerequisite_description": "Why the prerequisite must be completed first (e.g., 'The workspace record must exist before the intake form can be submitted')"
  }},
  {{
    "item": 2,
    "title": "Next step title",
    ...
  }}
  // Continue for ALL steps (aim for 10-15 total)
]
```

### GENERIC WORKFLOW REQUIREMENTS (CRITICAL):
- Extract EVERY distinct step from ALL policy documents - do NOT combine or skip steps
- Each step should be a SINGLE discrete action (creating a record, filling a form, running a check, configuring a setting, scheduling meeting, etc.)
- Include the FULL end-to-end process relevant to the DR question
- Keep descriptions DETAILED with specific form names, system names, field names, tool names mentioned in the policies
- CRITICAL: For EVERY step that involves a decision or branch or conditional (if/then/else, tier-based, count-based, type-based), you MUST fill subtask_description with the COMPLETE condition text listing ALL branches. Do NOT leave subtask_description empty for conditional steps.
- Look for conditions triggered by: tier/level selection, user counts/thresholds, feature or integration requirements, package types, approval status, validation outcomes
- Include prerequisites that show step dependencies (which step must complete before this one)
- **TARGET: 10-15 steps** (minimum 10 unless policy truly has fewer discrete actions; maximum 15).

### Generic Flowchart
```mermaid
graph TD
  A[Step 1] --> B[Step 2]
  B --> C{{Decision Point}}
  C -->|Condition X| D[Path X]
  C -->|Condition Y| E[Path Y]
  D --> F[Continue...]
```

## 2. Personalized Recommended Workflow

### User Situation Analysis
Extract key facts from user's personal evidence [DOC:src_N]:
- Specific entities mentioned (vendors, systems, people)
- Dates and timeframes
- Data types involved
- Constraints or special circumstances

**IMPORTANT**: Carefully search through ALL documents (especially emails, chats, and user-provided files) to extract user-specific facts. Look for:

**Source Types to Search:**
- **enterprise_email**: User's emails about the incident, vendor communications
- **enterprise_chat**: Mattermost/Slack/Teams messages about the situation
- **internal documents**: User's local files with specific details
- **persona block**: User role, responsibilities, constraints

**Facts to Extract** [DOC:src_N]:
- Specific entities: vendor names, system names, people involved
- Dates and timeframes: discovery date, incident timing, deadlines
- Data details: types of data, encryption status, affected counts
- Access facts: confirmed/denied access, subcontractor involvement
- Constraints: operational limits, survey windows, downtime restrictions
- User role: who they are, what they can approve

**List ALL facts found in documents:**
(Be explicit - list each fact found and cite the source document)

### Condition Resolution Table
For EACH step in the Generic Workflow that has a non-empty subtask_description, create ONE row below. Use the EXACT condition text from that step. Search ALL user evidence documents to determine which branch applies.

| Generic Condition (from subtask_description above) | User Evidence Found | Resolution |
|----------------------------------------------------|---------------------|------------|
| [Copy EXACT condition from step N's subtask_description] | [Specific user fact with DOC:src_N] or "No evidence found" | Resolved: [specific path chosen] OR Unresolved: keep condition |

**RESOLUTION RULES:**
- Search ALL available documents (emails, chats, user files) for evidence that determines which branch applies
- If user evidence clearly indicates which branch: mark "Resolved: [path]" and cite the evidence
- If no evidence found or evidence is ambiguous: mark "Unresolved: keep condition"
- You MUST have one row per conditional step - do not skip any

### Personalized Workflow Steps
**MANDATORY CITATIONS**: Every step MUST include [DOC:src_N] in task, task_description, AND justification. Use policy source IDs for process steps and personal source IDs (emails, user files) for user-specific facts. No step may be left without citations.

```json
[
  {{
    "item": 1,
    "title": "Personalized step title with user-specific details",
    "task": "PERSONALIZED task incorporating specific user facts - use actual names, numbers, systems from user evidence [DOC:src_N]",
    "task_description": "DETAILED personalized description using user's SPECIFIC facts. Replace generic terms with actual: names, system/tool names, counts, timeframes, tier or package names, deadlines, constraints from the evidence. Cite both policy and personal evidence. [DOC:src_N][DOC:src_M]",
    "subtask_description": "EMPTY string ('') if the Condition Resolution Table shows this condition as Resolved. KEEP the full conditional text if Unresolved.",
    "prerequisite": "Item number or empty",
    "prerequisite_description": "Same as generic or personalized if relevant",
    "justification": "Explain WHY this exact step is included in the personalized workflow for this user. State what policy requirement, user fact, or resolved condition makes the step necessary now; MUST end with [DOC:src_N] citations (policy and/or personal evidence)."
  }}
  // SAME NUMBER of steps as generic workflow. EVERY step: task, task_description, and justification MUST contain [DOC:src_N].
]
```

### PERSONALIZATION RULES:
1. **SAME NUMBER OF STEPS** as generic workflow
2. **PERSONALIZE task/task_description** with user's specific facts (dates, names, numbers, entities)
3. **Conditional branches**: If a condition in the generic workflow is RESOLVED by user's personal evidence, present the personalized workflow accordingly: use the single resolved path in the step title/task and leave subtask_description EMPTY. If the condition is NOT resolved by evidence, keep the conditional text in subtask_description so the user sees both branches.
4. **subtask_description**:
   - EMPTY ("") when user evidence resolves which path to take (condition is Resolved in table)
   - KEEP conditional text when evidence does NOT determine the path (condition is Unresolved)
5. **justification**:
   - Explain why this step belongs in the personalized workflow, not just what to do
   - Reference user's SPECIFIC facts that determine the applicable path
   - If the step is unconditional, explain what policy or user circumstance still makes it necessary
   - MUST cite both policy and personal evidence using [DOC:src_N]

### CITATION REQUIREMENTS FOR PERSONALIZED SECTIONS (MANDATORY):
- **Personalized Workflow Steps JSON**: Each of task, task_description, and justification MUST contain at least one [DOC:src_N]. Use valid source IDs from the AVAILABLE SOURCES list.
- **Section 3 (Step-by-Step Personalized Guidance)**: Each step's **Description** MUST end with [DOC:src_N]. At least one **Action Item** per step MUST cite a source, e.g. "[ ] Action detail [DOC:src_N]".

### EXAMPLE TRANSFORMATIONS:

**Generic step with condition (example):**
- task: "Select the appropriate option based on tier and size"
- subtask_description: "If tier is premium → select option A; else select option B. If count exceeds threshold → add additional pack; else keep default."

**Personalized (condition RESOLVED):**
- task: "Select option A and add the additional pack [DOC:src_1]"
- subtask_description: "" (empty because evidence resolved both conditions)
- justification: "Evidence shows premium tier and count above threshold [DOC:src_5], which triggers option A and the additional pack [DOC:src_1]."

**Personalized (condition NOT RESOLVED):**
- task: "Select option based on package type; determine which applies [DOC:src_1]"
- subtask_description: "If package is standard → path X. If premium → path Y."
- justification: "No evidence found specifying package type, so both branches must be preserved [DOC:src_1]."

### Personalized Flowchart
```mermaid
graph TD
  A[Personalized Step 1] --> B[Personalized Step 2]
  B --> C[Step 3]
  C --> D{{Unresolved Decision if any}}
```
Show resolved paths as direct arrows. Keep decision diamonds only for UNRESOLVED conditions.

## 3. Step-by-Step Personalized Guidance

For each step, provide. **You MUST add [DOC:src_N] citations** to every Description and to at least one Action Item per step (use valid source IDs from AVAILABLE SOURCES).

### Step N: [Title]
**Description**: Detailed personalized guidance. End with citation(s), e.g. "…per portal requirements [DOC:src_1]."
**Action Items**:
- [ ] Specific action with details [DOC:src_N]
- [ ] Additional actions as needed (cite where relevant)

## References
(Leave blank - auto-generated)

---

CRITICAL REQUIREMENTS:
1. Generate **10-15 DETAILED steps** in generic workflow - extract EVERY distinct procedural step from policy documents. If you have fewer than 10, you are merging steps; if you have more than 15, consolidate only where actions are truly atomic together.
2. For each step that has an if/then/else or decision or conditional in the policy, you MUST put that COMPLETE condition text in subtask_description with ALL branches listed
3. Condition Resolution Table: one row per conditional step, search ALL user evidence to resolve conditions
4. Personalized workflow has IDENTICAL number of steps as generic (10-15)
5. Personalize task/task_description with USER'S SPECIFIC FACTS (actual names, counts, systems, dates, constraints from evidence)
6. subtask_description in personalized: EMPTY when resolved by user evidence, KEEP condition text when NOT resolved
7. Every justification must reference specific user facts with citations
8. **CITATIONS (MANDATORY)**: Use [DOC:src_N] format. In the PERSONALIZED WORKFLOW (Section 2): every step's task, task_description, and justification MUST contain at least one [DOC:src_N]. In Section 3 (Step-by-Step Guidance): every step's Description and at least one Action Item per step MUST contain [DOC:src_N]. Only use source IDs from the AVAILABLE SOURCES list (src_1, src_2, etc.).
"""

    def _generate_filtered_references(self) -> str:
        """Generate references section, filtering out unknown sources"""
        if not self.citation_registry.citation_assignments:
            return "\n## References\n\nNo references cited."
        
        from collections import OrderedDict
        
        ordered_refs = OrderedDict()
        for doc_id, citation_num in sorted(self.citation_registry.citation_assignments.items(), key=lambda x: x[1]):
            if doc_id in self.citation_registry.documents:
                doc_info = self.citation_registry.documents[doc_id]
                if doc_info.document_type == "ai_synthesis":
                    continue
                
                source_info = doc_info.source_info
                source_type = source_info.get("type", "")
                title = source_info.get("title", "")
                
                if source_type in ["unknown", "research", "skip_citation"]:
                    continue
                if title in ["Unknown", "Research Finding", "Unknown Source", "Untitled", ""]:
                    continue
                
                ordered_refs[citation_num] = source_info
        
        if not ordered_refs:
            return "\n## References\n\nNo references cited."
        
        references = ["\n## References\n"]
        for citation_num, source_info in ordered_refs.items():
            ref_text = self._format_reference(citation_num, source_info)
            references.append(ref_text)
        
        return "\n\n".join(references)

    def _format_reference(self, citation_num: int, source_info: dict) -> str:
        """Format a single reference entry"""
        title = source_info.get("title", "Untitled")
        source_type = source_info.get("type", "")
        url = source_info.get("url", "")
        
        if len(title) > 200:
            title = title[:197] + "..."
        
        if source_type == "enterprise_email" or source_type == "email_message":
            sender = source_info.get("sender", source_info.get("from", "Unknown Sender"))
            to = source_info.get("to", "")
            date = source_info.get("date", "")
            if "<" in sender and ">" in sender:
                sender = sender.split("<")[1].split(">")[0]
            ref = f"[^{citation_num}]: **{title}** - Email from {sender}"
            if to:
                ref += f" to {to}"
            if date:
                ref += f" on {date}"
            return ref
        
        elif source_type == "enterprise_chat":
            user = source_info.get("user", "Unknown User")
            channel = source_info.get("channel", "Unknown Channel")
            team = source_info.get("team", "")
            location = f"Channel: {channel}"
            if team:
                location = f"Team: {team}, {location}"
            return f"[^{citation_num}]: **{title}** - Enterprise Chat (User: {user}, {location})"
        
        elif source_type == "internal":
            path = source_info.get("path", "")
            return f"[^{citation_num}]: **{title}** - Internal Document" + (f" (`{path}`)" if path else "")
        
        elif source_type == "external":
            if url and url != "Unknown URL":
                return f"[^{citation_num}]: **{title}** - Web Source ([{url}]({url}))"
            return f"[^{citation_num}]: **{title}** - Web Source"
        
        elif source_type == "enterprise_file":
            server = source_info.get("server", "Enterprise")
            path = source_info.get("path", "")
            return f"[^{citation_num}]: **{title}** - {server} File" + (f" (`{path}`)" if path else "")
        
        elif source_type == "enterprise_api":
            tool = source_info.get("source_tool", "API")
            return f"[^{citation_num}]: **{title}** - Enterprise API ({tool})"
        
        else:
            return f"[^{citation_num}]: **{title}** - Document"

    def _analyze_and_cluster_content(self, context: ResearchContext) -> Dict[str, List[Dict]]:
        """Analyze vector store content and cluster by themes"""
        
        if not self.vector_store:
            logger.warning("No vector store available")
            return {"general": []}

        try:
            search_queries = self._generate_search_queries(context.original_question, context.plan)
            all_content = {}

            logger.info(f"Generated {len(search_queries)} search queries")

            for query in search_queries:
                try:
                    results = self.vector_store.search(query, top_k=10, use_semantic=True)
                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")
                    continue
                    
                theme = self._extract_theme_from_query(query)
                if theme not in all_content:
                    all_content[theme] = []

                for result in results:
                    metadata = result.get("metadata", {})
                    score = result.get("similarity_score", 0)
                    
                    if score > 0.3:
                        doc_id = result.get("doc_id", result.get("id", ""))
                        if not doc_id:
                            continue
                            
                        all_content[theme].append({
                            "content": result.get("content", ""),
                            "metadata": metadata,
                            "score": score,
                            "doc_id": doc_id,
                            "theme": theme,
                        })

            for theme in all_content:
                all_content[theme] = self._deduplicate_content(all_content[theme])

            return all_content

        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")
            return {"general": []}

    def _collect_content_from_action_plan(
        self, context: ResearchContext, action_plan: Any
    ) -> Dict[str, List[Dict]]:
        """
        Build thematic content only from executed action plan results and context.findings.
        Used when flowchart_search_scope='action_plan_only' to preserve references for
        citation without querying the full vector DB.
        """
        thematic_content: Dict[str, List[Dict]] = {}
        seen_doc_ids: set = set()

        def add_item(theme: str, doc_id: str, content: str, metadata: dict, score: float = 0.5):
            if not doc_id or doc_id in seen_doc_ids:
                return
            seen_doc_ids.add(doc_id)
            if theme not in thematic_content:
                thematic_content[theme] = []
            content_snippet = (content or "")[:4000]
            thematic_content[theme].append({
                "content": content_snippet,
                "metadata": metadata or {},
                "score": score,
                "doc_id": doc_id,
                "theme": theme,
            })

        for key, finding in context.findings.items():
            if not isinstance(finding, dict):
                continue
            category = finding.get("category", "general")
            theme = "workflow" if "requirement" in str(category).lower() or "local" in str(category).lower() else "personal" if "evidence" in str(category).lower() or "enterprise" in str(category).lower() else "findings"
            doc_id = finding.get("doc_id") or key
            synthesis = finding.get("synthesis", finding.get("summary", finding.get("description", "")))
            if synthesis or finding.get("content"):
                content = synthesis or finding.get("content", "")
                add_item(theme, doc_id, str(content), finding.get("metadata", {}), 0.7)
            results = finding.get("results", [])
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict):
                        if "service" in r and isinstance(r.get("results"), list):
                            for inner in self._unpack_enterprise_service_results(finding):
                                add_item(theme, inner["doc_id"], inner["content"], inner["metadata"], 0.7)
                            break
                        rid = r.get("doc_id", r.get("id", ""))
                        if not rid:
                            continue
                        add_item(theme, rid, r.get("content", r.get("text", "")) or "", r.get("metadata", {}), float(r.get("similarity_score", r.get("score", 0.5))))
            elif isinstance(results, dict):
                local_docs = results.get("local_documents") or []
                if isinstance(local_docs, list):
                    for i, doc in enumerate(local_docs):
                        if not isinstance(doc, dict):
                            continue
                        file_path = doc.get("file_path") or ""
                        if not file_path:
                            continue
                        doc_id = f"{key}_doc_{i}"
                        content = doc.get("content_excerpt", doc.get("content", "")) or ""
                        meta = {
                            "file_path": file_path,
                            "filename": Path(file_path).name if file_path else "",
                            "source_type": "local_document",
                        }
                        add_item(theme, doc_id, content, meta, float(doc.get("relevance_score", 0.5)))
                for rid, r in results.items():
                    if isinstance(r, dict):
                        content = r.get("content", r.get("text", r.get("synthesis", "")))
                        add_item(theme, rid, str(content or ""), r.get("metadata", {}), 0.6)

        if action_plan and hasattr(action_plan, "actions"):
            for action in action_plan.actions:
                if action.status.value != "completed":
                    continue
                out = action.actual_output or {}
                if not isinstance(out, dict):
                    continue
                stage = getattr(action, "stage", None) or ""
                theme = "workflow" if "REQUIREMENT" in str(stage) or "GENERAL" in str(stage) else "personal" if "PERSONAL" in str(stage) else "findings"
                doc_id = out.get("doc_id", f"action_{action.id}")
                content = out.get("synthesis", out.get("summary", out.get("content", "")))
                if content:
                    add_item(theme, doc_id, str(content), out.get("metadata", {}), 0.7)
                results = out.get("results", [])
                if isinstance(results, list):
                    for r in results:
                        if isinstance(r, dict):
                            rid = r.get("doc_id", r.get("id", ""))
                            if rid:
                                add_item(theme, rid, r.get("content", r.get("text", "")) or "", r.get("metadata", {}), float(r.get("similarity_score", 0.5)))

        if not thematic_content:
            thematic_content = {"general": []}
        return thematic_content

    def _generate_search_queries(self, original_question: str, research_plan: Any) -> List[str]:
        """Generate search queries for content discovery - both policy and personal evidence"""
        
        queries = [original_question]
        
        if research_plan and isinstance(research_plan, dict):
            areas = research_plan.get("research_investigation_areas", [])
            for area in areas:
                if isinstance(area, dict):
                    focus = area.get("research_focus", "")
                    if focus:
                        queries.append(focus)
                    search_terms = area.get("search_terms", [])
                    queries.extend(search_terms[:3])
        
        queries.extend([
            f"step-by-step procedure {original_question}",
            f"workflow process {original_question}",
            f"policy playbook {original_question}",
            f"conditional decision branch {original_question}",
            f"gate check approval {original_question}",
            f"intake form configuration setup {original_question}",
        ])
        
        queries.extend([
            f"email {original_question}",
            f"user specific details requirements {original_question}",
        ])
        
        return queries[:15]

    def _extract_theme_from_query(self, query: str) -> str:
        """Extract theme from query"""
        query_lower = query.lower()
        if any(word in query_lower for word in ["workflow", "process", "procedure"]):
            return "workflow"
        elif any(word in query_lower for word in ["policy", "compliance", "requirement"]):
            return "requirements"
        elif any(word in query_lower for word in ["condition", "decision", "branch", "if"]):
            return "conditions"
        else:
            return "general"

    def _deduplicate_content(self, content_list: List[Dict]) -> List[Dict]:
        """Deduplicate and rank content"""
        seen = set()
        unique = []
        for item in content_list:
            h = hash(item.get("content", "")[:500])
            if h not in seen:
                seen.add(h)
                unique.append(item)
        unique.sort(key=lambda x: x.get("score", 0), reverse=True)
        return unique[:15]

    def _unpack_enterprise_service_results(self, finding: Dict) -> List[Dict]:
        """Unpack enterprise API service-level results into individual items with proper metadata.

        Enterprise API findings have results structured as:
        [{"service": "mattermost", "results": [post1, post2, ...]}, ...]

        Returns a list of dicts each with doc_id, content, and metadata suitable
        for _extract_source_info to classify as enterprise_chat / enterprise_email / enterprise_file.
        """
        items: List[Dict] = []
        results = finding.get("results", [])
        if not isinstance(results, list):
            return items

        for service_result in results:
            if not isinstance(service_result, dict):
                continue
            service_name = service_result.get("service", "")
            inner_results = service_result.get("results", [])
            if not isinstance(inner_results, list) or not service_name:
                continue

            for idx, inner in enumerate(inner_results):
                if not isinstance(inner, dict):
                    continue

                if service_name == "mattermost":
                    post_id = inner.get("id", inner.get("user_id", f"post_{idx}"))
                    doc_id = f"mattermost_post_{post_id}"
                    meta = {
                        "source": "mattermost",
                        "type": "mattermost_post",
                        "user_name": inner.get("user_name", inner.get("username", "")),
                        "channel_name": inner.get("channel_name", inner.get("channel", "")),
                        "team_name": inner.get("team_name", inner.get("team", "")),
                        "timestamp": inner.get("timestamp", inner.get("created_at", "")),
                    }
                    content = inner.get("message", inner.get("content", ""))
                    if not content:
                        content = f"Mattermost message from {meta['user_name']} in {meta['channel_name']}"

                elif service_name == "email_imap":
                    email_id = inner.get("id", f"email_{idx}")
                    doc_id = f"email_{email_id}"
                    to_raw = inner.get("to", inner.get("recipients", ""))
                    if isinstance(to_raw, list):
                        to_raw = ", ".join(str(t) for t in to_raw if t)
                    meta = {
                        "source": "email_imap",
                        "type": "email_message",
                        "sender": inner.get("from", inner.get("sender", "")),
                        "from": inner.get("from", inner.get("sender", "")),
                        "to": to_raw,
                        "subject": inner.get("subject", ""),
                        "date": inner.get("date", inner.get("timestamp", "")),
                    }
                    content = inner.get("preview", inner.get("content", inner.get("body", "")))
                    if not content:
                        content = f"Email from {meta['sender']}: {meta['subject']}"

                elif service_name in ("nextcloud", "filebrowser"):
                    file_path = inner.get("path", inner.get("file_path", ""))
                    file_name = inner.get("name", inner.get("filename", ""))
                    if not file_name and file_path:
                        file_name = Path(file_path).name
                    if not file_name:
                        continue
                    doc_id = f"{service_name}_{file_name}_{idx}"
                    meta = {
                        "source": service_name,
                        "type": f"{service_name}_file",
                        "file_name": file_name,
                        "filename": file_name,
                        "original_path": file_path,
                    }
                    content = inner.get("content", f"{service_name} file: {file_name}")
                else:
                    continue

                items.append({
                    "doc_id": doc_id,
                    "content": str(content)[:4000],
                    "metadata": meta,
                })

        return items

    def _extract_source_info(self, item: Dict) -> Dict[str, str]:
        """Extract source info from vector store result"""
        metadata = item.get("metadata", {})
        source = metadata.get("source", "")
        source_type = metadata.get("source_type", metadata.get("type", ""))
        
        if source == "mattermost" or source_type == "mattermost_post":
            if self.is_local_files:
                filename = (
                    metadata.get("filename")
                    or metadata.get("file_name")
                    or (Path(metadata.get("file_path", "")).name
                        if metadata.get("file_path") else "")
                )
                return {
                    "type": "internal",
                    "title": filename or "Mattermost Message",
                    "path": metadata.get("file_path", ""),
                }
            return {
                "type": "enterprise_chat",
                "title": "Mattermost Message",
                "user": metadata.get("user_name", metadata.get("username", "Unknown")),
                "channel": metadata.get("channel_name", "Unknown"),
                "team": metadata.get("team_name", ""),
            }
        elif source == "email_imap" or source_type == "email_message":
            if self.is_local_files:
                filename = (
                    metadata.get("filename")
                    or metadata.get("file_name")
                    or (Path(metadata.get("file_path", "")).name
                        if metadata.get("file_path") else "")
                )
                if filename:
                    return {
                        "type": "internal",
                        "title": filename,
                        "path": metadata.get("file_path", ""),
                    }
            subject = metadata.get("subject", metadata.get("title", ""))
            if not subject or subject == "Unknown":
                return {"type": "unknown", "title": "Unknown"}
            to_raw = metadata.get("to", metadata.get("recipients", ""))
            if isinstance(to_raw, list):
                to_raw = ", ".join(str(t) for t in to_raw if t)
            return {
                "type": "enterprise_email",
                "title": subject,
                "sender": metadata.get("sender", metadata.get("from", "")),
                "to": to_raw,
                "date": metadata.get("date", ""),
            }
        elif source_type == "local_document":
            filename = metadata.get("filename", metadata.get("relative_path", ""))
            if not filename or filename == "Unknown":
                return {"type": "unknown", "title": "Unknown"}
            return {
                "type": "internal",
                "title": filename,
                "path": metadata.get("file_path", ""),
            }
        elif source == "file":
            filename = metadata.get("filename", "")
            if not filename or filename == "Unknown":
                return {"type": "unknown", "title": "Unknown"}
            return {
                "type": "internal",
                "title": filename,
                "path": metadata.get("file_path", metadata.get("original_path", "")),
            }
        elif source == "url" or source_type == "search_result":
            title = metadata.get("title", "")
            if not title or title == "Unknown":
                return {"type": "unknown", "title": "Unknown"}
            return {
                "type": "external",
                "title": title,
                "url": metadata.get("url", ""),
            }
        elif source == "nextcloud" or source_type == "nextcloud_file":
            nc_path = metadata.get("nextcloud_path", metadata.get("path", metadata.get("original_path", "")))
            filename = metadata.get("nextcloud_filename", metadata.get("filename", metadata.get("file_name", "")))
            if not filename and nc_path:
                filename = Path(nc_path).name if "/" in nc_path else nc_path
            if filename and (filename.startswith("tmp") and len(filename) < 20) and nc_path:
                path_name = Path(nc_path).name if "/" in nc_path else ""
                if path_name and not path_name.startswith("tmp"):
                    filename = path_name
            if not filename or filename == "Unknown":
                return {"type": "unknown", "title": "Unknown"}
            return {
                "type": "enterprise_file",
                "title": filename,
                "path": nc_path,
                "server": "Nextcloud",
            }
        elif source == "filebrowser":
            filename = metadata.get("file_name", metadata.get("filename", ""))
            if not filename or filename == "Unknown":
                return {"type": "unknown", "title": "Unknown"}
            return {
                "type": "enterprise_file",
                "title": filename,
                "path": metadata.get("original_path", ""),
                "server": "FileBrowser",
            }
        else:
            title = metadata.get("title") or metadata.get("filename") or ""
            if title and title not in ["Unknown", "Research Finding", "Untitled", ""]:
                return {
                    "type": "internal",
                    "title": title,
                    "path": metadata.get("file_path", ""),
                }
            return {"type": "unknown", "title": "Unknown"}

    def _source_info_for_action_plan_item(self, item: Dict, doc_id: str) -> Dict[str, str]:
        """Fallback source info for action-plan items so citations match whole_vector_db format.

        Follows the report_tools.py approach: check explicit metadata fields
        first, then action_type keywords.
        """
        metadata = item.get("metadata", {})
        file_path = metadata.get("file_path") or metadata.get("path") or metadata.get("relative_path") or ""
        filename = metadata.get("filename") or (Path(file_path).name if file_path else "")
        source = metadata.get("source", "")
        source_type_meta = (metadata.get("source_type") or "").lower()

        if self.is_local_files and (
            source == "mattermost" or source_type_meta == "mattermost_post"
            or source == "email_imap" or source_type_meta == "email_message"
            or "chat" in (metadata.get("tool_used") or "").lower()
            or "email" in (metadata.get("tool_used") or "").lower()
        ):
            local_title = filename or doc_id
            return {"type": "internal", "title": local_title, "path": file_path}

        title = (
            filename
            or metadata.get("action_description")
            or metadata.get("description")
            or metadata.get("title")
            or ""
        )
        if isinstance(title, str) and len(title) > 120:
            title = title[:117] + "..."
        if not title or str(title).strip() in ["Unknown", "Research Finding", "Untitled", ""]:
            title = doc_id

        if source == "mattermost" or source_type_meta == "mattermost_post":
            return {"type": "enterprise_chat", "title": title, "user": metadata.get("user_name", ""), "channel": metadata.get("channel_name", ""), "team": metadata.get("team_name", "")}
        if source == "email_imap" or source_type_meta == "email_message":
            to_raw = metadata.get("to", metadata.get("recipients", ""))
            if isinstance(to_raw, list):
                to_raw = ", ".join(str(t) for t in to_raw if t)
            return {"type": "enterprise_email", "title": title, "sender": metadata.get("sender", metadata.get("from", "")), "to": to_raw, "date": metadata.get("date", "")}
        if source_type_meta == "local_document":
            return {"type": "internal", "title": title, "path": file_path}
        if source == "file":
            return {"type": "internal", "title": title, "path": metadata.get("file_path", metadata.get("original_path", file_path))}
        if source == "url" or source_type_meta == "search_result":
            return {"type": "external", "title": title, "url": metadata.get("url", "")}
        if source == "nextcloud" or source_type_meta == "nextcloud_file":
            nc_title = title
            nc_path = metadata.get("nextcloud_path", file_path)
            if nc_title and nc_title.startswith("tmp") and len(nc_title) < 20 and nc_path:
                path_name = Path(nc_path).name if "/" in nc_path else ""
                if path_name and not path_name.startswith("tmp"):
                    nc_title = path_name
            return {"type": "enterprise_file", "title": nc_title, "path": nc_path, "server": "Nextcloud"}
        if source == "filebrowser":
            return {"type": "enterprise_file", "title": title, "path": metadata.get("original_path", file_path), "server": "FileBrowser"}

        action_type = (metadata.get("action_type") or "").lower()
        if "enterprise" in action_type or action_type == "enterprise_api":
            if metadata.get("sender") or metadata.get("from") or metadata.get("subject"):
                to_raw = metadata.get("to", metadata.get("recipients", ""))
                if isinstance(to_raw, list):
                    to_raw = ", ".join(str(t) for t in to_raw if t)
                return {"type": "enterprise_email", "title": metadata.get("subject") or title,
                        "sender": metadata.get("sender") or metadata.get("from", ""), "to": to_raw, "date": metadata.get("date", "")}
            if metadata.get("user_name") or metadata.get("channel_name"):
                return {"type": "enterprise_chat", "title": "Mattermost Message",
                        "user": metadata.get("user_name", ""), "channel": metadata.get("channel_name", ""), "team": metadata.get("team_name", "")}
            return {"type": "enterprise_api", "title": title, "source_tool": "Enterprise API"}
        if "web_search" in action_type or "url_fetch" in action_type:
            return {"type": "external", "title": title, "url": metadata.get("url", "")}
        if "local" in action_type or "file" in action_type:
            return {"type": "internal", "title": title, "path": file_path}
        return {"type": "internal", "title": title, "path": file_path}

    def _source_info_from_finding_for_citation(self, finding: Dict, doc_id: str) -> Dict[str, str]:
        """Fallback source info for a finding so citations match whole_vector_db format.

        Follows the report_tools.py approach: check explicit metadata fields
        first, then tool-name keywords with email/chat before 'search'.
        """
        metadata = finding.get("metadata", {})
        tool = (finding.get("tool") or metadata.get("tool_used") or "").lower()
        file_path = metadata.get("file_path") or metadata.get("path") or ""
        filename = metadata.get("filename") or (Path(file_path).name if file_path else "")
        source = metadata.get("source", "")
        source_type_meta = (metadata.get("source_type") or "").lower()

        if self.is_local_files and (
            source == "mattermost" or source_type_meta == "mattermost_post"
            or source == "email_imap" or source_type_meta == "email_message"
            or "chat" in tool or "email" in tool
        ):
            local_title = filename or doc_id
            return {"type": "internal", "title": local_title, "path": file_path}

        title = (
            filename
            or finding.get("description")
            or metadata.get("action_description")
            or metadata.get("title")
            or ""
        )
        if isinstance(title, str) and len(title) > 120:
            title = title[:117] + "..."
        if not title or str(title).strip() in ["Unknown", "Research Finding", "Untitled", ""]:
            title = doc_id

        if source == "mattermost" or source_type_meta == "mattermost_post":
            return {"type": "enterprise_chat", "title": str(title), "user": metadata.get("user_name", ""), "channel": metadata.get("channel_name", ""), "team": metadata.get("team_name", "")}
        if source == "email_imap" or source_type_meta == "email_message":
            to_raw = metadata.get("to", metadata.get("recipients", ""))
            if isinstance(to_raw, list):
                to_raw = ", ".join(str(t) for t in to_raw if t)
            return {"type": "enterprise_email", "title": str(title), "sender": metadata.get("sender", metadata.get("from", "")), "to": to_raw, "date": metadata.get("date", "")}
        if source_type_meta == "local_document":
            return {"type": "internal", "title": str(title), "path": file_path}
        if source == "file":
            return {"type": "internal", "title": str(title), "path": metadata.get("file_path", metadata.get("original_path", file_path))}
        if source == "url" or source_type_meta == "search_result":
            return {"type": "external", "title": str(title), "url": metadata.get("url", "")}
        if source == "nextcloud" or source_type_meta == "nextcloud_file":
            nc_title = str(title)
            nc_path = metadata.get("nextcloud_path", file_path)
            if nc_title and nc_title.startswith("tmp") and len(nc_title) < 20 and nc_path:
                path_name = Path(nc_path).name if "/" in nc_path else ""
                if path_name and not path_name.startswith("tmp"):
                    nc_title = path_name
            return {"type": "enterprise_file", "title": nc_title, "path": nc_path, "server": "Nextcloud"}
        if source == "filebrowser":
            return {"type": "enterprise_file", "title": str(title), "path": metadata.get("original_path", file_path), "server": "FileBrowser"}

        description_lower = finding.get("description", "").lower()
        if "email" in tool or "email" in description_lower:
            sender = metadata.get("sender") or metadata.get("from") or ""
            date = metadata.get("date") or ""
            subject = metadata.get("subject") or ""
            to_raw = metadata.get("to", metadata.get("recipients", ""))
            if isinstance(to_raw, list):
                to_raw = ", ".join(str(t) for t in to_raw if t)
            if sender or subject:
                return {"type": "enterprise_email", "title": subject or str(title), "sender": sender, "to": to_raw, "date": date}
            return {"type": "enterprise_email", "title": str(title), "sender": "Unknown Sender", "to": to_raw, "date": ""}
        if "chat" in tool or "mattermost" in tool:
            user = metadata.get("user_name") or metadata.get("username") or ""
            channel = metadata.get("channel_name") or ""
            team = metadata.get("team_name") or ""
            if user or channel:
                return {"type": "enterprise_chat", "title": "Mattermost Message", "user": user, "channel": channel, "team": team}
            return {"type": "enterprise_chat", "title": str(title), "user": "Unknown User", "channel": "", "team": team}
        if "enterprise" in tool:
            if metadata.get("sender") or metadata.get("from") or metadata.get("subject"):
                to_raw = metadata.get("to", metadata.get("recipients", ""))
                if isinstance(to_raw, list):
                    to_raw = ", ".join(str(t) for t in to_raw if t)
                return {"type": "enterprise_email", "title": metadata.get("subject") or str(title),
                        "sender": metadata.get("sender") or metadata.get("from", ""), "to": to_raw, "date": metadata.get("date", "")}
            if metadata.get("user_name") or metadata.get("channel_name"):
                return {"type": "enterprise_chat", "title": "Mattermost Message",
                        "user": metadata.get("user_name", ""), "channel": metadata.get("channel_name", ""), "team": metadata.get("team_name", "")}
            return {"type": "enterprise_api", "title": str(title), "source_tool": tool or "Enterprise API"}
        if "local" in tool or "file" in tool:
            return {"type": "internal", "title": str(title), "path": file_path}
        if "internet" in tool or "web_search" in tool:
            return {"type": "external", "title": str(title), "url": metadata.get("url", "")}
        return {"type": "internal", "title": str(title), "path": file_path}

    def _extract_source_info_from_finding(self, finding: Dict) -> Dict[str, str]:
        """Extract source info from context.findings.

        Follows the report_tools.py approach: check explicit metadata fields
        (source, source_type) first, then fall back to tool-name keywords with
        email/chat checked before the generic 'search' keyword to avoid
        misclassifying enterprise search actions as external web sources.
        """
        metadata = finding.get("metadata", {})
        tool = finding.get("tool", metadata.get("tool_used", ""))
        source = metadata.get("source", "")
        source_type_meta = (metadata.get("source_type") or "").lower()

        if self.is_local_files and (
            source == "mattermost" or source_type_meta == "mattermost_post"
            or source == "email_imap" or source_type_meta == "email_message"
        ):
            file_path = metadata.get("file_path") or metadata.get("path") or ""
            filename = (
                metadata.get("filename")
                or metadata.get("file_name")
                or (Path(file_path).name if file_path else "")
            )
            if filename:
                return {"type": "internal", "title": filename, "path": file_path}

        title = metadata.get("title") or metadata.get("filename") or finding.get("description", "")[:60]
        if not title or title in ["Unknown", "Research Finding", ""]:
            return {"type": "unknown", "title": "Unknown"}

        if source == "mattermost" or source_type_meta == "mattermost_post":
            return {
                "type": "enterprise_chat", "title": title,
                "user": metadata.get("user_name", metadata.get("username", "")),
                "channel": metadata.get("channel_name", ""),
                "team": metadata.get("team_name", ""),
            }
        if source == "email_imap" or source_type_meta == "email_message":
            to_raw = metadata.get("to", metadata.get("recipients", ""))
            if isinstance(to_raw, list):
                to_raw = ", ".join(str(t) for t in to_raw if t)
            return {
                "type": "enterprise_email", "title": title,
                "sender": metadata.get("sender", metadata.get("from", "")),
                "to": to_raw,
                "date": metadata.get("date", ""),
            }
        if source_type_meta == "local_document":
            return {"type": "internal", "title": title, "path": metadata.get("file_path", "")}
        if source == "file":
            return {"type": "internal", "title": title, "path": metadata.get("file_path", metadata.get("original_path", ""))}
        if source == "url" or source_type_meta == "search_result":
            return {"type": "external", "title": title, "url": metadata.get("url", "")}
        if source == "nextcloud" or source_type_meta == "nextcloud_file":
            nc_title = title
            nc_path = metadata.get("nextcloud_path", metadata.get("path", ""))
            if nc_title and nc_title.startswith("tmp") and len(nc_title) < 20 and nc_path:
                path_name = Path(nc_path).name if "/" in nc_path else ""
                if path_name and not path_name.startswith("tmp"):
                    nc_title = path_name
            return {"type": "enterprise_file", "title": nc_title, "path": nc_path, "server": "Nextcloud"}
        if source == "filebrowser":
            return {"type": "enterprise_file", "title": title, "path": metadata.get("original_path", ""), "server": "FileBrowser"}

        tool_lower = tool.lower()
        description_lower = finding.get("description", "").lower()

        if "email" in tool_lower or "email" in description_lower:
            sender = metadata.get("sender") or metadata.get("from") or ""
            date = metadata.get("date") or ""
            subject = metadata.get("subject") or ""
            to_raw = metadata.get("to", metadata.get("recipients", ""))
            if isinstance(to_raw, list):
                to_raw = ", ".join(str(t) for t in to_raw if t)
            if sender or subject:
                return {"type": "enterprise_email", "title": subject or title, "sender": sender, "to": to_raw, "date": date}
            return {"type": "enterprise_email", "title": title, "sender": "Unknown Sender", "to": to_raw, "date": ""}
        if "chat" in tool_lower or "mattermost" in tool_lower:
            user = metadata.get("user_name") or metadata.get("username") or ""
            channel = metadata.get("channel_name") or ""
            team = metadata.get("team_name") or ""
            if user or channel:
                return {"type": "enterprise_chat", "title": "Mattermost Message", "user": user, "channel": channel, "team": team}
            return {"type": "enterprise_chat", "title": title, "user": "Unknown User", "channel": "", "team": team}
        if "enterprise" in tool_lower:
            if metadata.get("sender") or metadata.get("from") or metadata.get("subject"):
                to_raw = metadata.get("to", metadata.get("recipients", ""))
                if isinstance(to_raw, list):
                    to_raw = ", ".join(str(t) for t in to_raw if t)
                return {"type": "enterprise_email", "title": metadata.get("subject") or title,
                        "sender": metadata.get("sender") or metadata.get("from", ""), "to": to_raw, "date": metadata.get("date", "")}
            if metadata.get("user_name") or metadata.get("channel_name"):
                return {"type": "enterprise_chat", "title": "Mattermost Message",
                        "user": metadata.get("user_name", ""), "channel": metadata.get("channel_name", ""), "team": metadata.get("team_name", "")}
            return {"type": "enterprise_api", "title": title, "source_tool": tool}
        if "local" in tool_lower or "file" in tool_lower:
            return {"type": "internal", "title": title, "path": metadata.get("file_path", "")}
        if ("internet" in tool_lower or "web_search" in tool_lower
                or (tool_lower == "search" and "email" not in description_lower)):
            return {"type": "external", "title": title, "url": metadata.get("url", "")}
        return {"type": "internal", "title": title}

    def _generate_fallback_report(self, context: ResearchContext) -> str:
        """Generate fallback report when no valid sources available"""
        return f"""# Research Report: Personalized Workflow Analysis

## Executive Summary
Analysis was conducted but no valid citable sources were found in the knowledge base.

## 1. Generic Enterprise Workflow
Unable to generate due to insufficient source data.

## 2. Personalized Recommended Workflow
Unable to generate due to insufficient source data.

## References
No valid references available.
"""

    def _normalize_doc_citations(self, text: str) -> str:
        """Normalize citation format so [DOC_...] and [DOC ...] become [DOC:...]."""
        if not text:
            return text
        text = re.sub(r"\[DOC_([^\]]+)\]", r"[DOC:\1]", text)
        text = re.sub(r"\[DOC\s+([^\]]+)\]", r"[DOC:\1]", text)
        return text

    def _resolve_simplified_ids(self, text: str, id_map: Dict[str, str]) -> str:
        """Convert [DOC:src_N] back to [DOC:real_doc_id] using the mapping."""
        if not text or not id_map:
            return text

        def replace_simple(m):
            simple_id = m.group(1).strip()
            real_id = id_map.get(simple_id)
            if real_id:
                return f"[DOC:{real_id}]"
            return m.group(0)

        text = re.sub(r"\[DOC:(src_\d+)\]", replace_simple, text)

        all_ids = list(id_map.values())

        def replace_footnote(m):
            n = int(m.group(1))
            if 1 <= n <= len(all_ids):
                return f"[DOC:{all_ids[n-1]}]"
            return m.group(0)

        text = re.sub(r"\[\^(\d+)\]", replace_footnote, text)

        def replace_bare(m):
            n = int(m.group(1))
            if 1 <= n <= len(all_ids):
                return f"[DOC:{all_ids[n-1]}]"
            return m.group(0)

        text = re.sub(r"(?<!\[)\[(\d+)\](?!\()", replace_bare, text)

        return text

    def _count_doc_citations(self, text: str) -> int:
        """Count citations in text"""
        return len(re.findall(r"\[DOC:[^\]]+\]", text or ""))

    def _extract_step_citations(self, entry: Dict[str, Any]) -> List[str]:
        """Collect unique citations already present in a personalized workflow step."""
        combined = " ".join(
            str(entry.get(key, "")) for key in ("task", "task_description", "subtask_description")
        )
        seen: List[str] = []
        for citation in re.findall(r"\[DOC:[^\]]+\]|\[\^\d+\]", combined, re.IGNORECASE):
            if citation not in seen:
                seen.append(citation)
        return seen

    def _build_default_step_justification(self, entry: Dict[str, Any]) -> str:
        """
        Build a fallback explanation for why the personalized workflow includes
        this step, reusing citations that already appear in the step content.
        """
        title = str(entry.get("title", "")).strip() or f"step {entry.get('item', '?')}"
        prerequisite = str(entry.get("prerequisite", "")).strip()
        subtask_description = str(entry.get("subtask_description", "")).strip()
        citations = " ".join(self._extract_step_citations(entry))

        reason = f"This workflow includes {title} because the documented process requires this action"
        if prerequisite:
            reason += f" after step {prerequisite}"
        if subtask_description:
            reason += " and the personalized decision handling must remain explicit for this user."
        else:
            reason += " for the user's situation reflected in the personalized task details."
        if citations:
            reason += f" {citations}"
        return reason

    def _strip_existing_references(self, text: str) -> str:
        """Remove existing References section"""
        if not text:
            return text
        stripped = re.sub(r"\n## References\b[\s\S]*$", "", text, flags=re.IGNORECASE)
        return stripped.rstrip()

    def _validate_personalized_workflow(self, report_text: str) -> str:
        """
        Validate the personalized workflow JSON: ensure every step has a
        'justification' key and that the step as a whole has at least one
        citation across task, task_description, or justification.
        """
        section_pattern = re.compile(
            r'(#{2,4}\s*Personalized\s+Workflow\s+Steps[^\n]*\n)'
            r'(\s*(?:[^\n]*\n)*?```json\s*\n)(.*?)(\n\s*```)',
            re.DOTALL | re.IGNORECASE,
        )
        match = section_pattern.search(report_text)
        if not match:
            return report_text

        heading = match.group(1)
        fence_open = match.group(2)
        json_text = match.group(3)
        fence_close = match.group(4)

        try:
            steps = json.loads(json_text.strip())
            if not isinstance(steps, list) or not steps:
                return report_text
        except (json.JSONDecodeError, TypeError):
            return report_text

        citation_pat = re.compile(r'\[DOC:\s*\S+\]|\[\^\d+\]', re.IGNORECASE)
        changed = False

        for entry in steps:
            if not isinstance(entry, dict):
                continue

            justification = str(entry.get("justification", "") or "").strip()
            if not justification:
                entry["justification"] = self._build_default_step_justification(entry)
                changed = True

            combined = (
                f"{entry.get('task', '')} "
                f"{entry.get('task_description', '')} "
                f"{entry.get('justification', '')}"
            )
            if not citation_pat.search(combined):
                item = entry.get("item", "?")
                logger.warning("Workflow step %s has no citation in any field.", item)
                entry["justification"] = (
                    entry["justification"].rstrip()
                    + " (Citation required for this step.)"
                ).strip()
                changed = True

        if not changed:
            return report_text

        new_json = json.dumps(steps, indent=2, ensure_ascii=False)
        replacement = f"{heading}{fence_open}{new_json}{fence_close}"
        return report_text[:match.start()] + replacement + report_text[match.end():]

    def _repair_personalized_citations(
        self, report_text: str, all_docs: List[Dict]
    ) -> str:
        """
        Check if personalized workflow steps are missing [DOC:doc_id] citations.
        If any step's task or task_description lacks a citation, re-prompt the LLM
        with the steps and document contents so it can add accurate citations.
        """
        if not all_docs:
            return report_text

        section_pattern = re.compile(
            r'(#{2,4}\s*Personalized\s+Workflow\s+Steps[^\n]*\n)'
            r'(\s*(?:[^\n]*\n)*?```json\s*\n)(.*?)(\n\s*```)',
            re.DOTALL | re.IGNORECASE,
        )
        match = section_pattern.search(report_text)
        if not match:
            logger.info("No Personalized Workflow Steps JSON block found; skipping citation repair")
            return report_text

        heading = match.group(1)
        fence_open = match.group(2)
        json_text = match.group(3)
        fence_close = match.group(4)

        try:
            steps = json.loads(json_text.strip())
            if not isinstance(steps, list):
                return report_text
        except json.JSONDecodeError:
            logger.warning("Failed to parse personalized steps JSON for citation repair")
            return report_text

        doc_id_pat = re.compile(r'\[DOC:[^\]]+\]')
        uncited_indices = []
        for i, step in enumerate(steps):
            combined = (
                f"{step.get('task', '')} "
                f"{step.get('task_description', '')} "
                f"{step.get('justification', '')}"
            )
            step_has_citation = bool(doc_id_pat.search(combined))
            if not step_has_citation:
                uncited_indices.append(i)

        if not uncited_indices:
            logger.info("All personalized steps already have at least one citation")
            return report_text

        logger.info(
            f"{len(uncited_indices)} of {len(steps)} personalized steps "
            f"missing citations; re-prompting LLM"
        )

        findings_text = self._build_findings_text(all_docs)
        doc_ids_list = ", ".join(d["doc_id"] for d in all_docs)

        repair_prompt = f"""You are given personalized workflow steps where some steps are MISSING [DOC:doc_id] citations.
Your ONLY job is to add accurate [DOC:doc_id] citations so that every step has at least one.

RULES:
- Do NOT change, rewrite, or remove any existing text in task or task_description.
- For each step, read its task and task_description, find which document(s) below
  support the claim, and append [DOC:doc_id] at the end of the "task" and/or
  "task_description" values.
- Every step MUST have a "justification" key. If a step lacks one or has an empty one,
  add a "justification" value that briefly explains why this exact workflow step is
  included for the user, citing the supporting document(s) with [DOC:doc_id].
- Each step must have at least one [DOC:doc_id] citation somewhere across its task,
  task_description, or justification fields.
- Only use doc_ids from this list: {doc_ids_list}
- Prefer internal / policy doc_ids over web/external doc_ids.
- Return ONLY the JSON array — no markdown fences, no commentary.

### AVAILABLE DOCUMENTS:
{findings_text[:8000]}

### PERSONALIZED WORKFLOW STEPS (add citations and justification to these):
{json.dumps(steps, indent=2, ensure_ascii=False)}
"""

        try:
            repaired_text = prompt_llm(model=self.model, prompt=repair_prompt)
            repaired_text = self._normalize_doc_citations(repaired_text)

            repaired_text = re.sub(r'^```(?:json)?\s*\n?', '', repaired_text.strip())
            repaired_text = re.sub(r'\n?\s*```\s*$', '', repaired_text.strip())

            repaired_steps = json.loads(repaired_text)
            if not isinstance(repaired_steps, list) or len(repaired_steps) != len(steps):
                logger.warning(
                    "Repaired steps count mismatch "
                    f"({len(repaired_steps) if isinstance(repaired_steps, list) else 'N/A'} vs {len(steps)}); "
                    "keeping original"
                )
                return report_text

            def _steps_with_citation(step_list):
                count = 0
                for s in step_list:
                    combined = (
                        f"{s.get('task', '')} "
                        f"{s.get('task_description', '')} "
                        f"{s.get('justification', '')}"
                    )
                    if doc_id_pat.search(combined):
                        count += 1
                return count

            if _steps_with_citation(repaired_steps) <= _steps_with_citation(steps):
                logger.warning("LLM repair did not improve citations; keeping original")
                return report_text

            new_json = json.dumps(repaired_steps, indent=2, ensure_ascii=False)
            replacement = f"{heading}{fence_open}{new_json}{fence_close}"
            return report_text[: match.start()] + replacement + report_text[match.end() :]

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Personalized citation repair failed: {e}")
            return report_text

    def _repair_citations(self, flowchart_report: str, findings_text: str, available_doc_ids: List[str]) -> str:
        """Repair missing citations"""
        if not flowchart_report or not available_doc_ids:
            return flowchart_report

        repair_prompt = f"""
Add [DOC:doc_id] citations to this report. Rules:
1. DO NOT change content - only ADD citations
2. Use format: [DOC:doc_id]. Valid doc_ids: {', '.join(available_doc_ids[:30])}{'...' if len(available_doc_ids) > 30 else ''}
3. Add citations after factual claims and in justifications in the Generic Workflow Steps
4. **PERSONALIZED WORKFLOW (Section 2)**: In the "Personalized Workflow Steps" JSON, add [DOC:doc_id] to EACH step's task, task_description, and justification (policy and/or personal doc_ids). No step should lack citations.
5. **Section 3 (Step-by-Step Personalized Guidance)**: Add [DOC:doc_id] to each step's Description and to at least one Action Item per step
6. Replace empty () with proper citations

DOCUMENTS:
{findings_text[:5000]}

REPORT:
{flowchart_report}

Return the complete report with citations added throughout, especially in the personalized workflow and step-by-step guidance.
"""

        try:
            repaired = prompt_llm(model=self.model, prompt=repair_prompt)
            if self._count_doc_citations(repaired) > self._count_doc_citations(flowchart_report):
                return repaired
            return flowchart_report
        except Exception as e:
            logger.warning(f"Citation repair failed: {e}")
            return flowchart_report
