import json
import logging
import re
from typing import Any, Dict, List, Optional

from drflow.agents.utils import prompt_llm

from drflow.agents.drflow_agent.vector_store import VectorStore

from .base import ResearchContext
from .citation_registry import UnifiedCitationRegistry
from .model_config import get_report_config

logger = logging.getLogger(__name__)


class ReportAssembler:
    """Enhanced report assembler with intelligent content synthesis and comprehensive metadata tracking"""

    def __init__(
        self,
        model: str,
        vector_store: Optional[VectorStore] = None,
        capacity_tier: Optional[str] = None,
        max_content_length: int = None,
        max_total_tokens: int = None,
    ):
        self.model = model
        self.vector_store = vector_store

        config = get_report_config(
            capacity_tier=capacity_tier, max_content_length=max_content_length, max_total_tokens=max_total_tokens
        )

        self.max_content_length = config["max_content_length"]
        self.max_total_tokens = config["max_total_tokens"]

        self.evidence_metadata = {
            "document_ids": [],
            "evidence_map": {"web_sources": [], "internal_sources": [], "research_findings": []},
            "content_synthesis": {},
            "research_methodology": {},
        }
        self.citation_registry = UnifiedCitationRegistry()
        self._simplified_id_map: Dict[str, str] = {}
        self._doc_id_to_simple_id: Dict[str, str] = {}
        self._simplified_source_registry: Dict[str, Dict[str, str]] = {}

    def generate_comprehensive_report(
        self, context: ResearchContext, action_plan=None, persona_context: Optional[str] = None
    ) -> str:
        """Generate comprehensive report with intelligent content synthesis and rich metadata tracking.
        Personal evidence for workflow personalization is collected from retrieved content (emails, chats,
        internal files). The optional persona_context is not used; it is accepted only for backward compatibility.
        """

        try:
            self._reset_evidence_metadata()

            thematic_content = self._analyze_and_cluster_content(context)
            logger.info(f"Identified {len(thematic_content)} themes for report synthesis !!!!")

            synthesized_sections = self._synthesize_thematic_content(thematic_content, context)

            evidence_summary = self._build_evidence_metadata(context)

            report = self._assemble_final_report(synthesized_sections, evidence_summary, context)

            self._finalize_metadata(context, action_plan, thematic_content)

            return report

        except Exception as e:
            return f"Error generating comprehensive report: {e}"

    def _reset_evidence_metadata(self):
        """Reset evidence metadata for new report generation"""
        self.evidence_metadata = {
            "document_ids": [],
            "evidence_map": {"web_sources": [], "internal_sources": [], "research_findings": []},
            "content_synthesis": {},
            "research_methodology": {},
        }
        self.citation_registry = UnifiedCitationRegistry()
        self._simplified_id_map = {}
        self._doc_id_to_simple_id = {}
        self._simplified_source_registry = {}

    def _normalize_doc_citations(self, text: str) -> str:
        """Normalize common LLM citation variants to [DOC:...] format."""
        if not text:
            return text
        text = re.sub(r"\[DOC_([^\]]+)\]", r"[DOC:\1]", text)
        text = re.sub(r"\[DOC\s+([^\]]+)\]", r"[DOC:\1]", text)
        return text

    def _build_simplified_id_map(self, content_items: List[Dict]) -> None:
        """Assign prompt-friendly source IDs like src_1 to registered documents."""
        self._simplified_id_map = {}
        self._doc_id_to_simple_id = {}
        self._simplified_source_registry = {}

        next_index = 1
        for item in content_items:
            doc_id = item.get("doc_id")
            if not doc_id or doc_id in self._doc_id_to_simple_id:
                continue
            if doc_id not in self.citation_registry.documents:
                continue

            simple_id = f"src_{next_index}"
            next_index += 1
            self._simplified_id_map[simple_id] = doc_id
            self._doc_id_to_simple_id[doc_id] = simple_id
            self._simplified_source_registry[simple_id] = self.citation_registry.documents[doc_id].source_info

    def _format_available_sources_for_prompt(self, max_sources: Optional[int] = 30) -> str:
        """Render a compact source list for citation-aware prompts."""
        if not self._simplified_source_registry:
            return "No registered sources available."

        lines = []
        items = list(self._simplified_source_registry.items())
        if max_sources is not None:
            items = items[:max_sources]

        for simple_id, source_info in items:
            title = source_info.get("title", "Untitled")
            source_type = source_info.get("type", "unknown")
            description = (
                source_info.get("description")
                or source_info.get("path")
                or source_info.get("url")
                or source_info.get("source_tool")
                or source_type
            )
            lines.append(f"- {simple_id}: {title} ({description})")

        return "\n".join(lines)

    def _replace_real_ids_with_simplified(self, text: str) -> str:
        """Convert [DOC:real_doc_id] references to [DOC:src_N] for prompting."""
        if not text or not self._doc_id_to_simple_id:
            return text

        def replace_doc_reference(match: re.Match[str]) -> str:
            doc_id = match.group(1).strip()
            simple_id = self._doc_id_to_simple_id.get(doc_id)
            if simple_id:
                return f"[DOC:{simple_id}]"
            return match.group(0)

        return re.sub(r"\[DOC:([^\]]+)\]", replace_doc_reference, text)

    def _resolve_simplified_ids(self, text: str) -> str:
        """Convert [DOC:src_N], [^N], or [N] back to registered real doc IDs."""
        if not text or not self._simplified_id_map:
            return text

        def replace_simple(match: re.Match[str]) -> str:
            simple_id = match.group(1).strip()
            real_id = self._simplified_id_map.get(simple_id)
            if real_id:
                return f"[DOC:{real_id}]"
            return match.group(0)

        text = re.sub(r"\[DOC:(src_\d+)\]", replace_simple, text)

        all_ids = list(self._simplified_id_map.values())

        def replace_footnote(match: re.Match[str]) -> str:
            n = int(match.group(1))
            if 1 <= n <= len(all_ids):
                return f"[DOC:{all_ids[n - 1]}]"
            return match.group(0)

        text = re.sub(r"\[\^(\d+)\]", replace_footnote, text)

        def replace_bare(match: re.Match[str]) -> str:
            n = int(match.group(1))
            if 1 <= n <= len(all_ids):
                return f"[DOC:{all_ids[n - 1]}]"
            return match.group(0)

        text = re.sub(r"(?<!\[)\[(\d+)\](?!\()", replace_bare, text)
        return text

    def _analyze_and_cluster_content(self, context: ResearchContext) -> Dict[str, List[Dict]]:
        """Analyze vector store content and cluster by themes"""

        if not self.vector_store:
            return {"general": []}

        try:
            search_queries = self._generate_search_queries(context.original_question, context.plan)
            all_content = {}

            internal_queries = self._generate_internal_search_queries(context.original_question, context.plan)
            all_search_queries = search_queries + internal_queries
            logger.info(f"!!! Generated {len(all_search_queries)} search queries for content analysis")
            logger.info(f"!!! Search Queries: {all_search_queries}")

            for query in all_search_queries:
                results = self.vector_store.search(query, top_k=15, use_semantic=True)
                theme = self._extract_theme_from_query(query)

                if theme not in all_content:
                    all_content[theme] = []

                for result in results:
                    metadata = result.get("metadata", {})
                    is_internal = self._is_internal_source(metadata)
                    min_threshold = 0.5 if is_internal else 0.7

                    if result.get("similarity_score", 0) > min_threshold:
                        if (
                            metadata.get("tool_used") == "smart_analysis"
                            and metadata.get("type") == "ai_synthesis"
                            and not metadata.get("source_document_ids")
                        ):
                            logger.debug(
                                f"Skipping AI synthesis without sources in content analysis: {result.get('doc_id')}"
                            )
                            continue

                        all_content[theme].append(
                            {
                                "content": result.get("content", ""),
                                "metadata": metadata,
                                "score": result.get("similarity_score", 0),
                                "doc_id": result.get("doc_id", ""),
                                "theme": theme,
                                "is_internal": is_internal,
                            }
                        )

            for theme in all_content:
                original_count = len(all_content[theme])
                all_content[theme] = self._deduplicate_and_rank(all_content[theme])
                final_count = len(all_content[theme])

                internal_count = sum(1 for item in all_content[theme] if item.get("is_internal", False))
                external_count = final_count - internal_count

                logger.debug(
                    f"Theme '{theme}': {original_count}→{final_count} items ({internal_count} internal, {external_count} external)"
                )

            total_items = sum(len(items) for items in all_content.values())
            total_internal = sum(
                sum(1 for item in items if item.get("is_internal", False)) for items in all_content.values()
            )
            logger.info(
                f"🔍 Total content analysis: {total_items} items ({total_internal} internal, {total_items - total_internal} external)"
            )

            return all_content

        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")
            return {"general": []}

    def _generate_search_queries(self, original_question: str, research_plan: Any) -> List[str]:
        """Generate targeted search queries for comprehensive content discovery"""

        actionable_tasks = []
        if (
            research_plan
            and isinstance(research_plan, dict)
            and "research_investigation_areas" in research_plan
            and isinstance(research_plan["research_investigation_areas"], list)
        ):
            actionable_tasks = research_plan["research_investigation_areas"]

        research_plan_length = len(actionable_tasks) if actionable_tasks else 5
        research_questions = (
            "\n - ".join(
                [f'{task.get("research_focus", "")}: {task.get("business_rationale", "")}' for task in actionable_tasks]
            )
            if actionable_tasks
            else original_question
        )

        query_prompt = f"""
Generate comprehensive search queries to find ALL relevant documents for: "{original_question}"

Research Areas:
- {research_questions}

REQUIRED QUERY CATEGORIES (generate 2-3 queries per category):

1. **QUANTITATIVE DATA QUERIES:**
   - Cost analysis, budget impact, percentage changes
   - Customer surveys, satisfaction metrics, performance data
   - Financial analysis, ROI calculations, expense tracking

2. **ACHIEVEMENT & OUTCOME QUERIES:**
   - Win/loss ratio, sales performance, competitive analysis
   - Customer retention, churn analysis, renewal rates
   - Success stories, case studies, performance improvements

3. **COMPLIANCE & REGULATORY QUERIES:**
   - Regulatory requirements, compliance assessments, audit results
   - Legal analysis, policy documents, regulatory frameworks
   - Implementation guidelines, compliance gaps

4. **TECHNICAL IMPLEMENTATION QUERIES:**
   - Architecture assessments, technical requirements
   - Resource planning, staffing needs, implementation costs
   - System capabilities, technical gaps, upgrade requirements

5. **MARKET & CUSTOMER QUERIES:**
   - Customer concerns, market research, industry analysis
   - Competitor analysis, market positioning, customer feedback
   - Industry trends, market opportunities, customer segments

Use specific document type keywords: "analysis", "assessment", "report", "study", "survey", "review", "evaluation", "comparison", "performance", "metrics", "data", "statistics", "costs", "budget", "financial", "achievement", "outcome", "success", "improvement"

Return only the queries, one per line:
"""

        try:
            response = prompt_llm(model=self.model, prompt=query_prompt)
            queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
            return queries
        except:
            fallback_tasks = actionable_tasks if actionable_tasks else []
            return [
                original_question,
                f"background {original_question}",
                f"current trends {original_question}",
                f"analysis {original_question}",
                f"implementation {original_question}",
            ] + [task.get("research_focus", original_question) for task in fallback_tasks]

    def _extract_theme_from_query(self, query: str) -> str:
        """Extract theme identifier from search query"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["background", "history", "overview"]):
            return "background"
        elif any(word in query_lower for word in ["trends", "current", "recent", "latest"]):
            return "current_trends"
        elif any(word in query_lower for word in ["analysis", "research", "study", "data"]):
            return "analysis"
        elif any(word in query_lower for word in ["implementation", "practical", "how to", "guide"]):
            return "implementation"
        elif any(word in query_lower for word in ["future", "prediction", "forecast"]):
            return "future_outlook"
        else:
            return "general"

    def _deduplicate_and_rank(
        self, content_list: List[Dict], internal_ratio: float = 0.65, max_results: Optional[int] = 10
    ) -> List[Dict]:
        """Remove duplicates and rank content by quality, prioritizing internal sources"""

        unique_content = []
        seen_content = set()

        for item in content_list:
            content_hash = hash(item["content"][:600])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_content.append(item)

        def ranking_key(x):
            is_internal = x.get("is_internal", False)
            score = x["score"]
            content_length = len(x["content"])
            priority_score = (2.0 if is_internal else 1.0) * score
            return (priority_score, content_length)

        ranked_content = sorted(unique_content, key=ranking_key, reverse=True)

        internal_items = [x for x in ranked_content if x.get("is_internal", False)]
        external_items = [x for x in ranked_content if not x.get("is_internal", False)]

        max_results = max_results or min(len(ranked_content), 8)
        result = (
            internal_items[: int(max_results * internal_ratio)]
            + external_items[: int(max_results * (1 - internal_ratio))]
        )

        return result

    def _build_evidence_metadata(self, context: ResearchContext) -> Dict[str, Any]:
        """Build comprehensive evidence metadata from research context and source registry"""

        internal_count = 0
        enterprise_count = 0
        web_count = 0
        research_count = 0

        for doc_id, doc_info in self.citation_registry.documents.items():
            if doc_info.document_type == "ai_synthesis":
                continue
            source_info = doc_info.source_info
            source_type = source_info.get("type", "")

            if source_type == "internal":
                internal_count += 1
            elif source_type in ["enterprise_chat", "enterprise_file", "enterprise_api", "enterprise_email"]:
                enterprise_count += 1
            elif source_type == "external":
                web_count += 1
            elif source_type in ["research", "synthesis"]:
                research_count += 1

        for key, finding in context.findings.items():
            if isinstance(finding, dict) and finding.get("success"):
                for file_path in finding.get("processed_files", []):
                    if file_path not in self.evidence_metadata["document_ids"]:
                        self.evidence_metadata["document_ids"].append(file_path)

                for content_item in finding.get("fetched_content", []):
                    url = content_item.get("url")
                    if url and url not in self.evidence_metadata["document_ids"]:
                        self.evidence_metadata["document_ids"].append(url)

        return {
            "total_sources": internal_count + enterprise_count + web_count + research_count,
            "web_sources": web_count,
            "internal_sources": internal_count,
            "enterprise_sources": enterprise_count,
            "research_findings": research_count,
            "themes_identified": [],
        }

    def _extract_doc_citations(self, text: str) -> List[str]:
        """Collect unique inline citations from text in appearance order."""
        if not text:
            return []

        seen = set()
        citations = []
        for citation in re.findall(r"\[DOC:[^\]]+\]|\[\^\d+\]", text, re.IGNORECASE):
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)
        return citations

    def _validate_and_format_workflow(self, content: str) -> str:
        """Validate workflow JSON and warn when a step has no citations anywhere."""
        raw = self._normalize_doc_citations(content.strip())
        raw = self._resolve_simplified_ids(raw)
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            raw = match.group(1).strip()
        if not raw:
            return content
        try:
            steps = json.loads(raw)
            if not isinstance(steps, list) or not steps:
                return content
        except (json.JSONDecodeError, TypeError):
            return content

        for entry in steps:
            if not isinstance(entry, dict):
                continue
            item = entry.get("item", "?")

            step_citations = []
            for field_name in ("task", "task_description", "subtask_description", "justification"):
                step_citations.extend(self._extract_doc_citations(str(entry.get(field_name, "") or "")))
            step_citations = list(dict.fromkeys(step_citations))

            if not step_citations:
                logger.warning(
                    "Workflow step %s has no citations in task, task_description, subtask_description, or justification.",
                    item,
                )

        formatted_steps = json.dumps(steps, indent=2, ensure_ascii=False)
        return f"```json\n{formatted_steps}\n```"

    def personalized_workflow_synthesis(self, synthesized_text: str, query: str) -> str:
        """Synthesize a personalized step-by-step workflow from all gathered theme insights.
        The model personalizes using the DR question and the Generic Insights,
        resolving conditionals where possible."""
        if not synthesized_text:
            logger.warning("No synthesized text provided for workflow generation.")
            return ""

        prompt_ready_synthesis = self._replace_real_ids_with_simplified(synthesized_text)
        available_sources = self._format_available_sources_for_prompt()

        max_insights_chars = 12000
        if len(prompt_ready_synthesis) > max_insights_chars:
            prompt_ready_synthesis = prompt_ready_synthesis[:max_insights_chars] + "\n... (truncated for brevity)"
            logger.debug("Truncated combined synthesis to %d chars for workflow prompt.", max_insights_chars)

        personalization_prompt = f"""You are an expert personalized workflow generator. Your task is to take a deep research question (DR Question) and the analysis findings (Generic Insights gathered from all research themes), and produce a personalized workflow of **10-15** steps for the user.

**Do not assume a separate "user context" or "personal evidence" block.** Personalize only from:
1. The DR Question (it often encodes the user's situation).
2. The Generic Insights below (policy, procedures, and any situation-specific details already in the findings).

**Conditional steps (multiple options):** When a step has conditions with multiple options (e.g. "If X then A, else B"), search through the information in the Generic Insights and the DR question to determine what is applicable to the user. If you can resolve which path applies, set 'subtask_description' to empty and personalize 'task' and 'task_description' to that path. If you cannot determine which path applies from the available information, keep 'subtask_description' with the possible paths.

---BEGIN DR QUESTION---
{query}
---END DR QUESTION---

### AVAILABLE SOURCES
Use only these source IDs for citations:
{available_sources}

---BEGIN GENERIC INSIGHTS (ALL THEMES)---
{prompt_ready_synthesis}
---END GENERIC INSIGHTS---

### Output Format (JSON array only):
Return ONLY a valid JSON array. Do NOT include any heading, title, or commentary before or after the JSON. Do NOT include "### Personalized Workflow Recommendations:" or any other heading. Output the JSON array directly.

[
  {{
    "item": 1,
    "title": "short title for this step",
    "task": "what to do in this step",
    "task_description": "detailed low-level description personalized from the DR question and findings",
    "subtask_description": "",
    "prerequisite": "",
    "prerequisite_description": "",
    "justification": "why this step is relevant, ideally citing specific findings [DOC:src_N]."
  }}
]

### Rules:
1. **Step count:** Produce **10-15** workflow steps (minimum 10, maximum 15). Split merged actions until you reach at least 10 discrete steps when the findings support it; do not pad with filler.
2. **Step-by-step order:** Items 1, 2, 3, ... in logical sequence. Use prerequisite field to link dependencies.
3. **Resolve conditionals:** Use the DR question and findings to determine which path applies; empty subtask_description when resolved.
4. **Personalize from findings only:** Adapt task and task_description using details from the DR question and the Generic Insights.
5. **Citations encouraged in EVERY step:** Each step should include at least one [DOC:src_N] citation somewhere across 'task', 'task_description', or 'justification'.
6. Citation format: [DOC:src_N] with colon. Use individual citations: [DOC:src_1][DOC:src_2] not [DOC:src_1; DOC:src_2].
7. If you can support a step from the findings, include a citation in whichever field reads most naturally. Do not invent citations.
8. Be comprehensive: cover all major workflow steps from the findings across the 10-15 steps.
9. Only use citations from the AVAILABLE SOURCES list above. Do not invent source IDs.

Example (domain-agnostic):
[
  {{
    "item": 1,
    "title": "Create or open the required record",
    "task": "Create a new record in the appropriate system [DOC:src_1]",
    "task_description": "Using the system indicated in the findings, create or open the record. Populate the required fields from the DR question and findings [DOC:src_1].",
    "subtask_description": "",
    "prerequisite": "",
    "prerequisite_description": "",
    "justification": "A record is required per policy so the workflow can proceed [DOC:src_1]."
  }},
  {{
    "item": 2,
    "title": "Select the applicable option or track",
    "task": "Choose the option that applies based on findings [DOC:src_2]",
    "task_description": "Determine which branch applies from the findings. Set the selection and save [DOC:src_2].",
    "subtask_description": "",
    "prerequisite": 1,
    "prerequisite_description": "The record must exist before the option can be set.",
    "justification": "The findings indicate the applicable option for this situation [DOC:src_2]."
  }}
]

CRITICAL REMINDER: Output ONLY the JSON array with **10-15** steps. Try to include at least one [DOC:src_N] citation somewhere in each step when the findings support it."""

        try:
            personalized_synthesis = prompt_llm(model=self.model, prompt=personalization_prompt)
            if not personalized_synthesis or not personalized_synthesis.strip():
                logger.error("LLM returned empty response for personalized workflow.")
                return ""
            personalized_synthesis = self._normalize_doc_citations(personalized_synthesis)
            personalized_synthesis = self._resolve_simplified_ids(personalized_synthesis)
            return personalized_synthesis
        except Exception as e:
            logger.error("Failed to generate personalized workflow: %s", e, exc_info=True)
            return ""

    def _synthesize_thematic_content(
        self, thematic_content: Dict[str, List[Dict]], context: ResearchContext
    ) -> Dict[str, str]:
        """Synthesize content for each theme using focused LLM calls with inline citations.
        Workflow personalization is done by prompting the model to personalize from the DR question
        and the Generic Insights, and to resolve conditionals using whatever in the findings is applicable.
        """

        base_content_length = self.max_content_length

        total_items = sum(len(items) for items in thematic_content.values())
        themes_count = len(thematic_content)
        avg_items_per_theme = total_items / themes_count if themes_count > 0 else 1

        if avg_items_per_theme > 10:
            max_content_length = base_content_length // 2
        elif avg_items_per_theme > 5:
            max_content_length = int(base_content_length * 0.75)
        else:
            max_content_length = base_content_length

        logger.debug(f"Using max_content_length={max_content_length} for {avg_items_per_theme:.1f} avg items per theme")

        all_items = []
        for theme_items in thematic_content.values():
            all_items.extend(theme_items)

        seen_doc_ids = set()
        unique_items = []
        for item in all_items:
            doc_id = item.get("doc_id")
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_items.append(item)

        for item in unique_items:
            self._get_source_citation_id(item)

        self._build_simplified_id_map(unique_items)

        global_sources = self._build_theme_source_registry(unique_items)

        synthesized_sections = {}

        for theme, content_items in thematic_content.items():
            if not content_items:
                continue

            prioritized_items = self._prioritize_sources(content_items)

            content_for_synthesis = []
            seen_citation_ids = set()

            for item in prioritized_items:
                source_id = self._get_source_citation_id(item)
                metadata = item["metadata"]
                source = metadata.get("source", "")
                source_type = metadata.get("source_type", metadata.get("type", ""))

                tool_used = metadata.get("tool_used", "")
                source_document_ids = metadata.get("source_document_ids", [])

                if tool_used == "smart_analysis" and source_document_ids:
                    priority = "internal"
                elif (
                    source == "file"
                    or source in ["mattermost", "nextcloud", "enterprise", "api", "email_imap"]
                    or source_type
                    in [
                        "local_document",
                        "mattermost_post",
                        "nextcloud_file",
                        "email_message",
                        "enterprise_email",
                    ]
                ):
                    priority = "internal"
                elif source == "url" or source_type == "search_result":
                    priority = "external"
                else:
                    priority = "research"

                if source_id == "skip":
                    doc_id = item.get("doc_id")
                    if doc_id and doc_id in self.citation_registry.documents:
                        doc_info = self.citation_registry.documents[doc_id]
                        underlying_citations = doc_info.underlying_docs

                        if underlying_citations:
                            simplified_underlying = [
                                self._doc_id_to_simple_id.get(source_doc_id, source_doc_id)
                                for source_doc_id in underlying_citations
                                if self._doc_id_to_simple_id.get(source_doc_id, source_doc_id)
                            ]
                            content = self._replace_real_ids_with_simplified(item["content"])
                            if len(content) > max_content_length:
                                content = self._smart_truncate_content(content, max_content_length)

                            has_doc_refs = "[DOC:" in content

                            content_summary = {
                                "doc_id": item["doc_id"],
                                "text": content,
                                "source_type": "ai_synthesis",
                                "relevance": item["score"],
                                "citation_id": "HAS_DOC_REFS" if has_doc_refs else "NO_CITATION",
                                "source_priority": priority,
                                "source_description": f"AI synthesis from {len(underlying_citations)} sources",
                                "underlying_doc_ids": simplified_underlying,
                            }
                            content_for_synthesis.append(content_summary)

                            # Note: Don't add underlying citation IDs to seen_citation_ids
                    continue

                doc_id = item["doc_id"]
                if doc_id in seen_citation_ids:
                    continue
                seen_citation_ids.add(doc_id)

                content = self._replace_real_ids_with_simplified(item["content"])
                if len(content) > max_content_length:
                    content = self._smart_truncate_content(content, max_content_length)
                content_summary = {
                    "doc_id": item["doc_id"],
                    "text": content,
                    "source_type": source,
                    "relevance": item["score"],
                    "citation_id": self._doc_id_to_simple_id.get(item["doc_id"], item["doc_id"]),
                    "source_priority": priority,
                    "source_description": self._get_source_description_for_synthesis(metadata),
                }
                content_for_synthesis.append(content_summary)

            estimated_tokens = len(json.dumps(content_for_synthesis)) + len(json.dumps(global_sources)) + 2000
            if estimated_tokens > self.max_total_tokens:
                logger.debug(f"Content too large ({estimated_tokens} tokens), applying adaptive truncation")
                content_for_synthesis = self._adaptive_content_batching(content_for_synthesis, max_content_length // 2)

            synthesis_prompt = f"""
As an expert research analyst, synthesize the following content into a coherent, insightful, and well-supported analysis for the theme: "{theme}" directly related to the overarching research question: "{context.original_question}"

Content for Synthesis (ordered by source importance):
{json.dumps(content_for_synthesis, indent=2)}

Available Citations for Inline Reference:
{json.dumps(global_sources, indent=2)}

Source Priority Guidelines:
1.  **"internal"**: Highest priority (e.g., internal company documents, proprietary files, confidential reports, enterprise chat messages, local documents, CRM data, internal APIs, project management tools). Insights from these sources should form the primary foundation of the analysis.
2.  **"external"**: Medium priority (e.g., public web sources, academic papers, industry reports, news articles). Use these to provide broader context, external validation, or contrasting perspectives.
3.  **"research"**: Supporting priority (e.g., AI-generated research findings, analytical synthesis from multiple sources, preliminary data from exploratory analysis). These provide valuable cross-source insights and meta-analysis that connect patterns across multiple documents.

**Synthesis Requirements:**

* **QUANTITATIVE PRIORITY:** Lead with numerical data, calculations, and aggregations
    * Extract ALL percentages, costs, metrics, and performance data
    * Perform mathematical operations: aggregate percentages, calculate increases, sum totals
    * Example: "Finance customers (35%) combined with healthcare (40%) represent 75% of regulated industry concerns [DOC:src_1][DOC:src_2]"
    * Example: "The $2,000,000 investment represents a 20% increase over baseline costs [DOC:src_3]"

* **ACHIEVEMENT & OUTCOME EMPHASIS:**
    * Prioritize: win/loss improvements, retention gains, cost savings, efficiency increases
    * Look for: before/after comparisons, ROI data, performance benchmarks
    * Extract: specific business results and quantifiable achievements
    * Use phrases like "achieved", "improved", "increased", "reduced", "enhanced"

* **FACT VERIFICATION (CRITICAL):**
    * ONLY state what documents explicitly contain - no inference or extrapolation
    * Use exact quotes for key numerical claims: "As stated in the document: '[exact quote]' [DOC:src_1]"
    * If documents reference "internal analyses" without providing results, state: "document references internal analysis but doesn't provide specific findings"
    * Never claim data exists if not explicitly shown in documents

* **Narrative Flow with Quantitative Foundation:**
    * Lead each paragraph with specific numerical findings or achievements
    * Support with qualitative context and implications
    * Maintain logical flow while preserving exact data points
    * Connect quantitative findings to business impact

* **Mathematical Transparency:**
    * Show calculations when aggregating data: "Combined percentage: 35% + 40% = 75% [DOC:src_1][DOC:src_2]"
    * Explain derivations: "20% cost increase calculated from $2M investment against $10M baseline [DOC:src_3]"

* **Citation Usage (Critical):**
    * **Format:** Reference sources only with the provided simplified IDs: "Internal review shows 15% increase [DOC:src_1]"
    * **Use INDIVIDUAL citations:** [DOC:src_1][DOC:src_2] NOT [DOC:src_1; DOC:src_2]
    * **PRESERVE EXISTING CITATIONS:** When reusing AI synthesis content with [DOC:src_N] citations, preserve exactly as they appear
    * **ACTIVELY RETRIEVE CITATIONS:** For AI synthesis without explicit citations, add citations to underlying documents from "underlying_doc_ids"
    * **NEVER HALLUCINATE CITATIONS:** Only use provided src_N values from Available Citations for Inline Reference
    * **Cite every numerical claim and calculation with source documents**

* **Conciseness & Comprehensiveness:** 2-4 substantive paragraphs with quantitative focus
* **Integration:** Seamlessly weave numerical findings with supporting context
* **Output Format:** Markdown with inline [DOC:src_N] citations only - no reference lists

Generate 2-4 paragraphs of synthesized analysis with proper inline citations:
"""

            try:
                synthesis = prompt_llm(model=self.model, prompt=synthesis_prompt)

                synthesis = self._clean_spurious_citations(synthesis)
                synthesis = self._normalize_doc_citations(synthesis)
                synthesis = self._resolve_simplified_ids(synthesis)


                # Don't clean citations here - the LLM might reference documents from other themes
                synthesized_sections[theme] = synthesis

                self._track_synthesis_metadata(theme, prioritized_items)

            except Exception as e:
                logger.warning(f"Theme synthesis failed for {theme}: {e}")
                synthesized_sections[theme] = (
                    f"Content analysis available for {len(prioritized_items)} sources in {theme}"
                )

        return synthesized_sections

    def _clean_spurious_citations(self, synthesis_text: str) -> str:
        """Remove any spurious citation sections that the LLM might have added to individual sections"""

        citation_line_pattern = r"^\s*\[.*?\]:.*$"

        lines = synthesis_text.split("\n")
        cleaned_lines = []
        skip_next_lines = False

        for line in lines:
            if re.match(citation_line_pattern, line.strip()):
                skip_next_lines = True
                continue
            elif skip_next_lines and line.strip() == "":
                continue
            else:
                skip_next_lines = False
                cleaned_lines.append(line)

        while cleaned_lines and cleaned_lines[-1].strip() == "":
            cleaned_lines.pop()

        return "\n".join(cleaned_lines)

    def _smart_truncate_content(self, content: str, max_length: int) -> str:
        """Smart truncation that preserves important content and citations"""
        if len(content) <= max_length:
            return content

        if len(content) > max_length * 3:
            cut_point = max_length
            for i in range(max_length - 100, min(len(content), max_length + 100)):
                if content[i] == "." and i + 1 < len(content) and content[i + 1] in " \n":
                    cut_point = i + 1
                    break
            content = content[:cut_point]


        paragraphs = []
        if "\n\n" in content:
            paragraphs = content.split("\n\n")
        elif "\n" in content:
            paragraphs = content.split("\n")
        else:
            import re

            paragraphs = re.split(r"(?<=[.!?])\s+", content)


        important_paragraphs = []
        regular_paragraphs = []

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para or len(para) < 10:
                continue

            is_important = (
                any(
                    keyword in para.lower()
                    for keyword in [
                        "conclusion",
                        "summary",
                        "key insight",
                        "important",
                        "critical",
                        "recommendation",
                        "action",
                        "next steps",
                        "findings",
                        "results",
                        "key finding",
                        "main point",
                        "essential",
                        "crucial",
                    ]
                )
                or para.startswith(("1.", "2.", "3.", "•", "-", "*", "#"))
                or any(char.isdigit() and "%" in para for char in para)
                or len([c for c in para if c.isdigit()]) > 3
                or i == 0
            )

            if is_important:
                important_paragraphs.append(para)
            else:
                regular_paragraphs.append(para)

        result = ""
        separator = "\n\n" if "\n\n" in content else "\n"

        for para in important_paragraphs:
            needed_space = len(para) + len(separator)
            if len(result) + needed_space <= max_length:
                if result:
                    result += separator
                result += para
            else:
                remaining = max_length - len(result) - len(separator) - 3
                if remaining > 50:
                    if result:
                        result += separator
                    truncated = para[:remaining]
                    last_sentence = truncated.rfind(".")
                    if last_sentence > remaining * 0.7:
                        result += truncated[: last_sentence + 1] + "..."
                    else:
                        result += truncated + "..."
                break

        for para in regular_paragraphs:
            needed_space = len(para) + len(separator)
            if len(result) + needed_space <= max_length:
                if result:
                    result += separator
                result += para
            else:
                remaining = max_length - len(result) - len(separator) - 3
                if remaining > 50:
                    if result:
                        result += separator
                    truncated = para[:remaining]
                    last_sentence = truncated.rfind(".")
                    if last_sentence > remaining * 0.7:
                        result += truncated[: last_sentence + 1] + "..."
                    else:
                        result += truncated + "..."
                break

        if not result.strip():
            result = content[: max_length - 3] + "..."

        return result.strip()

    def _adaptive_content_batching(self, content_items: List[Dict], target_length: int) -> List[Dict]:
        """Adaptively batch content items to fit within target length while preserving quality"""
        if not content_items:
            return content_items

        def priority_score(item):
            priority_map = {"internal": 3, "external": 2, "research": 1}
            source_priority = priority_map.get(item.get("source_priority", "research"), 1)
            relevance = item.get("relevance", 0)
            return source_priority * 10 + relevance

        sorted_items = sorted(content_items, key=priority_score, reverse=True)

        result = []
        current_length = 0

        for item in sorted_items:
            item_length = len(item.get("text", ""))

            if not result:
                if item_length > target_length:
                    item["text"] = self._smart_truncate_content(item["text"], target_length)
                result.append(item)
                current_length = len(item["text"])
                continue

            if current_length + item_length <= target_length:
                result.append(item)
                current_length += item_length
            else:
                remaining = target_length - current_length
                if remaining > 200:
                    item["text"] = self._smart_truncate_content(item["text"], remaining)
                    result.append(item)
                break

        logger.debug(f"Adaptive batching: {len(content_items)} -> {len(result)} items ({current_length} chars)")
        return result

    def _get_highest_priority_from_source_docs(self, source_document_ids: List[str]) -> str:
        """Determine the highest priority from a list of source document IDs"""
        if not source_document_ids:
            return "research"

        priorities = []
        for doc_id in source_document_ids:
            if self.vector_store:
                docs = self.vector_store.search(f"doc_id:{doc_id}", top_k=1)
                if docs:
                    doc = docs[0]
                    metadata = doc.get("metadata", {})
                    source = metadata.get("source", "")
                    source_type = metadata.get("source_type", metadata.get("type", ""))

                    if (
                        source == "file"
                        or source in ["mattermost", "nextcloud", "enterprise", "api", "email_imap"]
                        or source_type
                        in [
                            "local_document",
                            "mattermost_post",
                            "nextcloud_file",
                            "email_message",
                            "enterprise_email",
                        ]
                    ):
                        priorities.append("internal")
                    elif source == "url" or source_type == "search_result":
                        priorities.append("external")
                    else:
                        priorities.append("research")

        priority_order = ["internal", "external", "research"]
        for priority in priority_order:
            if priority in priorities:
                return priority

        return "research"

    def _prioritize_sources(self, content_items: List[Dict]) -> List[Dict]:
        """Prioritize sources: internal docs (including enterprise) > external web sources > research"""
        internal_sources = []
        external_sources = []
        synthesized_sources = []
        research_sources = []

        for item in content_items:
            metadata = item["metadata"]
            source = metadata.get("source", "")
            source_type = metadata.get("source_type", metadata.get("type", ""))
            tool_used = metadata.get("tool_used", "")

            if (
                source == "file"
                or source in ["mattermost", "nextcloud", "enterprise", "api", "email_imap"]
                or source_type
                in [
                    "local_document",
                    "mattermost_post",
                    "nextcloud_file",
                    "email_message",
                    "enterprise_email",
                ]
            ):
                internal_sources.append(item)
            elif source == "url" or source_type == "search_result":
                external_sources.append(item)
            elif source_type == "ai_synthesis_with_sources":
                synthesized_sources.append(item)
            else:
                research_sources.append(item)

        internal_sources.sort(key=lambda x: x["score"], reverse=True)
        external_sources.sort(key=lambda x: x["score"], reverse=True)
        synthesized_sources.sort(key=lambda x: x["score"], reverse=True)
        research_sources.sort(key=lambda x: x["score"], reverse=True)

        return internal_sources + synthesized_sources + external_sources + research_sources

    def _get_source_description_for_synthesis(self, metadata: Dict) -> str:
        """Generate a brief source description for synthesis prompts"""
        source = metadata.get("source", "")
        source_type = metadata.get("type", "")

        if source == "file":
            return f"internal document ({metadata.get('filename', 'unknown')})"
        elif source == "mattermost" or source_type == "mattermost_post":
            return f"enterprise chat message from {metadata.get('user_id', 'unknown user')}"
        elif source == "email_imap" or source_type in ["email_message", "enterprise_email"]:
            return f"enterprise email from {metadata.get('sender', metadata.get('from', 'unknown sender'))}"
        elif source == "nextcloud" or source_type == "nextcloud_file":
            return f"enterprise file from Nextcloud ({metadata.get('filename', metadata.get('name', 'unknown'))})"
        elif source == "url" or source_type == "search_result":
            return f"web source ({metadata.get('title', metadata.get('url', 'unknown'))})"
        else:
            return f"research finding ({source or source_type or 'unknown'})"

    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain name from URL for use as fallback title"""
        try:
            from urllib.parse import urlparse

            if not url or url == "Unknown URL":
                return ""

            parsed = urlparse(url)
            domain = parsed.netloc

            if domain.startswith("www."):
                domain = domain[4:]

            return domain.capitalize() if domain else ""
        except Exception:
            return ""

    def _generate_internal_search_queries(self, original_question: str, research_plan: Any) -> List[str]:
        """Generate targeted search queries for comprehensive content discovery"""

        actionable_tasks = []
        if (
            research_plan
            and isinstance(research_plan, dict)
            and "research_investigation_areas" in research_plan
            and isinstance(research_plan["research_investigation_areas"], list)
        ):
            actionable_tasks = research_plan["research_investigation_areas"]

        research_plan_length = len(actionable_tasks) if actionable_tasks else 5
        research_questions = (
            "\n - ".join(
                [f'{task.get("research_focus", "")}: {task.get("business_rationale", "")}' for task in actionable_tasks]
            )
            if actionable_tasks
            else original_question
        )

        query_prompt = f"""
As an expert internal research assistant, generate {research_plan_length}-{2 * research_plan_length} highly specific and targeted search queries to comprehensively investigate the following within internal enterprise services (e.g., chat logs, emails, file shares, CRM records, internal wikis, project management tools):

Original Research Question: "{original_question}"

Key Research Plan Questions:
- {research_questions}

Each query should be designed to uncover relevant internal communications, documents, and data, covering the following aspects:

1.  **Direct References & Project/Topic Names:** Queries that directly use exact names of projects, initiatives, departments, or specific terms mentioned in the original question or research plan.
2.  **Key Personnel & Teams:** Queries involving the names of individuals, teams, or departments likely to be involved in discussions or holding relevant information.
3.  **Decision-Making & Approvals:** Queries to find discussions, emails, or documents related to specific decisions, approvals, justifications, or policy changes.
4.  **Problem Solving & Issue Resolution:** Queries focused on identifying discussions around challenges, solutions, bug reports, customer issues, or internal incidents.
5.  **Documentation & Data Artifacts:** Queries for specific document types (e.g., "meeting minutes", "proposal", "report", "spec", "SOP"), file names, or data points within CRM/databases.
6.  **Timeline & Date-Specific Information:** Queries incorporating dates, date ranges, or time-sensitive keywords (e.g., "Q1 2024", "last month", "since June 2023") if relevant to the research.
7.  **Rationale & Background Discussions:** Queries aiming to uncover the "why" behind decisions, the context of projects, or early-stage brainstorming.
8.  **Internal Stakeholder Communications:** Queries focusing on discussions between specific internal groups (e.g., "sales marketing sync", "engineering product review").

Prioritize using exact phrases where possible (e.g., by enclosing in quotes for tools that support it). Consider variations in how terms might be used internally (e.g., abbreviations, internal jargon). Avoid overly generic terms that would yield too many irrelevant results. Assume basic keyword search functionality is available across services.

Return only the queries, one per line. Do not include any additional text, enumeration, or formatting.
"""

        try:
            response = prompt_llm(model=self.model, prompt=query_prompt)
            queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
            queries = queries[: max(7, research_plan_length * 2)]
            workflow_queries = """Based on the user's situation, recommend specific actionable workflow steps from the findings that are relevant to the user's context."""
            queries.append(workflow_queries)
            return queries
        except:
            fallback_tasks = actionable_tasks if actionable_tasks else []
            return [
                original_question,
                f"background {original_question}",
                f"current trends {original_question}",
                f"analysis {original_question}",
                f"implementation {original_question}",
            ] + [task.get("research_focus", original_question) for task in fallback_tasks]

    def _is_internal_source(self, metadata: Dict) -> bool:
        """Determine if a source is internal/enterprise based on metadata"""
        source = metadata.get("source", "")
        source_type = metadata.get("source_type", metadata.get("type", ""))

        return (
            source in ["file", "mattermost", "nextcloud", "enterprise", "email_imap"]
            or source_type
            in ["local_document", "mattermost_post", "nextcloud_file", "internal_document", "email_message", "enterprise_email"]
            or "mattermost" in str(metadata.get("filename", "")).lower()
            or "internal" in str(metadata.get("query_context", "")).lower()
        )

    def _build_theme_source_registry(self, content_items: List[Dict]) -> Dict[str, Dict[str, str]]:
        """Build source registry for theme-specific citations using simplified IDs."""
        theme_sources = {}

        for item in content_items:
            source_id = self._get_source_citation_id(item)
            doc_id = item["doc_id"]

            if source_id == "skip":
                continue

            if doc_id in self.citation_registry.documents:
                source_info = self.citation_registry.documents[doc_id].source_info
                simple_id = self._doc_id_to_simple_id.get(doc_id)
                if simple_id:
                    theme_sources[simple_id] = source_info
            else:
                logger.warning(f"Document {doc_id} not found in citation registry")

        return theme_sources

    def _get_source_citation_id(self, item: Dict) -> str:
        """Register document in unified citation registry and return registration status.

        Returns "skip" for AI synthesis documents, "registered" for regular documents.
        """
        doc_id = item["doc_id"]
        source_info = self._extract_source_info(item)

        if source_info.get("type") == "skip_citation":
            return "skip"
        
        if source_info.get("type") == "ai_synthesis_with_sources":
            source_doc_ids = source_info.get("source_document_ids", [])
            logger.debug(f"AI synthesis has {len(source_doc_ids)} source documents to register")

            for source_doc_id in source_doc_ids:
                if self.vector_store:
                    try:
                        doc_data = self.vector_store.get_document(source_doc_id)
                        if doc_data:
                            source_metadata = doc_data.get("metadata", {})
                            source_info_extracted = self._extract_source_info(
                                {"doc_id": source_doc_id, "metadata": source_metadata}
                            )
                            self.citation_registry.register_document(
                                doc_id=source_doc_id, source_info=source_info_extracted, document_type="regular"
                            )
                    except Exception as e:
                        logger.debug(f"Could not register source document {source_doc_id}: {e}")

            self.citation_registry.register_document(
                doc_id=doc_id, source_info=source_info, underlying_docs=source_doc_ids, document_type="ai_synthesis"
            )
            return "skip"

        elif source_info.get("type") == "ai_synthesis":
            self.citation_registry.register_document(
                doc_id=doc_id, source_info=source_info, document_type="ai_synthesis"
            )
            return "skip"

        self.citation_registry.register_document(doc_id=doc_id, source_info=source_info, document_type="regular")
        return "registered"


    def _extract_source_info(self, item: Dict) -> Dict[str, str]:
        """Extract source information for citations with enhanced support for all source types"""
        metadata = item["metadata"]

        if metadata.get("source") == "mattermost" or metadata.get("type") == "mattermost_post":
            username = metadata.get("user_name", metadata.get("username", metadata.get("user_id", "Unknown User")))
            channel_name = metadata.get("channel_name", metadata.get("channel_display_name", metadata.get("channel_id", "Unknown Channel")))
            team_name = metadata.get("team_name", metadata.get("team_display_name", ""))
            return {
                "type": "enterprise_chat",
                "title": f"Mattermost Message",
                "channel": channel_name,
                "channel_slug": metadata.get("channel_id", metadata.get("channel_slug", "")),
                "team": team_name,
                "team_slug": metadata.get("team_id", metadata.get("team_slug", "")),
                "user": username,
                "message_preview": metadata.get("message_preview", ""),
                "description": f"Mattermost message from {username}",
                "timestamp": metadata.get("timestamp", ""),
            }
        elif metadata.get("source") == "email_imap" or metadata.get("type") == "email_message":
            sender = metadata.get("sender", metadata.get("from", "Unknown Sender"))
            subject = metadata.get("subject", metadata.get("title", ""))
            date = metadata.get("date", metadata.get("timestamp", "Unknown Date"))
            email_id = metadata.get("email_id", metadata.get("id", "unknown"))
            to_raw = metadata.get("to", metadata.get("recipients", ""))
            if isinstance(to_raw, list):
                to_raw = ", ".join(str(t) for t in to_raw if t)
            if not subject or subject.strip() == "":
                first_line = (item.get("content") or "").split("\n", 1)[0].strip()
                subject = first_line or "Email Message"
            return {
                "type": "enterprise_email",
                "title": subject or "Email Message",
                "sender": sender,
                "from": sender,
                "to": to_raw,
                "date": date,
                "source": "email_imap",
                "description": f"Email from {sender}",
                "timestamp": metadata.get("timestamp", ""),
                "email_id": email_id,
            }
        elif metadata.get("source_type") == "local_document":
            file_path = metadata.get("file_path", "Unknown Path")
            relative_path = metadata.get("relative_path", metadata.get("filename", "Unknown"))
            return {
                "type": "internal",
                "title": relative_path,
                "path": file_path,
                "description": f"Local document: {relative_path}",
                "timestamp": metadata.get("ingestion_time", metadata.get("timestamp", "")),
                "folder_path": metadata.get("folder_path", ""),
                "file_size": metadata.get("file_size_bytes", 0),
            }

        elif metadata.get("source") == "file":
            if metadata.get("url"):
                return {
                    "type": "external",
                    "title": metadata.get("title", metadata.get("filename", "Web Document")),
                    "url": metadata.get("url"),
                    "description": f"Downloaded web document: {metadata.get('title', metadata.get('filename', 'Unknown'))}",
                    "timestamp": metadata.get("timestamp", ""),
                }
            else:
                return {
                    "type": "internal",
                    "title": metadata.get("filename", "Internal Document"),
                    "path": metadata.get(
                        "original_path", metadata.get("file_path", metadata.get("path", "Unknown Path"))
                    ),
                    "description": f"Internal document: {metadata.get('filename', 'Unknown')}",
                    "timestamp": metadata.get("timestamp", ""),
                }

        elif metadata.get("source") == "url":
            title = (
                metadata.get("title")
                or metadata.get("filename")
                or self._extract_domain_from_url(metadata.get("url", ""))
                or "Web Source"
            )
            return {
                "type": "external",
                "title": title,
                "url": metadata.get("url", "Unknown URL"),
                "description": f"Web source: {title}",
                "timestamp": metadata.get("timestamp", ""),
            }

        elif metadata.get("type") == "search_result":
            return {
                "type": "external",
                "title": metadata.get("title", "Search Result"),
                "url": metadata.get("url", "Unknown URL"),
                "description": f"Search result: {metadata.get('title', 'Unknown')}",
                "search_rank": metadata.get("search_rank", "N/A"),
                "timestamp": metadata.get("timestamp", ""),
            }

        elif metadata.get("source") == "nextcloud" or metadata.get("type") == "nextcloud_file":
            nextcloud_path = metadata.get(
                "nextcloud_path",
                metadata.get(
                    "path",
                    metadata.get("file_path", metadata.get("original_path", "Unknown Path")),
                ),
            )

            title = metadata.get("nextcloud_filename")
            if not title and nextcloud_path:
                title = nextcloud_path.split("/")[-1] if "/" in nextcloud_path else nextcloud_path
            if not title:
                title = metadata.get("filename", metadata.get("name", "Enterprise File"))

            if title and title.startswith("tmp") and len(title) < 20:
                if nextcloud_path and "/" in nextcloud_path:
                    path_name = nextcloud_path.split("/")[-1]
                    if path_name and not path_name.startswith("tmp"):
                        title = path_name
                    else:
                        title = "Nextcloud File"
                else:
                    title = "Nextcloud File"

            return {
                "type": "enterprise_file",
                "title": title,
                "path": nextcloud_path,
                "server": "Nextcloud",
                "description": f"Nextcloud file: {title}",
                "timestamp": metadata.get("timestamp", metadata.get("last_modified", "")),
            }

        elif metadata.get("source") == "filebrowser" or metadata.get("service_name") == "filebrowser":
            filebrowser_path = metadata.get("original_path", metadata.get("file_path", "Unknown Path"))

            title = metadata.get("file_name")
            if not title and filebrowser_path:
                title = filebrowser_path.split("/")[-1] if "/" in filebrowser_path else filebrowser_path
            if not title:
                title = metadata.get("filename", "FileBrowser File")

            if title and title.startswith("tmp") and len(title) < 20:
                if filebrowser_path and "/" in filebrowser_path:
                    path_name = filebrowser_path.split("/")[-1]
                    if path_name and not path_name.startswith("tmp"):
                        title = path_name
                    else:
                        title = "FileBrowser File"
                else:
                    title = "FileBrowser File"

            return {
                "type": "enterprise_file",
                "title": title,
                "path": filebrowser_path,
                "server": "FileBrowser",
                "description": f"FileBrowser file: {title}",
                "timestamp": metadata.get("timestamp", ""),
            }

        elif metadata.get("tool_used") == "smart_analysis" or metadata.get("type") in [
            "ai_synthesis_with_sources",
            "ai_synthesis",
        ]:
            source_doc_ids = metadata.get("source_document_ids", [])
            docs_analyzed = (
                metadata.get("docs_analyzed") or len(source_doc_ids) or metadata.get("documents_analyzed", 0)
            )

            synthesis_type = metadata.get("type")
            if not synthesis_type:
                synthesis_type = "ai_synthesis_with_sources" if source_doc_ids else "ai_synthesis"

            if synthesis_type == "ai_synthesis_with_sources" or source_doc_ids:
                return {
                    "type": "ai_synthesis_with_sources",
                    "title": metadata.get("title", "AI Research Synthesis"),
                    "source_tool": metadata.get("tool_used", "smart_analysis"),
                    "description": f"AI-generated analysis synthesized from {docs_analyzed} documents",
                    "docs_analyzed": docs_analyzed,
                    "source_document_ids": source_doc_ids,
                    "synthesis_method": metadata.get("synthesis_method", "vector_search"),
                    "timestamp": metadata.get("timestamp", ""),
                }
            else:
                return {
                    "type": synthesis_type,
                    "title": metadata.get("title", "AI Research Analysis"),
                    "source_tool": metadata.get("tool_used", "smart_analysis"),
                    "description": f"AI-generated research analysis",
                    "docs_analyzed": docs_analyzed,
                    "synthesis_method": metadata.get("synthesis_method", "vector_search"),
                    "timestamp": metadata.get("timestamp", ""),
                }

        elif metadata.get("source") in ["enterprise", "api"] or (metadata.get("tool_used") and metadata.get("tool_used") != "local_document_search"):
            tool_used = metadata.get("tool_used", "Unknown Tool")
            api_type = metadata.get("api_type", "")
            query_context = metadata.get("query_context", "")
            service_name = metadata.get("service_name", "")

            if tool_used == "enhanced_enterprise_api":
                if metadata.get("source") == "mattermost":
                    username = metadata.get("user_name", metadata.get("username", metadata.get("user_id", "Unknown User")))
                    channel_name = metadata.get("channel_name", metadata.get("channel_id", "Unknown Channel"))
                    team_name = metadata.get("team_name", "")
                    return {
                        "type": "enterprise_chat",
                        "title": "Mattermost Message",
                        "channel": channel_name,
                        "team": team_name,
                        "user": username,
                        "description": f"Mattermost message from {username}",
                        "timestamp": metadata.get("timestamp", ""),
                    }
                elif metadata.get("source") == "email_imap":
                    sender = metadata.get("sender", metadata.get("from", "Unknown Sender"))
                    subject = metadata.get("subject", metadata.get("title", ""))
                    date = metadata.get("date", metadata.get("timestamp", "Unknown Date"))
                    to_raw = metadata.get("to", metadata.get("recipients", ""))
                    if isinstance(to_raw, list):
                        to_raw = ", ".join(str(t) for t in to_raw if t)
                    if not subject or subject.strip() == "":
                        content = item.get("content") or ""
                        first_line = content.split("\n", 1)[0].strip() if content else ""
                        subject = first_line or "Email Message"
                    return {
                        "type": "enterprise_email",
                        "title": subject or "Email Message",
                        "sender": sender,
                        "from": sender,
                        "to": to_raw,
                        "date": date,
                        "description": f"Email from {sender}",
                        "timestamp": metadata.get("timestamp", ""),
                    }
                elif metadata.get("source") == "nextcloud":
                    service_name = "Nextcloud"
                elif metadata.get("source") == "filebrowser":
                    service_name = "FileBrowser"
                elif "file" in str(metadata.get("original_path", "")).lower():
                    service_name = "File Server"
                elif metadata.get("filename"):
                    service_name = "Enterprise File System"
                elif "search" in str(query_context).lower():
                    service_name = "Enterprise Search"

                if not service_name:
                    if metadata.get("sender") or metadata.get("subject"):
                        sender = metadata.get("sender", metadata.get("from", "Unknown Sender"))
                        subject = metadata.get("subject", "Email Message")
                        date = metadata.get("date", metadata.get("timestamp", ""))
                        to_raw = metadata.get("to", metadata.get("recipients", ""))
                        if isinstance(to_raw, list):
                            to_raw = ", ".join(str(t) for t in to_raw if t)
                        return {
                            "type": "enterprise_email",
                            "title": subject,
                            "sender": sender,
                            "from": sender,
                            "to": to_raw,
                            "date": date,
                            "description": f"Email from {sender}",
                            "timestamp": metadata.get("timestamp", ""),
                        }
                    if metadata.get("user_name") or metadata.get("channel_name"):
                        return {
                            "type": "enterprise_chat",
                            "title": "Mattermost Message",
                            "user": metadata.get("user_name", "Unknown User"),
                            "channel": metadata.get("channel_name", "Unknown Channel"),
                            "team": metadata.get("team_name", ""),
                            "description": f"Mattermost message from {metadata.get('user_name', 'Unknown')}",
                            "timestamp": metadata.get("timestamp", ""),
                        }

            if service_name:
                title = f"{service_name} Enterprise Data"
            elif api_type:
                title = f"Enterprise {api_type.title()} Data"
            elif query_context:
                context_snippet = query_context[:50] + "..." if len(query_context) > 50 else query_context
                title = f"Enterprise Search: {context_snippet}"
            else:
                title = "Enterprise API Data"

            return {
                "type": "enterprise_api",
                "title": title,
                "source_tool": tool_used,
                "description": f"Enterprise data retrieved via {tool_used}",
                "timestamp": metadata.get("timestamp", ""),
            }

        elif metadata.get("type") == "research_finding" or not metadata.get("source"):
            if metadata.get("synthesized_from_sources"):
                return {
                    "type": "synthesis",
                    "title": "Research Synthesis",
                    "description": f"Analysis synthesized from {len(metadata.get('synthesized_from_sources', []))} sources",
                    "source_count": len(metadata.get("synthesized_from_sources", [])),
                    "timestamp": metadata.get("timestamp", ""),
                }
            else:
                # Skip citations for generic research findings to avoid "Unknown source" entries
                return {
                    "type": "skip_citation",
                    "title": "Research Finding",
                    "description": "AI-generated research finding (citation skipped)",
                    "timestamp": metadata.get("timestamp", ""),
                }

        else:
            return {
                "type": "unknown",
                "title": f"Unknown Source ({metadata.get('type', 'unspecified')})",
                "description": f"Source type: {metadata.get('source', metadata.get('type', 'unknown'))}",
                "timestamp": metadata.get("timestamp", ""),
            }

    def _get_retrieval_description(self, source_info: Dict[str, str]) -> str:
        """Get a concise description of how the source was retrieved"""
        source_type = source_info.get("type", "unknown")

        if source_type == "enterprise_file":
            return f"{source_info.get('title', 'File')} ({source_info.get('server', 'Enterprise')})"
        elif source_type == "enterprise_chat":
            user = source_info.get("user", "Unknown")
            channel = source_info.get("channel", "Unknown")
            return f"Message from {user} in #{channel}"
        elif source_type == "external":
            return source_info.get("title", "Web Source")
        else:
            return source_info.get("description", "Unknown Source")

    def _track_synthesis_metadata(self, theme: str, content_items: List[Dict]):
        """Track metadata for synthesized content"""

        for item in content_items:
            metadata = item["metadata"]
            doc_id = item["doc_id"]

            if doc_id and doc_id not in self.evidence_metadata["document_ids"]:
                self.evidence_metadata["document_ids"].append(doc_id)

            if metadata.get("source") == "url":
                url = metadata.get("url")
                if url and url not in self.evidence_metadata["document_ids"]:
                    self.evidence_metadata["document_ids"].append(url)

                self.evidence_metadata["evidence_map"]["web_sources"].append(
                    {
                        "url": url,
                        "title": metadata.get("title", "Unknown"),
                        "doc_id": doc_id,
                        "relevance_score": item["score"],
                        "content_length": len(item["content"]),
                        "theme_coverage": [theme],
                        "date_accessed": metadata.get("timestamp", ""),
                        "content_type": metadata.get("content_type", "unknown"),
                    }
                )

            elif metadata.get("source") == "file":
                file_path = metadata.get("original_path") or metadata.get("file_path")
                if file_path and file_path not in self.evidence_metadata["document_ids"]:
                    self.evidence_metadata["document_ids"].append(file_path)

                self.evidence_metadata["evidence_map"]["internal_sources"].append(
                    {
                        "file_path": file_path,
                        "filename": metadata.get("filename", "Unknown"),
                        "doc_id": doc_id,
                        "relevance_score": item["score"],
                        "content_length": len(item["content"]),
                        "theme_coverage": [theme],
                        "content_type": metadata.get("content_type", "unknown"),
                    }
                )

            else:
                self.evidence_metadata["evidence_map"]["research_findings"].append(
                    {
                        "source": metadata.get("tool_used", "unknown"),
                        "doc_id": doc_id,
                        "relevance_score": item["score"],
                        "theme_coverage": [theme],
                        "finding_type": metadata.get("type", "research_finding"),
                    }
                )

    def _extract_clean_question(self, original_question: str) -> str:
        """Extract the actual question from an enhanced query or return the original question"""
        if "QUESTION:" in original_question:
            parts = original_question.split("QUESTION:", 1)
            if len(parts) > 1:
                clean_question = parts[1].strip()
                return clean_question.split("\n")[0].strip()
        return original_question

    def _assemble_final_report(
        self,
        synthesized_sections: Dict[str, str],
        evidence_summary: Dict[str, Any],
        context: ResearchContext,
    ) -> str:
        """Assemble the final clean, readable report. Theme sections are added first;
        then a single step-by-step personalized workflow is generated from all themes combined."""

        exec_summary = self._generate_executive_summary(synthesized_sections, context, evidence_summary)

        report_parts = []

        clean_question = self._extract_clean_question(context.original_question)

        report_parts.append(f"# Research Report: {clean_question}\n")

        report_parts.append("## Executive Summary\n")
        report_parts.append(f"{exec_summary}\n")

        report_parts.append("## Analysis\n")

        theme_order = [
            "background",
            "current_trends",
            "analysis",
            "implementation",
            "future_outlook",
            "general",
        ]
        theme_titles = {
            "background": "Background & Context",
            "current_trends": "Current Trends & Developments",
            "analysis": "Key Research Findings",
            "implementation": "Practical Applications",
            "future_outlook": "Future Outlook",
            "general": "Additional Insights",
        }

        for theme in theme_order:
            if theme in synthesized_sections:
                title = theme_titles.get(theme, theme.replace("_", " ").title())
                report_parts.append(f"### {title}\n")
                section_text = synthesized_sections[theme]
                section_text = re.sub(
                    r"\n*###\s*Personalized Workflow Recommendations:?\s*\n*",
                    "\n\n",
                    section_text,
                    flags=re.IGNORECASE,
                )
                report_parts.append(f"{section_text}\n")

        combined_synthesis = "\n\n".join(synthesized_sections.values())
        logger.info("Generating personalized workflow from %d themes (%d chars)...",
                     len(synthesized_sections), len(combined_synthesis))
        workflow_content = self.personalized_workflow_synthesis(
            combined_synthesis, context.original_question
        )
        if workflow_content and workflow_content.strip():
            workflow_content = re.sub(
                r"^###\s*Personalized Workflow (?:Recommendations|Steps):?\s*\n*",
                "",
                workflow_content.strip(),
                flags=re.IGNORECASE,
            )
            workflow_content = self._validate_and_format_workflow(workflow_content)
            report_parts.append("### Personalized Workflow Recommendations:\n\n")
            report_parts.append(f"{workflow_content}\n")
        else:
            logger.error("Personalized workflow generation returned empty; section omitted.")

        main_report = "\n".join(report_parts)
        main_report = self._normalize_doc_citations(main_report)
        main_report = self._resolve_simplified_ids(main_report)

        final_report, citation_assignments = self.citation_registry.finalize_citations(main_report)

        references_section = self.citation_registry.generate_references_section()

        if citation_assignments:
            final_report += "\n" + references_section
        else:
            final_report += "\n\n---\n"
            final_report += "*Analysis completed without external source citations.*"

        return final_report

    def _generate_executive_summary(
        self,
        synthesized_sections: Dict[str, str],
        context: ResearchContext,
        evidence_summary: Dict[str, Any],
        max_len: int = 1000,
    ) -> str:
        """Generate executive summary from synthesized content"""

        internal_sources_count = len(self.evidence_metadata["evidence_map"]["internal_sources"])
        web_sources_count = len(self.evidence_metadata["evidence_map"]["web_sources"])

        clean_question = self._extract_clean_question(context.original_question)
        prompt_ready_sections = {
            key: self._replace_real_ids_with_simplified(
                value[:max_len] + "..." if len(value) > max_len else value
            )
            for key, value in synthesized_sections.items()
        }
        available_sources = self._format_available_sources_for_prompt()

        summary_prompt = f"""
        Generate a concise executive summary for this research report:
        
        Research Question: {clean_question}
        
        Key Findings from Analysis (with internal sources prioritized):
        {json.dumps(prompt_ready_sections, indent=2)}

        Available Sources for Citations:
        {available_sources}
        
        Research Scope: {evidence_summary['total_sources']} sources analyzed 
        ({internal_sources_count} internal documents, {web_sources_count} external sources)
        
        Requirements:
        - 2-3 paragraphs maximum
        - Lead with the most important findings, especially from internal sources
        - Emphasize insights from enterprise/internal documents when available
        - Include confidence level based on evidence strength
        - Natural, professional tone
        - Focus on insights and implications, not process
        - Mention source prioritization (internal over external) if relevant
        - Include 1-3 inline citations using [DOC:src_N] for the strongest factual claims
        - Only use source IDs from the Available Sources list
        
        Generate executive summary:
        """

        try:
            summary = prompt_llm(model=self.model, prompt=summary_prompt)
            summary = self._normalize_doc_citations(summary)
            summary = self._resolve_simplified_ids(summary)
            return summary
        except Exception as e:
            return f"Key findings synthesized from {evidence_summary['total_sources']} sources across {len(synthesized_sections)} research themes. Analysis reveals comprehensive insights addressing the research question with evidence-based recommendations."

    def _finalize_metadata(self, context: ResearchContext, action_plan, thematic_content: Dict[str, List[Dict]]):
        """Finalize comprehensive metadata for the report"""

        self.evidence_metadata["content_synthesis"] = {
            "total_sources_used": len(self.evidence_metadata["document_ids"]),
            "primary_sources": len(self.evidence_metadata["evidence_map"]["web_sources"]),
            "secondary_sources": len(self.evidence_metadata["evidence_map"]["internal_sources"]),
            "research_findings": len(self.evidence_metadata["evidence_map"]["research_findings"]),
            "content_themes": list(thematic_content.keys()),
            "synthesis_confidence": ("high" if len(self.evidence_metadata["document_ids"]) > 5 else "medium"),
            "thematic_coverage": {theme: len(items) for theme, items in thematic_content.items()},
        }

        vector_searches = 0

        self.evidence_metadata["research_methodology"] = {
            "vector_searches_performed": vector_searches,
            "content_clustering_method": "thematic",
            "synthesis_approach": "multi-source_cross_validation_with_citations",
            "quality_filters_applied": ["relevance>0.7", "content_length>100"],
            "source_prioritization": "internal_documents_first",
            "citation_system": "markdown_footnotes",
            "total_operations": len(context.findings),
            "successful_operations": sum(
                1 for finding in context.findings.values() if isinstance(finding, dict) and finding.get("success", True)
            ),
            "citations_generated": len(self.citation_registry.documents),
            "internal_vs_external_ratio": self._calculate_source_ratio(),
        }

    def _calculate_source_ratio(self) -> Dict[str, float]:
        """Calculate ratio of internal vs external sources"""
        internal_count = len(self.evidence_metadata["evidence_map"]["internal_sources"])
        external_count = len(self.evidence_metadata["evidence_map"]["web_sources"])
        total = internal_count + external_count

        if total == 0:
            return {"internal_ratio": 0.0, "external_ratio": 0.0}

        return {"internal_ratio": internal_count / total, "external_ratio": external_count / total}

    def get_evidence_metadata(self) -> Dict[str, Any]:
        """Get comprehensive evidence metadata for the generated report"""
        metadata = self.evidence_metadata.copy()

        citation_stats = self.citation_registry.get_statistics()
        metadata["citation_registry"] = {
            "total_citations": citation_stats["total_citations"],
            "unique_cited_documents": citation_stats["unique_cited_documents"],
            "total_documents": citation_stats["total_documents"],
            "source_mapping": {
                doc_id: {
                    "citation_number": self.citation_registry.get_citation_number(doc_id),
                    "source_type": doc_info.source_info.get("type", "unknown"),
                    "title": doc_info.source_info.get("title", "Untitled"),
                    "document_type": doc_info.document_type,
                }
                for doc_id, doc_info in self.citation_registry.documents.items()
            },
        }
        return metadata
