import datetime
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from drflow.agents.drflow_agent.agent_tools.base import ResearchContext
from drflow.agents.utils import prompt_llm

logger = logging.getLogger(__name__)


def _normalize_action_parameters(params: Any) -> Dict[str, Any]:
    """Ensure parameters are a dict (LLM/plan JSON sometimes emits a string)."""
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    if isinstance(params, str):
        stripped = params.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return {"query": stripped}
    return {}


class ActionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ActionType(Enum):
    WEB_SEARCH = "web_search"
    ENTERPRISE_API = "enterprise_api"
    MCP_QUERY = "mcp_query"
    URL_FETCH = "url_fetch"
    FILE_DOWNLOAD = "file_download"
    DATA_ANALYSIS = "data_analysis"
    CONTEXT_SYNTHESIS = "context_synthesis"
    LOCAL_DOCUMENT_SEARCH = "local_document_search"
    LOCAL_FILE_ANALYSIS = "local_file_analysis"
    FLOWCHART_DESIGN = "flowchart_design"
    FLOWCHART_PERSONALIZATION = "flowchart_personalization"


@dataclass
class Action:
    """Represents a single executable action"""

    id: str
    type: ActionType
    description: str
    parameters: Dict[str, Any]
    status: ActionStatus = ActionStatus.PENDING
    priority: float = 0.5
    dependencies: List[str] = field(default_factory=list)
    expected_output: str = ""
    actual_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    completion_time: Optional[str] = None
    iteration_completed: Optional[int] = None
    created_from_research_step: Optional[str] = None
    preferred_tool: Optional[str] = None
    score: float = 0.0
    stage: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert Action to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "parameters": self.parameters,
            "status": self.status.value,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "error": self.error,
            "execution_time": self.execution_time,
            "completion_time": self.completion_time,
            "iteration_completed": self.iteration_completed,
            "created_from_research_step": self.created_from_research_step,
            "preferred_tool": self.preferred_tool,
            "score": self.score,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create Action from dictionary"""
        return cls(
            id=data["id"],
            type=ActionType(data["type"]),
            description=data["description"],
            parameters=_normalize_action_parameters(data.get("parameters")),
            status=ActionStatus(data["status"]),
            priority=data["priority"],
            dependencies=data["dependencies"],
            expected_output=data["expected_output"],
            actual_output=data.get("actual_output"),
            error=data.get("error"),
            execution_time=data.get("execution_time"),
            completion_time=data.get("completion_time"),
            iteration_completed=data.get("iteration_completed"),
            created_from_research_step=data.get("created_from_research_step"),
            preferred_tool=data.get("preferred_tool"),
            score=data.get("score", 0.0),
            stage=data.get("stage"),
        )


@dataclass
class ActionPlan:
    """Represents a complete action plan"""

    id: str
    research_query: str
    actions: List[Action] = field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 10
    completed_actions: int = 0
    failed_actions: int = 0

    def add_action(self, action: Action):
        """Add an action to the plan"""
        self.actions.append(action)

    def get_next_actions(self, max_concurrent: int = 3) -> List[Action]:
        """Get the next actions that can be executed with enhanced scoring"""
        ready_actions = []

        for action in self.actions:
            if action.status == ActionStatus.PENDING:
                dependencies_met = all(
                    (
                        self.get_action_by_id(dep_id) is None
                        or self.get_action_by_id(dep_id).status == ActionStatus.COMPLETED
                    )
                    for dep_id in action.dependencies
                )

                if dependencies_met:
                    ready_actions.append(action)

        for action in ready_actions:
            action.score = self._calculate_action_score(action)

        ready_actions.sort(key=lambda a: a.score, reverse=True)
        return ready_actions[:max_concurrent]

    def get_action_by_id(self, action_id: str) -> Optional[Action]:
        """Get an action by its ID"""
        for action in self.actions:
            if action.id == action_id:
                return action
        return None

    def mark_completed(
        self,
        action_id: str,
        output: Dict[str, Any],
        execution_time: Optional[float] = None,
        iteration: Optional[int] = None,
    ):
        """Mark an action as completed with timing information"""
        action = self.get_action_by_id(action_id)
        if action:
            action.status = ActionStatus.COMPLETED
            action.actual_output = output
            action.completion_time = datetime.datetime.now().isoformat()
            if execution_time is not None:
                action.execution_time = execution_time
            if iteration is not None:
                action.iteration_completed = iteration
            self.completed_actions += 1

    def mark_failed(
        self, action_id: str, error: str, execution_time: Optional[float] = None, iteration: Optional[int] = None
    ):
        """Mark an action as failed with timing information"""
        action = self.get_action_by_id(action_id)
        if action:
            action.status = ActionStatus.FAILED
            action.error = error
            action.completion_time = datetime.datetime.now().isoformat()
            if execution_time is not None:
                action.execution_time = execution_time
            if iteration is not None:
                action.iteration_completed = iteration
            self.failed_actions += 1

    def is_complete(self) -> bool:
        """Check if the action plan is complete"""
        pending_actions = [a for a in self.actions if a.status == ActionStatus.PENDING]
        in_progress_actions = [a for a in self.actions if a.status == ActionStatus.IN_PROGRESS]
        return len(pending_actions) == 0 and len(in_progress_actions) == 0

    def _calculate_action_score(self, action: Action) -> float:
        """Calculate information gain score for an action"""
        score = action.priority

        novelty_score = self._calculate_novelty_score(action)
        score *= 1.0 + novelty_score * 0.5

        diversity_score = self._calculate_source_diversity_score(action)
        score *= 1.0 + diversity_score * 0.3

        action_age = self.actions.index(action) if action in self.actions else 0
        recency_factor = 1.0 - (action_age / max(len(self.actions), 1)) * 0.1
        score *= recency_factor

        return score

    def _calculate_novelty_score(self, action: Action) -> float:
        """Calculate how novel/unique this action is compared to completed actions"""
        if not self.actions:
            return 1.0

        completed_actions = [a for a in self.actions if a.status == ActionStatus.COMPLETED]
        if not completed_actions:
            return 1.0

        similarities = []
        for completed in completed_actions:
            if completed.type == action.type:
                if action.type in [ActionType.WEB_SEARCH, ActionType.ENTERPRISE_API]:
                    query1 = action.parameters.get("query", "").lower()
                    query2 = completed.parameters.get("query", "").lower()
                    if query1 and query2:
                        words1 = set(query1.split())
                        words2 = set(query2.split())
                        if words1 and words2:
                            overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                            similarities.append(overlap)

        return 1.0 - max(similarities) if similarities else 1.0

    def _calculate_source_diversity_score(self, action: Action) -> float:
        """Calculate source diversity to balance internal and external sources"""
        completed_actions = [a for a in self.actions if a.status == ActionStatus.COMPLETED]

        if not completed_actions:
            return 1.0 if action.type in [ActionType.ENTERPRISE_API, ActionType.MCP_QUERY] else 0.5

        internal_count = sum(
            1 for a in completed_actions if a.type in [ActionType.ENTERPRISE_API, ActionType.MCP_QUERY]
        )
        external_count = sum(1 for a in completed_actions if a.type in [ActionType.WEB_SEARCH, ActionType.URL_FETCH])

        total = internal_count + external_count
        if total == 0:
            return 0.5

        internal_ratio = internal_count / total

        if action.type in [ActionType.ENTERPRISE_API, ActionType.MCP_QUERY]:
            return 1.0 - internal_ratio
        else:
            return internal_ratio

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "total_actions": len(self.actions),
            "completed": self.completed_actions,
            "failed": self.failed_actions,
            "pending": len([a for a in self.actions if a.status == ActionStatus.PENDING]),
            "in_progress": len([a for a in self.actions if a.status == ActionStatus.IN_PROGRESS]),
            "current_iteration": self.current_iteration,
            "success_rate": self.completed_actions / len(self.actions) if self.actions else 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert ActionPlan to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "research_query": self.research_query,
            "actions": [action.to_dict() for action in self.actions],
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "completed_actions": self.completed_actions,
            "failed_actions": self.failed_actions,
            "stats": self.get_stats(),
            "timestamp": datetime.datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionPlan":
        """Create ActionPlan from dictionary"""
        plan = cls(
            id=data["id"],
            research_query=data["research_query"],
            current_iteration=data["current_iteration"],
            max_iterations=data["max_iterations"],
            completed_actions=data["completed_actions"],
            failed_actions=data["failed_actions"],
        )
        plan.actions = [Action.from_dict(action_data) for action_data in data["actions"]]
        return plan

    def save_to_file(self, file_path: str) -> None:
        """Save ActionPlan to JSON file"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: str) -> "ActionPlan":
        """Load ActionPlan from JSON file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def _get_action_plan_guidelines(available_tool_names):
    """Generate action plan guidelines based on available tools."""

    has_enterprise = any("EnterpriseAPITool" in t for t in available_tool_names)
    has_local = any("LocalFileSearchTool" in t for t in available_tool_names)

    tool_lines = []
    if has_local:
        tool_lines.append("- LocalFileSearchTool (LOCAL_DOCUMENT_SEARCH): uploaded documents, policies, playbooks")
    if has_enterprise:
        tool_lines.append("- EnterpriseAPITool (ENTERPRISE_API): emails, chats, enterprise files, document repos")
    tool_lines.append("- InternetSearchTool (WEB_SEARCH): external web — use only when internal sources are insufficient")
    tool_block = "\n    ".join(tool_lines)

    action_types_str = ", ".join(at.value for at in ActionType)

    return f"""ACTION PLAN GUIDELINES:

    AVAILABLE INTERNAL & EXTERNAL TOOLS:
    {tool_block}

    SOURCE PRIORITY: internal sources (LOCAL + ENTERPRISE) first, web search only as fallback.

    WORKFLOW STAGES:
    1. REQUIREMENT_COLLECTION — find policy/procedural documents with step-by-step workflows.
    2. GENERAL_FLOWCHART_DESIGN — extract the generic workflow from collected documents.
    3. PERSONAL_EVIDENCE_COLLECTION — find user-specific facts (emails, chats, notes, records)
       that resolve conditional branches in the workflow.
    4. FLOWCHART_PERSONALIZATION — personalize the workflow using the collected evidence.

    DIVERSITY IS CRITICAL:
    - Each action MUST use a meaningfully different search query. Rephrasings of the same
      query (e.g. "business procedure for XYZ" vs "XYZ business procedures") are wasteful.
    - Think about what different ASPECTS of the question can be searched independently:
      different sub-topics, different document types, different data stores, different
      entities mentioned in the question, etc.
    - Vary the tools used when multiple are available.
    - For PERSONAL_EVIDENCE_COLLECTION: search for different kinds of user evidence
      (correspondence about specific people/vendors, filed forms, meeting notes,
      scheduling records, approvals, etc.) — not just the same topic query repeated.
    - For REQUIREMENT_COLLECTION: search for different kinds of policy content
      (procedures, checklists, compliance guides, training materials, templates, etc.)
      using different terminology.

    FOR EACH ACTION, SPECIFY:
    1. type — one of: {action_types_str}
    2. description — specific, concrete description of what this action searches for and why
    3. parameters — query terms, top_k, service, etc.
    4. priority — 0.8-1.0 for internal, 0.3-0.5 for web search
    5. expected_output — what kind of information this should return
    6. preferred_tool — which tool to use
    7. stage — REQUIREMENT_COLLECTION or PERSONAL_EVIDENCE_COLLECTION

    Return a JSON array of actions. Just return valid JSON, no other text.
    """  # noqa: E501


class ActionPlanner:
    """Plans and manages action execution separate from research planning"""

    def __init__(self, model: str):
        self.model = model
        self.active_plans: Dict[str, ActionPlan] = {}

    def create_action_plan(
        self, research_plan: Optional[Dict[str, Any]], context: ResearchContext, tool_registry
    ) -> ActionPlan:
        """Create an action plan from a research plan"""

        plan_id = f"action_plan_{hash(context.original_question)}_{len(self.active_plans)}"
        action_plan = ActionPlan(id=plan_id, research_query=context.original_question)

        actionable_tasks = research_plan.get("research_investigation_areas", []) if research_plan else []

        all_actions = []
        if actionable_tasks:
            for i, task in enumerate(actionable_tasks):
                actions = self._generate_actions_for_task(task, tool_registry, context)
                for action in actions:
                    action.created_from_research_step = self._extract_task_id(task)
                    all_actions.append(action)
        else:
            fallback_actions = self._generate_fallback_actions(context, tool_registry)
            all_actions.extend(fallback_actions)

        unique_actions = self._deduplicate_actions(all_actions, [])
        for action in unique_actions:
            action_plan.add_action(action)

        self.active_plans[plan_id] = action_plan
        return action_plan

    def evolve_action_plan(
        self, plan_id: str, new_findings: Dict[str, Any], context: ResearchContext, tool_registry
    ) -> List[Action]:
        """Evolve the action plan based on new findings:
        1) Extract the generic workflow from findings and identify conditional steps.
        2) Add actions to search for personalized evidence to resolve those conditional steps in the personalized workflow.
        3) Add gap-filling actions (policy vs personal evidence) as needed."""

        action_plan = self.active_plans.get(plan_id)
        if not action_plan:
            return []

        condition_actions = self._generate_condition_resolution_actions(context, tool_registry)

        gap_actions = self._generate_adaptive_actions(action_plan, new_findings, context, tool_registry)

        new_actions = condition_actions + gap_actions
        deduplicated_actions = self._deduplicate_actions(new_actions, action_plan.actions)

        for action in deduplicated_actions:
            action_plan.add_action(action)

        return deduplicated_actions

    def _generate_tool_guidelines(self, tool_registry) -> str:
        """Generate dynamic tool selection guidelines from the tool registry"""
        guidelines = []
        for tool in tool_registry.tools:
            tool_name = tool.__class__.__name__
            try:
                purpose = tool.purpose
                guidelines.append(f'    - Use "{tool_name}" for {purpose}')
            except (AttributeError, NotImplementedError):
                guidelines.append(f'    - "{tool_name}" is available')

        return "\n".join(guidelines) if guidelines else "    - Use appropriate tools based on task requirements"

    def _extract_task_id(self, task: Dict[str, Any]) -> str:
        """Extract task identifier from research investigation area"""
        area_id = task.get("area_id")
        if area_id:
            return str(area_id)

        focus = task.get("research_focus", "")
        if focus:
            focus = re.sub(r"[^a-zA-Z0-9\s]", "", focus)
            focus = focus.replace(" ", "_").lower()[:20]
            return f"task_{focus}"

        return "unknown"

    def _generate_actions_for_task(self, task: Dict[str, Any], tool_registry, context: ResearchContext) -> List[Action]:
        """Generate specific actions for a research task"""

        research_focus = task.get("research_focus", "")
        information_needs = task.get("information_needs", [])
        knowledge_sources = task.get("knowledge_sources", ["both"])
        research_approach = task.get("research_approach", "strategic_assessment")
        stage = task.get("stage", "REQUIREMENT_COLLECTION")

        task_id = self._extract_task_id(task)

        tool_guidelines = self._generate_tool_guidelines(tool_registry)
        available_tool_names = [tool.__class__.__name__ for tool in tool_registry.tools]

        original_question = context.original_question if context else ""
        
        prompt = f"""
    Generate 2-3 diverse, targeted actions for this research investigation area.

    ORIGINAL DR QUESTION: {original_question}

    CURRENT STAGE: {stage}
    Research Focus: {research_focus}
    Information Needs: {information_needs}
    Knowledge Sources: {knowledge_sources}
    Available Tools: {available_tool_names}

    Tool Selection Guidelines:
{tool_guidelines}

{_get_action_plan_guidelines(available_tool_names)}

    IMPORTANT: Each action must use a genuinely different search query — not a rephrasing
    of the same terms. Think about different aspects, sub-topics, document types, or
    terminology that would retrieve complementary results.

    Return a JSON array with 2-3 actions.
    """

        try:
            action_configs = prompt_llm_and_parse_json(self.model, prompt)

            actions = []
            for i, config in enumerate(action_configs):
                try:
                    action = Action(
                        id=f"{task_id}_action_{i}",
                        type=ActionType(config.get("type", "web_search")),
                        description=config.get("description", ""),
                        parameters=_normalize_action_parameters(config.get("parameters", {})),
                        priority=config.get("priority", 0.5),
                        expected_output=config.get("expected_output", ""),
                        dependencies=[],
                        preferred_tool=config.get("preferred_tool"),
                        stage=stage,
                    )
                    actions.append(action)
                except Exception as e:
                    logger.warning(f"Error parsing action config. Skipping action: {e}")
                    continue

            if len(actions) > 1:
                detected_deps = self._detect_action_dependencies(actions, research_focus)

                for action in actions:
                    if action.id in detected_deps:
                        action.dependencies = detected_deps[action.id]

            return actions

        except Exception as e:
            logger.error(f"Error generating actions for task: {e}")
            return []

    def _fallback_task_actions(self, task: Dict[str, Any], tool_registry) -> List[Action]:
        """Fallback action generation when LLM fails"""

        research_focus = task.get("research_focus", "")
        task_id = self._extract_task_id(task)
        available_tool_names = [tool.__class__.__name__.lower() for tool in tool_registry.tools]

        actions = []

        if any("localdocument" in tool_name for tool_name in available_tool_names):
            actions.append(
                Action(
                    id=f"{task_id}_local_doc_search",
                    type=ActionType.LOCAL_DOCUMENT_SEARCH,
                    description=f"Search local documents for: {research_focus}",
                    parameters={"query": research_focus, "top_k": 10},
                    priority=0.9,
                    expected_output="Relevant information from local document collection",
                    preferred_tool="LocalFileSearchTool",
                )
            )

        if any("search" in tool_name for tool_name in available_tool_names):
            actions.append(
                Action(
                    id=f"{task_id}_web_search",
                    type=ActionType.WEB_SEARCH,
                    description=f"Search web ONLY if internal docs insufficient: {research_focus}",
                    parameters={"query": research_focus, "num_results": 5},
                    priority=0.3,
                    expected_output="External regulations only if not found internally",
                    preferred_tool="InternetSearchTool",
                )
            )

        if any("enterprise" in tool_name for tool_name in available_tool_names):
            actions.append(
                Action(
                    id=f"{task_id}_enterprise_search",
                    type=ActionType.ENTERPRISE_API,
                    description=f"Search enterprise services for: {research_focus}",
                    parameters={"query": research_focus},
                    priority=0.8,
                    expected_output="Enterprise data related to query",
                    preferred_tool="EnterpriseAPITool",
                )
            )

        return actions

    def _generate_fallback_actions(self, context: ResearchContext, tool_registry) -> List[Action]:
        """Generate basic actions when no research plan is available"""

        query = context.original_question
        available_tool_names = [tool.__class__.__name__ for tool in tool_registry.tools]

        tool_guidelines = self._generate_tool_guidelines(tool_registry)

        available_tool_names = [tool.__class__.__name__ for tool in tool_registry.tools]

        prompt = f"""
    Generate specific executable actions for this research task with SOURCE PRIORITIZATION:

    Task: {query}
    Available Tools: {available_tool_names}

    Tool Selection Guidelines:
{tool_guidelines}

    {_get_action_plan_guidelines(available_tool_names)}
    """

        try:
            action_configs = prompt_llm_and_parse_json(self.model, prompt)

            actions = []
            for i, config in enumerate(action_configs):
                try:
                    action = Action(
                        id=f"fallback_action_{i}",
                        type=ActionType(config.get("type", "web_search")),
                        description=config.get("description", ""),
                        parameters=_normalize_action_parameters(config.get("parameters", {})),
                        priority=config.get("priority", 0.5),
                        expected_output=config.get("expected_output", ""),
                        dependencies=[],
                        preferred_tool=config.get("preferred_tool"),
                        created_from_research_step="fallback_generation",
                    )
                    actions.append(action)
                except Exception as e:
                    logger.warning(f"Error parsing fallback action config: {e}")
                    continue

            return actions

        except Exception as e:
            logger.error(f"Error generating fallback actions: {e}")
            return self._get_basic_fallback_actions(query, tool_registry)

    def _get_basic_fallback_actions(self, query: str, tool_registry) -> List[Action]:
        """Generate very basic fallback actions when LLM fails - PRIORITIZE INTERNAL SOURCES"""

        available_tool_names = [tool.__class__.__name__.lower() for tool in tool_registry.tools]
        actions = []

        if any("local" in tool_name or "file" in tool_name for tool_name in available_tool_names):
            actions.append(
                Action(
                    id="basic_local_search",
                    type=ActionType.LOCAL_DOCUMENT_SEARCH,
                    description=f"Search local documents for: {query}",
                    parameters={"query": query, "top_k": 15},
                    priority=0.95,
                    expected_output="Internal policy documents",
                    preferred_tool="LocalFileSearchTool",
                    created_from_research_step="basic_fallback",
                )
            )

        if any("enterprise" in tool_name for tool_name in available_tool_names):
            actions.append(
                Action(
                    id="basic_enterprise_search",
                    type=ActionType.ENTERPRISE_API,
                    description=f"Search enterprise systems for: {query}",
                    parameters={"query": query},
                    priority=0.85,
                    expected_output="Enterprise search results",
                    preferred_tool="EnterpriseAPITool",
                    created_from_research_step="basic_fallback",
                )
            )

        return actions

    def _extract_generic_workflow_and_conditions(
        self, findings: Dict[str, Any], research_query: str, vector_store=None
    ) -> Dict[str, Any]:
        """1) Extract the generic workflow from policy/workflow findings.
        2) Identify conditional steps in that workflow that need user evidence to resolve.
        Returns { "generic_workflow_steps": [...], "conditional_steps": [...] }.
        Does not mutate findings; doc_ids are preserved for flowchart references."""
        if not findings and not vector_store:
            logger.info("Workflow extraction: no findings and no vector store available")
            return {"generic_workflow_steps": [], "conditional_steps": []}

        summary_parts = []
        if findings:
            for key, finding in findings.items():
                if not isinstance(finding, dict):
                    continue
                text = (
                    finding.get("synthesis")
                    or finding.get("summary")
                    or finding.get("description")
                    or ""
                )
                results_val = finding.get("results")
                if isinstance(results_val, list):
                    for r in results_val[:8]:
                        if isinstance(r, dict):
                            chunk = r.get("content") or r.get("text") or ""
                            if chunk:
                                text = (text + "\n" + chunk).strip()
                elif isinstance(results_val, dict):
                    res_synthesis = results_val.get("synthesis", "")
                    if res_synthesis and res_synthesis != text:
                        text = (text + "\n" + res_synthesis).strip() if text else res_synthesis
                    for doc_list_key in ("local_documents", "documents", "items"):
                        doc_list = results_val.get(doc_list_key)
                        if isinstance(doc_list, list):
                            for r in doc_list[:8]:
                                if isinstance(r, dict):
                                    chunk = r.get("content_excerpt") or r.get("content") or r.get("text") or ""
                                    if chunk:
                                        text = (text + "\n" + chunk).strip()
                            break
                    if not text:
                        text = str(results_val)[:800]
                if isinstance(text, str) and len(text) > 2000:
                    text = text[:2000] + "..."
                if text and text.strip():
                    summary_parts.append(f"[{key}]: {text}")

        findings_text_len = sum(len(p) for p in summary_parts)

        if vector_store and findings_text_len < 500:
            logger.info(
                "Workflow extraction: findings text too thin (%d chars), searching vector store for '%s'",
                findings_text_len,
                research_query[:80],
            )
            try:
                vs_results = vector_store.search(query=research_query, top_k=15)
                for r in vs_results:
                    if not isinstance(r, dict):
                        continue
                    metadata = r.get("metadata", {})
                    # Skip AI-synthesized docs to avoid circular references
                    if metadata.get("type") in ("ai_synthesis_with_sources", "ai_synthesis", "research_finding"):
                        continue
                    chunk = r.get("content") or r.get("text") or ""
                    if chunk and len(chunk.strip()) > 30:
                        source = metadata.get("file_path") or metadata.get("source") or r.get("doc_id", "vs")
                        truncated = chunk[:1500] + "..." if len(chunk) > 1500 else chunk
                        summary_parts.append(f"[vector_store:{source}]: {truncated}")
                logger.info(
                    "Workflow extraction: after vector store search, %d summary parts, %d total chars",
                    len(summary_parts),
                    sum(len(p) for p in summary_parts),
                )
            except Exception as e:
                logger.warning("Workflow extraction: vector store search failed: %s", e)

        if not summary_parts:
            logger.info("Workflow extraction: no content available to analyze")
            return {"generic_workflow_steps": [], "conditional_steps": []}
        findings_summary = "\n\n".join(summary_parts[:20])
        logger.info(
            "Workflow extraction: %d summary parts, total chars: %d",
            len(summary_parts),
            len(findings_summary),
        )

        prompt = f"""You are analyzing research findings (policy documents, workflow procedures) for THIS specific research question. Extract only workflow and conditions that are RELEVANT to this question.

RESEARCH QUESTION: {research_query}

FINDINGS (policy/workflow content):
{findings_summary}

TASK:
A) EXTRACT THE GENERIC WORKFLOW that is directly relevant to the research question above. List the main steps (e.g. step 1: "Create incident record", step 2: "Complete Gate 1", ...). For each step set is_conditional: true only if that step has explicit decision branches (if/then/else, or "if X then path A, else path B") in the text.
B) For each step where is_conditional is true, add ONE entry to "conditional_steps" with:
   - condition_id: short id based on THAT step (e.g. "consent_waiver_vs_written")
   - step_number: the step index
   - description: the exact decision from the workflow (e.g. "Whether consent can be waived or written consent is required")
   - branches: the actual branch options from the document (e.g. ["waiver applicable", "written consent required"])
   - evidence_needed: what user/personal fact would resolve it (e.g. "study design, vulnerable population, data category")
   - search_queries: 2-3 diverse search phrases to find that evidence in user emails/chats/files. Each query must be meaningfully different from the others — not rephrasings. Each query across ALL conditional_steps must also be distinct.

IMPORTANT: conditional_steps must ONLY include conditions that appear in the extracted workflow for THIS research question. Do not add generic conditions (e.g. "vendor vs internal", "encrypted vs not") unless they actually appear in the findings for this question.

Return a single JSON object only, no other text. Example:
{{
  "generic_workflow_steps": [
    {{ "step": 1, "title": "Create project record", "is_conditional": false }},
    {{ "step": 2, "title": "Determine consent approach", "is_conditional": true }}
  ],
  "conditional_steps": [
    {{
      "condition_id": "consent_waiver_vs_written",
      "step_number": 2,
      "description": "Whether consent can be waived or written consent is required",
      "branches": ["waiver applicable", "written consent required"],
      "evidence_needed": "Study design, data category, vulnerable population",
      "search_queries": ["relevant term from workflow", "evidence search phrase", "another distinct angle"]
    }}
  ]
}}

If the workflow has no conditional steps, set "conditional_steps": [].
"""

        try:
            out = prompt_llm_and_parse_json(self.model, prompt)
            if isinstance(out, list):
                conditional = [c for c in out if isinstance(c, dict) and c.get("search_queries")]
                return {"generic_workflow_steps": [], "conditional_steps": conditional}
            if not isinstance(out, dict):
                logger.warning("Workflow extraction: LLM returned unexpected type %s", type(out).__name__)
                return {"generic_workflow_steps": [], "conditional_steps": []}
            steps = out.get("generic_workflow_steps") or []
            conditional_raw = out.get("conditional_steps") or []
            conditional = [c for c in conditional_raw if isinstance(c, dict) and c.get("search_queries")]
            if len(conditional_raw) != len(conditional):
                logger.warning(
                    "Workflow extraction: %d conditional_steps filtered out (missing search_queries)",
                    len(conditional_raw) - len(conditional),
                )
            logger.info(
                "Extracted generic workflow: %d steps, %d conditional steps (for this DR question)",
                len(steps),
                len(conditional),
            )
            return {"generic_workflow_steps": steps, "conditional_steps": conditional}
        except Exception as e:
            logger.warning("Generic workflow and condition extraction failed: %s", e)
            return {"generic_workflow_steps": [], "conditional_steps": []}

    def _extract_conditions_from_findings(
        self, findings: Dict[str, Any], research_query: str
    ) -> List[Dict[str, Any]]:
        """Return conditional steps from the extracted generic workflow (for backward compatibility)."""
        data = self._extract_generic_workflow_and_conditions(findings, research_query)
        return data.get("conditional_steps", [])

    def _generate_fallback_personal_evidence_actions(
        self, context: ResearchContext, tool_registry
    ) -> List[Action]:
        """When no conditional steps are extracted, use the LLM to generate 2-3
        diverse actions to search for personalized evidence relevant to the DR question."""
        available_tool_names = [t.__class__.__name__ for t in tool_registry.tools]
        has_local = any("LocalFileSearchTool" in n for n in available_tool_names)
        has_enterprise = any("EnterpriseAPITool" in n for n in available_tool_names)
        if not has_enterprise and not has_local:
            return []

        tool_guidelines = self._generate_tool_guidelines(tool_registry)

        existing_queries = []
        for plan in self.active_plans.values():
            for a in plan.actions:
                q = (a.parameters.get("query") or "").strip()
                if q:
                    existing_queries.append(q)

        prompt = f"""Generate 2-3 diverse search actions to find PERSONAL/USER-SPECIFIC evidence
relevant to this research question. These actions should search the user's own
emails, chats, files, and documents — NOT policy documents (those are already collected).

RESEARCH QUESTION: {context.original_question}

ALREADY-USED QUERIES (do NOT repeat or closely rephrase these):
{json.dumps(existing_queries[:10], indent=2)}

Available Tools: {available_tool_names}
Tool Guidelines:
{tool_guidelines}

Each action MUST explore a genuinely different angle of the question.
Vary the tool used, the query terms, and the type of evidence sought.

Return a JSON array of 2-3 actions:
[
  {{
    "type": "local_document_search" or "enterprise_api",
    "description": "What this action searches for and why",
    "parameters": {{"query": "specific search terms", ...}},
    "priority": 0.8-0.9,
    "expected_output": "What kind of user evidence this should find",
    "preferred_tool": "LocalFileSearchTool" or "EnterpriseAPITool"
  }}
]

Return valid JSON only, no other text.
"""

        try:
            action_configs = prompt_llm_and_parse_json(self.model, prompt)
            actions = []
            for i, config in enumerate(action_configs[:3]):
                try:
                    action_type_str = config.get("type", "local_document_search")
                    actions.append(
                        Action(
                            id=f"alternative_personal_evidence_{i}",
                            type=ActionType(action_type_str),
                            description=config.get("description", ""),
                            parameters=_normalize_action_parameters(config.get("parameters", {})),
                            priority=config.get("priority", 0.85),
                            expected_output=config.get("expected_output", ""),
                            preferred_tool=config.get("preferred_tool"),
                            stage="PERSONAL_EVIDENCE_COLLECTION",
                            created_from_research_step="alternative_personal_evidence",
                        )
                    )
                except Exception as e:
                    logger.warning("Error parsing alternative personal evidence action: %s", e)
                    continue
            logger.info("LLM generated %d diverse alternative personal evidence actions", len(actions))
            return actions
        except Exception as e:
            logger.warning("LLM alternative personal evidence generation failed: %s", e)
            return []

    def _generate_condition_resolution_actions(
        self, context: ResearchContext, tool_registry
    ) -> List[Action]:
        """Use the LLM to generate PERSONAL_EVIDENCE_COLLECTION actions that search for
        personalized evidence to resolve conditional steps in the workflow.

        The LLM receives the extracted conditions, available tools, and already-used
        queries, and freely decides what diverse actions to generate."""
        data = self._extract_generic_workflow_and_conditions(
            context.findings, context.original_question, vector_store=context.vector_store
        )
        conditional_steps = data.get("conditional_steps") or []
        if not conditional_steps:
            return self._generate_fallback_personal_evidence_actions(context, tool_registry)

        available_tool_names = [t.__class__.__name__ for t in tool_registry.tools]
        tool_guidelines = self._generate_tool_guidelines(tool_registry)

        existing_queries = []
        for plan in self.active_plans.values():
            for a in plan.actions:
                q = (a.parameters.get("query") or "").strip()
                if q:
                    existing_queries.append(q)

        conditions_summary = json.dumps(conditional_steps, indent=2, default=str)

        prompt = f"""Generate 3-6 diverse search actions to find PERSONAL/USER-SPECIFIC evidence
that can resolve the conditional steps in a workflow. The goal is personalization:
find facts about the user's specific situation so that each condition can be resolved
to a concrete branch.

RESEARCH QUESTION: {context.original_question}

CONDITIONAL STEPS TO RESOLVE:
{conditions_summary}

ALREADY-USED QUERIES (do NOT repeat or closely rephrase these):
{json.dumps(existing_queries[:15], indent=2)}

Available Tools: {available_tool_names}
Tool Guidelines:
{tool_guidelines}

REQUIREMENTS:
- Each action should target a DIFFERENT conditional step OR a different angle of the same step.
- Vary the tools used (mix local document search, enterprise API, etc.) when multiple are available.
- Search for USER-SPECIFIC evidence (emails, chats, user files, records) — NOT general policy documents.
- Queries should be concrete and specific — terms that would appear in real user communications.
- Do NOT just rephrase the condition description as a query. Think about what actual user data
  would contain the answer (e.g., specific correspondence, filed forms, meeting notes, approvals).

Return a JSON array of 3-6 actions:
[
  {{
    "type": "enterprise_api" or "local_document_search",
    "description": "What this searches for and which condition it resolves",
    "parameters": {{"query": "specific search terms", "service": "auto"}} or {{"query": "...", "top_k": 10}},
    "priority": 0.8-0.95,
    "expected_output": "What user-specific evidence this should find",
    "preferred_tool": "EnterpriseAPITool" or "LocalFileSearchTool"
  }}
]

Return valid JSON only, no other text.
"""

        try:
            action_configs = prompt_llm_and_parse_json(self.model, prompt)
            actions = []
            for i, config in enumerate(action_configs[:6]):
                try:
                    action_type_str = config.get("type", "local_document_search")
                    condition_id = config.get("condition_id", f"cond_{i}")
                    actions.append(
                        Action(
                            id=f"cond_evidence_{condition_id}_{i}",
                            type=ActionType(action_type_str),
                            description=config.get("description", ""),
                            parameters=_normalize_action_parameters(config.get("parameters", {})),
                            priority=config.get("priority", 0.9),
                            expected_output=config.get("expected_output", ""),
                            preferred_tool=config.get("preferred_tool"),
                            stage="PERSONAL_EVIDENCE_COLLECTION",
                            created_from_research_step="condition_resolution",
                        )
                    )
                except Exception as e:
                    logger.warning("Error parsing condition resolution action: %s", e)
                    continue
            logger.info(
                "LLM generated %d personalized-evidence actions to resolve %d conditional steps",
                len(actions),
                len(conditional_steps),
            )
            return actions
        except Exception as e:
            logger.warning("LLM condition resolution action generation failed: %s", e)
            return self._generate_fallback_personal_evidence_actions(context, tool_registry)

    def _generate_adaptive_actions(
        self,
        action_plan: ActionPlan,
        new_findings: Dict[str, Any],
        context: ResearchContext,
        tool_registry,
    ) -> List[Action]:
        """Generate new actions based on findings from completed actions.

        Diversity: the prompt now includes all previously used queries so the LLM
        can propose genuinely novel search angles instead of rephrasing old queries.
        """

        completed_actions = [a for a in action_plan.actions if a.status == ActionStatus.COMPLETED]

        internal_findings = self._analyze_source_composition(new_findings)

        existing_queries = []
        existing_descriptions = []
        for a in action_plan.actions:
            q = (a.parameters.get("query") or "").strip()
            if q:
                existing_queries.append(q)
            existing_descriptions.append(f"[{a.stage or '?'}] {a.description[:120]}")

        available_tool_names = [tool.__class__.__name__ for tool in tool_registry.tools]

        prompt = f"""
Based on the research progress, suggest 0-2 additional actions ONLY if genuine gaps exist.

Original Research Query: {action_plan.research_query}
Completed Actions: {len(completed_actions)}

Current Action Plan Status:
{json.dumps(action_plan.get_stats(), indent=2)}

SOURCE ANALYSIS:
- Policy Documents Found: {internal_findings['has_internal']}
- Personal Evidence Found: {internal_findings['has_enterprise']}

ALREADY-USED SEARCH QUERIES (DO NOT repeat or rephrase these):
{json.dumps(existing_queries[:15], indent=2)}

EXISTING ACTION DESCRIPTIONS:
{chr(10).join(existing_descriptions[:15])}

**CRITICAL DIVERSITY RULES:**
1. Do NOT suggest queries that overlap >50% in keywords with any already-used query above.
2. Each new action MUST explore a DIFFERENT evidence dimension (e.g. people, dates,
   specific systems, regulatory bodies, incident IDs) — not just rephrase the topic.
3. If REQUIREMENT_COLLECTION already has 7+ actions, do NOT add more for that stage.
4. If PERSONAL_EVIDENCE_COLLECTION already has 7+ actions, do NOT add more for that stage.
5. If both policy documents AND personal evidence are found, return [].

Available Tools: {available_tool_names}

Return a JSON array (0-2 actions). Return [] if no genuine gaps exist:
[
  {{
    "type": "local_document_search",
    "description": "A genuinely different search angle not covered above",
    "parameters": {{"query": "novel keywords not used before", "top_k": 15}},
    "priority": 0.85,
    "expected_output": "New information not already retrieved",
    "preferred_tool": "LocalFileSearchTool",
    "stage": "PERSONAL_EVIDENCE_COLLECTION"
  }}
]

Create valid JSON only, no other text.
"""

        try:
            action_configs = prompt_llm_and_parse_json(self.model, prompt)

            new_actions = []
            for i, config in enumerate(action_configs):
                action_type = config.get("type", "web_search")
                desc_snippet = config.get("description", "")[:15].replace(" ", "_").lower()
                action_id = (
                    f"adaptive_{action_plan.current_iteration}_{action_type}_{i}"
                    if not desc_snippet
                    else f"adaptive_{action_plan.current_iteration}_{action_type}_{desc_snippet}_{i}"
                )
                try:
                    action = Action(
                        id=action_id,
                        type=ActionType(config.get("type", "web_search")),
                        description=config.get("description", ""),
                        parameters=_normalize_action_parameters(config.get("parameters", {})),
                        priority=config.get("priority", 0.5),
                        expected_output=config.get("expected_output", ""),
                        dependencies=config.get("dependencies", []),
                        preferred_tool=config.get("preferred_tool"),
                        stage=config.get("stage"),
                    )
                    new_actions.append(action)
                except Exception as e:
                    logger.warning(f"Error parsing adaptive action config. Skipping action: {e}")
                    continue

            return new_actions

        except Exception as e:
            logger.error(f"Error generating adaptive actions: {e}")
            return []

    def _deduplicate_actions(self, new_actions: List[Action], existing_actions: List[Action]) -> List[Action]:
        """Remove duplicate or highly similar actions"""
        unique_actions = []

        for new_action in new_actions:
            is_duplicate = False

            for existing in existing_actions:
                if self._is_similar_action(new_action, existing):
                    logger.debug(f"Skipping duplicate action: {new_action.description}")
                    is_duplicate = True
                    break

            if not is_duplicate:
                for unique in unique_actions:
                    if self._is_similar_action(new_action, unique):
                        logger.debug(f"Skipping duplicate action: {new_action.description}")
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_actions.append(new_action)

        logger.info(f"Deduplication: {len(new_actions)} actions -> {len(unique_actions)} unique actions")
        return unique_actions

    def _is_similar_action(self, action1: Action, action2: Action) -> bool:
        """Check if two actions are similar enough to be considered duplicates"""
        _SEARCH_TYPES = {
            ActionType.WEB_SEARCH,
            ActionType.ENTERPRISE_API,
            ActionType.LOCAL_DOCUMENT_SEARCH,
            ActionType.LOCAL_FILE_ANALYSIS,
        }

        both_search = action1.type in _SEARCH_TYPES and action2.type in _SEARCH_TYPES
        if not both_search and action1.type != action2.type:
            return False

        desc1 = action1.description.lower().strip()
        desc2 = action2.description.lower().strip()
        if desc1 == desc2:
            return True

        query1 = (action1.parameters.get("query") or "").lower().strip()
        query2 = (action2.parameters.get("query") or "").lower().strip()

        if query1 and query2:
            words1 = set(query1.split())
            words2 = set(query2.split())
            if words1 and words2:
                overlap = len(words1 & words2)
                similarity = overlap / max(len(words1), len(words2))
                if similarity > 0.55:
                    return True

        if both_search:
            d_words1 = set(re.findall(r"\b\w{3,}\b", desc1))
            d_words2 = set(re.findall(r"\b\w{3,}\b", desc2))
            filler = {"search", "for", "the", "and", "documents", "local",
                       "enterprise", "files", "with", "related", "relevant",
                       "procedures", "procedure", "specific", "regarding",
                       "involving", "policies", "policy", "personalized",
                       "evidence", "resolve", "conditional", "step", "workflow"}
            d_words1 -= filler
            d_words2 -= filler
            if d_words1 and d_words2:
                overlap = len(d_words1 & d_words2)
                similarity = overlap / max(len(d_words1), len(d_words2))
                if similarity > 0.6:
                    return True

        if action1.type == ActionType.URL_FETCH:
            urls1 = set(action1.parameters.get("urls", []))
            urls2 = set(action2.parameters.get("urls", []))
            if urls1 and urls2 and urls1 == urls2:
                return True

        return False

    def _analyze_source_composition(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze findings to understand source type composition"""

        analysis = {
            "has_internal": False,
            "has_external": False,
            "has_enterprise": False,
            "internal_insights": [],
            "external_insights": [],
            "enterprise_insights": [],
            "gaps": [],
        }

        for key, finding in findings.items():
            if not isinstance(finding, dict):
                continue

            if finding.get("success"):
                if finding.get("tool") == "enhanced_enterprise_api":
                    analysis["has_enterprise"] = True
                    if finding.get("content_stored_in_vector", 0) > 0:
                        analysis["enterprise_insights"].append(
                            f"Found {finding.get('content_stored_in_vector', 0)} enterprise sources"
                        )

                if finding.get("processed_files"):
                    analysis["has_internal"] = True
                    analysis["internal_insights"].append(
                        f"Processed {len(finding.get('processed_files', []))} internal files"
                    )

                if finding.get("tool") == "internet_search":
                    analysis["has_external"] = True
                    if finding.get("urls_processed", 0) > 0:
                        analysis["external_insights"].append(f"Found {finding.get('urls_processed', 0)} web sources")

                if finding.get("url") and finding.get("success"):
                    analysis["has_external"] = True
                    analysis["external_insights"].append(f"Fetched content from {finding.get('url')}")

        if not analysis["has_internal"] and not analysis["has_enterprise"]:
            analysis["gaps"].append("Missing internal/enterprise sources")
        if not analysis["has_external"]:
            analysis["gaps"].append("Missing external validation sources")
        if analysis["has_internal"] and not analysis["has_external"]:
            analysis["gaps"].append("Need external context to validate internal findings")
        if analysis["has_external"] and not analysis["has_internal"]:
            analysis["gaps"].append("Need internal perspective to complement external research")

        return analysis

    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get the status of an action plan"""
        action_plan = self.active_plans.get(plan_id)
        if not action_plan:
            return {"error": "Plan not found"}

        return {
            "plan_id": plan_id,
            "query": action_plan.research_query,
            "stats": action_plan.get_stats(),
            "is_complete": action_plan.is_complete(),
            "actions": [
                {
                    "id": a.id,
                    "type": a.type.value,
                    "description": a.description,
                    "status": a.status.value,
                    "priority": a.priority,
                }
                for a in action_plan.actions
            ],
        }

    def cleanup_completed_plans(self):
        """Remove completed action plans to save memory"""
        completed_plans = [plan_id for plan_id, plan in self.active_plans.items() if plan.is_complete()]

        for plan_id in completed_plans:
            del self.active_plans[plan_id]

        return len(completed_plans)

    def _detect_action_dependencies(self, actions: List[Action], task_context: str = "") -> Dict[str, List[str]]:
        """Use LLM to detect dependencies between actions"""

        if len(actions) <= 1:
            return {}

        actions_info = []
        for action in actions:
            actions_info.append(
                {
                    "id": action.id,
                    "type": action.type.value,
                    "description": action.description,
                    "expected_output": action.expected_output,
                }
            )

        prompt = f"""
    Task Context: {task_context}

    Analyze these actions and determine which ones depend on others:

    {json.dumps(actions_info, indent=2)}

    Return a JSON object where each action ID maps to a list of action IDs it depends on.
    Only create dependencies where an action truly needs the output/result from another action.

    Rules:
    - No circular dependencies
    - No self-dependencies  
    - Only use action IDs from the list above
    - Prefer parallel execution when possible

    Example format:
    {{
    "task_action_0": [],
    "task_action_1": ["task_action_0"],
    "task_action_2": ["task_action_0", "task_action_1"]
    }}

    Return only the JSON:
    """

        try:
            dependencies = prompt_llm_and_parse_json(self.model, prompt)

            valid_action_ids = {action.id for action in actions}
            validated_deps = {}

            for action_id, deps in dependencies.items():
                if action_id in valid_action_ids:
                    valid_deps = [dep for dep in deps if dep in valid_action_ids and dep != action_id]
                    validated_deps[action_id] = valid_deps

            return validated_deps

        except Exception as e:
            logger.error(f"Error detecting dependencies: {e}")
            return {}


def prompt_llm_and_parse_json(model: str, prompt: str, retries: int = 3):
    """
    Call prompt_llm and robustly parse a JSON response, handling code blocks and cleaning.
    Retries on JSONDecodeError or ValueError.
    """

    @retry(
        stop=stop_after_attempt(retries),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
    )
    def _call():
        response = prompt_llm(model=model, prompt=prompt)
        clean_response = re.sub(r"^```json\s*|\s*```$", "", response.strip())
        m = re.search(r"```json\s*(.*?)\s*```", clean_response, re.DOTALL)
        clean_response = m.group(1) if m else clean_response
        return json.loads(clean_response)

    return _call()
