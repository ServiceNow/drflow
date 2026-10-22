import datetime
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from drflow.agents.utils import prompt_llm
from drflow.drflow_enterprise_space import DrFlowEnterpriseSearchSpace


@dataclass
class ResearchContext:
    """Accumulates research findings and context throughout the process with bounded memory"""

    original_question: str
    plan: Optional[Dict] = None
    findings: Dict[str, Any] = field(default_factory=dict)
    files_created: List[str] = field(default_factory=list)
    apis_discovered: List[Dict] = field(default_factory=list)

    findings_summary: Dict[str, str] = field(default_factory=dict)
    findings_archive: List[str] = field(default_factory=list)
    max_findings_size: int = 15
    vector_store: Optional[Any] = None

    def add_finding(self, key: str, value: Any, category: str = "general"):
        """Add a finding with automatic archiving of old findings and proper doc_id assignment"""
        if len(self.findings) >= self.max_findings_size and self.vector_store:
            keys_to_archive = list(self.findings.keys())[:5]

            archive_content = {
                "archived_findings": {k: self.findings[k] for k in keys_to_archive},
                "timestamp": datetime.datetime.now().isoformat(),
                "research_question": self.original_question,
            }

            doc_id = self.vector_store.store_document(
                content=json.dumps(archive_content, indent=2),
                metadata={"type": "archived_findings", "keys": keys_to_archive, "category": "research_context_archive"},
            )

            if doc_id:
                self.findings_archive.append(doc_id)
                for k in keys_to_archive:
                    del self.findings[k]

        if isinstance(value, dict):
            if "doc_id" not in value:
                value["doc_id"] = key

            results_list = value.get("results")
            if isinstance(results_list, list):
                for i, item in enumerate(results_list):
                    if isinstance(item, dict) and "doc_id" not in item and "id" not in item:
                        item["doc_id"] = f"{key}_result_{i}"
                    elif isinstance(item, dict) and "doc_id" not in item and item.get("id"):
                        item["doc_id"] = item["id"]
            
            if "metadata" not in value:
                value["metadata"] = {}
            
            if "category" not in value["metadata"]:
                value["metadata"]["category"] = category
            
            if "timestamp" not in value["metadata"]:
                value["metadata"]["timestamp"] = datetime.datetime.now().isoformat()

        self.findings[key] = value

        if isinstance(value, dict) and value.get("summary"):
            self.findings_summary[category] = value.get("summary", str(value)[:200])

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a condensed summary suitable for LLM context"""
        return {
            "question": self.original_question,
            "findings_count": len(self.findings),
            "archived_count": len(self.findings_archive),
            "categories": list(self.findings_summary.keys()),
            "recent_findings": {k: self._summarize_finding(v) for k, v in list(self.findings.items())[-5:]},
            "category_summaries": self.findings_summary,
        }

    def _summarize_finding(self, finding: Any) -> str:
        """Create a brief summary of a finding"""
        if isinstance(finding, dict):
            if "summary" in finding:
                return finding["summary"]
            elif "results" in finding:
                return str(finding["results"])[:200] + "..."
        return str(finding)[:200] + "..."


class Tool(ABC):
    """Base interface for all research tools with standardized output"""

    @property
    @abstractmethod
    def purpose(self) -> str:
        """Return a brief description of what this tool is used for"""
        pass

    @abstractmethod
    def execute(self, query: str, context: ResearchContext) -> Dict[str, Any]:
        """Execute the tool and return standardized output"""
        pass

    def load_extracted_content(self, result: Dict[str, Any]) -> str:
        """
        Lazy load extracted content from file path
        
        Args:
            result: Tool result containing extracted_path
            
        Returns:
            The extracted content as string, or empty string if not available
        """
        extracted_path = result.get("extracted_path")
        if not extracted_path:
            return ""
            
        try:
            with open(extracted_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    def _standardize_output(
        self,
        raw_output: Dict[str, Any],
        tool_name: str,
        query: str,
        force_success: Optional[bool] = None,
        force_data_retrieved: Optional[bool] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Standardize tool output to ensure consistent format with proper citation metadata

        Args:
            raw_output: The original output from the tool
            tool_name: Name of the tool (e.g., "internet_search")
            query: Original query string
            force_success: Override success detection
            force_data_retrieved: Override data_retrieved detection
            doc_id: Optional document ID for citation tracking

        Returns:
            Standardized output dictionary
        """
        standardized = raw_output.copy()

        standardized["tool"] = tool_name
        standardized["query"] = query

        if force_success is not None:
            success = force_success
        elif "success" in raw_output:
            success = raw_output["success"]
        elif "error" in raw_output and raw_output["error"]:
            success = False
        else:
            success = self._detect_success(raw_output)

        standardized["success"] = success

        if force_data_retrieved is not None:
            data_retrieved = force_data_retrieved
        elif "data_retrieved" in raw_output:
            data_retrieved = raw_output["data_retrieved"]
        else:
            data_retrieved = self._detect_data_retrieved(raw_output)

        standardized["data_retrieved"] = data_retrieved

        if not success and "error" not in standardized:
            standardized["error"] = "Operation failed"

        standardized["summary"] = self._generate_summary(standardized)

        if "metadata" not in standardized:
            standardized["metadata"] = {}
        
        standardized["metadata"]["tool_used"] = tool_name
        standardized["metadata"]["query_context"] = query[:200] if query else ""
        
        if "timestamp" not in standardized["metadata"]:
            import datetime
            standardized["metadata"]["timestamp"] = datetime.datetime.now().isoformat()
        
        if doc_id:
            standardized["doc_id"] = doc_id
        elif "doc_id" not in standardized:
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8] if query else "unknown"
            standardized["doc_id"] = f"{tool_name}_{query_hash}"

        return standardized

    def _detect_success(self, output: Dict[str, Any]) -> bool:
        """Auto-detect if operation was successful"""

        if output.get("error"):
            return False

        success_indicators = [
            output.get("results") and len(output["results"]) > 0,
            output.get("apis_found") and len(output["apis_found"]) > 0,
            output.get("analyzed_files") and len(output["analyzed_files"]) > 0,
            output.get("documents_found", 0) > 0,
            output.get("synthesis") and len(str(output["synthesis"])) > 10,
            output.get("content") and len(str(output["content"])) > 10,
            output.get("processed_files") and len(output["processed_files"]) > 0,
            output.get("urls_processed", 0) > 0,
            output.get("servers_discovered", 0) > 0,
            output.get("file_path") and os.path.exists(str(output["file_path"])),
            output.get("stored_in_vector", False),
            output.get("status_code") in [200, 201, 202] if "status_code" in output else None,
        ]

        return any(indicator for indicator in success_indicators if indicator is not None)

    def _detect_data_retrieved(self, output: Dict[str, Any]) -> bool:
        """Auto-detect if meaningful data was retrieved"""

        data_indicators = [
            output.get("results") and len(output["results"]) > 0,
            output.get("apis_found") and len(output["apis_found"]) > 0,
            output.get("analyzed_files") and len(output["analyzed_files"]) > 0,
            output.get("documents_found", 0) > 0,
            output.get("items_found", 0) > 0,
            output.get("processed_files") and len(output["processed_files"]) > 0,
            output.get("synthesis") and len(str(output["synthesis"])) > 50,
            output.get("content") and len(str(output["content"])) > 50,
            output.get("text_content") and len(str(output["text_content"])) > 50,
            output.get("extracted_path") and os.path.exists(str(output["extracted_path"])) and output.get("content_length", 0) > 50,
            output.get("stored_in_vector", False),
            output.get("content_length", 0) > 100,
            output.get("documents_stored", 0) > 0,
            output.get("data") and len(str(output["data"])) > 10,
            output.get("response_size", 0) > 100,
        ]

        return any(indicator for indicator in data_indicators if indicator is not None)

    def _generate_summary(self, output: Dict[str, Any]) -> str:
        """Generate a brief summary of the tool execution"""

        tool_name = output.get("tool", "unknown")
        success = output.get("success", False)
        data_retrieved = output.get("data_retrieved", False)

        if not success:
            error = output.get("error", "Unknown error")
            return f"{tool_name}: Failed - {error}"

        if data_retrieved:
            data_count = (
                len(output.get("results", []))
                or len(output.get("processed_files", []))
                or output.get("documents_found", 0)
                or output.get("items_found", 0)
                or output.get("urls_processed", 0)
                or output.get("documents_stored", 0)
                or 1
            )
            return f"{tool_name}: Success - Retrieved {data_count} items"
        else:
            return f"{tool_name}: Success - Operation completed"

    def create_success_output(
        self, tool_name: str, query: str, results: Any = None, data_retrieved: bool = True, 
        doc_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Helper method to create standardized success output with citation support"""

        output = {"tool": tool_name, "query": query, "success": True, "data_retrieved": data_retrieved, **kwargs}

        if results is not None:
            output["results"] = results
        
        if metadata is not None:
            output["metadata"] = metadata

        return self._standardize_output(output, tool_name, query, doc_id=doc_id)

    def create_error_output(self, tool_name: str, query: str, error: str, doc_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Helper method to create standardized error output"""

        output = {
            "tool": tool_name,
            "query": query,
            "success": False,
            "data_retrieved": False,
            "error": error,
            **kwargs,
        }

        return self._standardize_output(output, tool_name, query, doc_id=doc_id)


class ToolRegistry:
    """Manages available research tools"""

    def __init__(self):
        self.tools: List[Tool] = []

    def register_tool(self, tool: Tool):
        self.tools.append(tool)

    def select_tools(self, query: str = None, context: ResearchContext = None) -> List[Tool]:
        """Return all available tools - let action planner handle intelligent selection"""
        return self.tools


class FileManager:
    """Handles local file operations"""

    def __init__(self, workspace_dir: str = "./research_workspace"):
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)

    def save_file(self, filename: str, content: str) -> str:
        filepath = os.path.join(self.workspace_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

    def extract_text(self, filepath: str) -> str:
        """Extract text from file (starting with plain text support)"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"


class QueryPlanner:
    """Plans research sections and generates sub-queries"""

    def __init__(self, model: str):
        self.model = model

    def create_research_plan(
        self,
        question: str,
        tool_registry: ToolRegistry = None,
        env: Optional[DrFlowEnterpriseSearchSpace] = None,
    ) -> Dict:
        """Generate structured research plan with sections and sub-queries"""

        available_tools = [f"{tool.__class__.__name__}" for tool in tool_registry.tools]

        enterprise_services = []
        if env is not None:
            enterprise_services = env.get_available_apps()
            enterprise_services = [
                f"{service['name']}: {service['description']}" for service in enterprise_services.values()
            ]
        tools_section = f"""
Available Research Tools:
{chr(10).join(f"- {tool}" for tool in available_tools)}

Available Enterprise Services:
{chr(10).join(f"- {service}" for service in enterprise_services)}
"""

        planning_prompt = f"""
Design a comprehensive research plan to answer: "{question}"

{tools_section}

As a senior enterprise workflow architect, design a research plan that will:
1. Create a COMPREHENSIVE research plan covering both generic workflow and user evidence.
2. Drive action planning so it properly covers ALL items in the generic workflow (policy steps, conditional branches, prerequisites).
3. Drive action planning to collect necessary user/personal evidence to resolve conditional branches where possible.
4. Produce both a generic workflow and a personalized workflow, each with detailed task descriptions under every to-do; conditional branches resolved by user evidence must be presented as resolved in the personalized workflow.

Your plan MUST follow these 4 stages:

### STAGE 1: REQUIREMENT_COLLECTION (CRITICAL - EXTRACT FULL WORKFLOWS)
**Goal**: Find INTERNAL policy documents that describe STEP-BY-STEP procedures

**INTERNAL SOURCES (both are high priority):**
- LOCAL: Use LocalFileSearchTool for uploaded documents, policies, playbooks
- ENTERPRISE: Use EnterpriseAPITool for enterprise document repositories

**Search Strategy:**
- Search LOCAL for playbooks, manuals, guides that contain numbered steps
- Search ENTERPRISE for incident response procedures with detailed workflows
- Search for compliance procedures with decision points
- Look for documents with conditional logic (if X then Y, else Z)
- PRIORITIZE INTERNAL (local + enterprise) over external web searches
- AVOID external web search unless internal documents are insufficient

**What to extract**:
- Every step in the workflow (aim for 8-15 steps)
- Prerequisites between steps
- Conditional branches at each step
- Required forms, systems, approvals at each step

### STAGE 2: GENERAL_FLOWCHART_DESIGN
- Map ALL steps from collected policies into a comprehensive workflow
- Preserve ALL conditional branches - do not resolve them
- Include prerequisites and dependencies between steps
- The workflow should be generic (applies to ANY user)

### STAGE 3: PERSONAL_EVIDENCE_COLLECTION (CRITICAL - RESOLVE CONDITIONS)
**Goal**: Find user-specific facts that can resolve conditional branches
- Search user's EMAILS for incident details, vendor communications, timelines
- Search user's CHAT messages for discussions about the situation
- Search user's LOCAL files for relevant documentation
- Search enterprise files for user-specific context

**What to extract**:
- Specific entities (vendor names, system names, people involved)
- Dates and timeframes
- Data types affected
- Counts and measurements
- Constraints or special circumstances (e.g., "survey window", "no downtime")

### STAGE 4: FLOWCHART_PERSONALIZATION
- For each conditional step, check if personal evidence resolves it
- Personalize task descriptions with user's specific facts
- Keep conditions unresolved when no evidence determines the path
- Cite both policy sources AND personal evidence sources

Generate a JSON object with EXACTLY 2 research investigation areas (1 per stage).

**STRICT LIMITS - FOCUS ON FLOWCHART BUILDING**: 
- REQUIREMENT_COLLECTION: EXACTLY 1 investigation area (comprehensive search)
- PERSONAL_EVIDENCE_COLLECTION: EXACTLY 1 investigation area (comprehensive search)
- This minimal research allows MORE TIME for building detailed workflows

{{
  "research_investigation_areas": [
    {{
      "area_id": 1,
      "stage": "REQUIREMENT_COLLECTION",
      "research_focus": "INTERNAL (local + enterprise) policies, playbooks, and procedures related to [topic from question]",
      "information_needs": ["Step-by-step procedures", "Decision points", "Required forms/approvals"],
      "knowledge_sources": ["internal"],
      "research_approach": "requirement_analysis",
      "search_terms": ["[main topic] procedure", "[main topic] playbook", "[main topic] policy"],
      "business_rationale": "Find INTERNAL documents (local files + enterprise docs) with workflow steps (8-15 steps)",
      "importance_level": "critical"
    }},
    {{
      "area_id": 2,
      "stage": "PERSONAL_EVIDENCE_COLLECTION",
      "research_focus": "User's emails, chats, and files about the specific situation",
      "information_needs": ["Incident details", "Vendor/entity names", "Dates", "Constraints"],
      "knowledge_sources": ["internal"],
      "research_approach": "evidence_gathering",
      "search_terms": ["[entities from question]", "[dates from question]"],
      "business_rationale": "Personal evidence to personalize and resolve workflow conditions",
      "importance_level": "critical"
    }}
  ],
  "workflow_strategy": {{
    "requirement_gathering_approach": "Single comprehensive search for procedure documents",
    "personal_evidence_strategy": "Single comprehensive search for user context",
    "flowchart_focus": "Spend majority of effort on building detailed 8-15 step workflows",
    "generic_workflow_coverage": ["Extract every policy step (8-15)", "Map all conditional branches", "Capture prerequisites between steps", "Required forms/systems/approvals"],
    "personal_evidence_needs": ["Entities (vendors, systems, people)", "Dates and timelines", "Data types and encryption/access facts", "Constraints and approver availability"]
  }}
}}

Keep it simple: 1 policy search + 1 personal evidence search = focus on workflow building.
"""

        response = prompt_llm(model=self.model, prompt=planning_prompt)
        try:
            clean_response = re.sub(r"^```json\s*|\s*```$", "", response.strip())
            m = re.search(r"```json\s*(.*?)\s*```", clean_response, re.DOTALL)
            clean_response = m.group(1) if m else clean_response
            return json.loads(clean_response)
        except:
            return {
                "research_investigation_areas": [
                    {
                        "area_id": 1,
                        "stage": "REQUIREMENT_COLLECTION",
                        "research_focus": f"INTERNAL (local + enterprise) policies and procedures for: {question}",
                        "information_needs": ["Step-by-step procedures", "Playbooks", "Decision points"],
                        "knowledge_sources": ["internal"],
                        "research_approach": "requirement_analysis",
                        "business_rationale": "Find workflow steps from INTERNAL documents (local files + enterprise docs)",
                        "importance_level": "critical"
                    },
                    {
                        "area_id": 2,
                        "stage": "PERSONAL_EVIDENCE_COLLECTION",
                        "research_focus": f"User's context: {question}",
                        "information_needs": ["Emails", "Chats", "Files", "Specific facts"],
                        "knowledge_sources": ["internal"],
                        "research_approach": "evidence_gathering",
                        "business_rationale": "Personal evidence to personalize workflow",
                        "importance_level": "critical"
                    }
                ],
                "workflow_strategy": {
                    "requirement_gathering_approach": "Single comprehensive policy search",
                    "personal_evidence_strategy": "Single comprehensive personal evidence search",
                    "flowchart_focus": "Focus on building detailed workflows",
                    "generic_workflow_coverage": ["Extract every policy step (8-15)", "Map all conditional branches", "Capture prerequisites"],
                    "personal_evidence_needs": ["Entities", "Dates", "Data types and access facts", "Constraints and approvers"]
                }
            }
