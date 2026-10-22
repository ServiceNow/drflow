from typing import List, Dict, Any
import re
import tiktoken
import tqdm
import pandas as pd

from drflow.metrics.base import DrBenchMetric
from drflow.agents import utils
from drflow import task_loader


class CitationFactuality(DrBenchMetric):
    def __init__(self, model: str):
        """
        Initialize the ClaimFactuality metric.

        Args:
            model: The name of the model to use for scoring
            max_context_tokens: Size of text chunks for semantic retrieval system
            embedding_model: OpenAI embedding model to use
        """
        super().__init__(name="factuality", model=model)
        self.model = model

        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))

    def _chunk_text(self, text: str, max_tokens: int = 3000) -> List[str]:
        """
        Break text into chunks that don't exceed max_tokens.

        Args:
            text: The text to chunk
            max_tokens: Maximum tokens per chunk

        Returns:
            List of text chunks
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            if self._count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def _extract_links(self, text: str) -> List[str]:
        """
        Extracts all URLs from a given text string.

        Args:
            text (str): The input text containing URLs.

        Returns:
            List[str]: A list of extracted URLs.
        """
        url_pattern = re.compile(r'(https?://[^\s<>"]+|www\.[^\s<>"]+)', re.IGNORECASE)
        return url_pattern.findall(text)

    def compute(self, report_dict: Dict[str, Any], task_data=None, eval_data=None) -> dict:
        """
        Compute factuality scores using RAG-based evaluation.

        Args:
            report_dict: Dictionary containing 'report_text' and 'report_insights'
            task_data: Task-specific data
            eval_data: Evaluation data

        Returns:
            Dict: Standardized result with claim-based factuality scores
        """
        insight_list = report_dict.get("report_insights", [])

        insights_with_citations = [
            insight for insight in insight_list if insight["citations"] and len(insight["citations"]) > 0
        ]
        insights_without_citations = [
            insight for insight in insight_list if not insight["citations"] or len(insight["citations"]) == 0
        ]

        env_files = task_loader.from_task_config_to_env_files(task_data)

        if len(insight_list) == 0:
            factual_insights = []
            total_count = 0
            evaluation = {
                "score": 0.0,
                "summary": "no insights found",
                "metric_result": {
                    "factual_claims": [],
                    "unfactual_claims": [],
                    "factuality_percentage": 0.0,
                    "total_claims": 0,
                },
            }
        else:
            factuality_list = []

            for insight_citation in tqdm.tqdm(
                insights_with_citations, desc="Checking insight factuality...", leave=False
            ):

                insight = insight_citation["claim"]
                citations = insight_citation["citations"]

                factuality_verdict = utils.get_factuality_verdict_multi(
                    insight, citations, file_list=env_files, model=self.model
                )

                factuality_verdict["insight"] = insight
                factuality_verdict["citations"] = citations
                factuality_list.append(factuality_verdict)

            for insight_no_citation in insights_without_citations:
                factuality_verdict = {
                    "insight": insight_no_citation["claim"],
                    "citations": [],
                    "is_factual": False,
                    "explanation": "No citation provided to verify the claim",
                    "source_details": [],
                }
                factuality_list.append(factuality_verdict)

            df = pd.DataFrame(factuality_list)
            factual_insights = df[df["is_factual"]]["insight"].tolist()
            unfactual_insights = df[~df["is_factual"]]["insight"].tolist()
            factuality_percentage = len(factual_insights) / len(df) * 100
            total_count = len(df)

            metric_result = {
                "factual_claims": factual_insights,
                "unfactual_claims": unfactual_insights,
                "factuality_percentage": factuality_percentage,
                "total_claims": total_count,
                "detailed_factuality": factuality_list,
            }

            evaluation = {}
            evaluation["metric_result"] = metric_result
            evaluation["score"] = factuality_percentage / 100.0

            evaluation["summary"] = ""
            evaluation[
                "summary"
            ] += f"**Factuality Score:** {evaluation['score']:.4f} which is {len(factual_insights)}/{total_count} claims\n\n"
            evaluation["summary"] += f"--------------------------------\n\n"

            unfactual_insights = [
                factuality_dict for factuality_dict in factuality_list if not factuality_dict["is_factual"]
            ]
            factual_insights = [factuality_dict for factuality_dict in factuality_list if factuality_dict["is_factual"]]

            evaluation["summary"] += f"\n\n**Factual Claims:**\n\n--------------------------------\n\n"
            for factuality_dict in factual_insights:
                evaluation["summary"] += f"**Claim:** {factuality_dict['insight']}\n\n"
                citations = factuality_dict.get("citations", [])
                citations_str = "; ".join(citations) if citations else "None"
                evaluation["summary"] += f"**Citations:** {citations_str}\n\n"
                evaluation["summary"] += f"**Factuality:** {factuality_dict['is_factual']}\n\n"
                evaluation["summary"] += f"**Explanation:** {factuality_dict['explanation']}\n\n"

                if "source_details" in factuality_dict and factuality_dict["source_details"]:
                    evaluation["summary"] += f"**Source Details:**\n"
                    for detail in factuality_dict["source_details"]:
                        status = "✓" if detail["content_available"] else "✗"
                        evaluation[
                            "summary"
                        ] += f"  {status} {detail['citation']} (content length: {detail['content_length']})\n"
                    evaluation["summary"] += "\n"

                evaluation["summary"] += f"--------------------------------\n\n"

            evaluation["summary"] += f"\n\n**Unfactual Claims:**\n\n--------------------------------\n\n"
            for factuality_dict in unfactual_insights:
                evaluation["summary"] += f"**Claim:** {factuality_dict['insight']}\n\n"
                citations = factuality_dict.get("citations", [])
                citations_str = "; ".join(citations) if citations else "None"
                evaluation["summary"] += f"**Citations:** {citations_str}\n\n"
                evaluation["summary"] += f"**Factuality:** {factuality_dict['is_factual']}\n\n"
                evaluation["summary"] += f"**Explanation:** {factuality_dict['explanation']}\n\n"

                if "source_details" in factuality_dict and factuality_dict["source_details"]:
                    evaluation["summary"] += f"**Source Details:**\n"
                    for detail in factuality_dict["source_details"]:
                        status = "✓" if detail["content_available"] else "✗"
                        evaluation[
                            "summary"
                        ] += f"  {status} {detail['citation']} (content length: {detail['content_length']})\n"
                    evaluation["summary"] += "\n"

                evaluation["summary"] += f"--------------------------------\n\n"

        evaluation["summary"] += f"--------------------------------\n\n"
        evaluation["summary"] += (
            "Score: "
            + str(evaluation["score"])
            + " which is "
            + str(len(factual_insights))
            + "/"
            + str(total_count)
            + " claims"
        )
        return evaluation
