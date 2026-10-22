import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pymupdf as fitz
import requests
from bs4 import BeautifulSoup
from docx import Document
from openai import OpenAI

from drflow import config

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger("source_reader")



OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5", "gpt-5.2", "gpt-5.5", "gpt-5-mini"]

logger = logging.getLogger(__name__)


def _get_openrouter_base_url() -> str:
    base_url = (config.OPENROUTER_API_URL or "https://openrouter.ai/api/v1").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def prompt_llm(prompt, model="together_ai/meta-llama/Meta-Llama-3-8B-Instruct-Lite", **kwargs):
    show_cost = kwargs.pop("show_cost", False)
    if model in OPENAI_MODELS:
        routed_model = model
        if config.OPENAI_API_KEY:
            client = OpenAI(api_key=config.OPENAI_API_KEY)
        elif config.OPENROUTER_API_KEY:
            client = OpenAI(
                base_url=_get_openrouter_base_url(),
                api_key=config.OPENROUTER_API_KEY,
            )
            routed_model = f"openai/{model}"
        else:
            raise ValueError(
                "Neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set. Please set one in your .env file or environment."
            )
        if "response_format" in kwargs.keys():
            response = client.responses.parse(
                model=routed_model,
                input=[{"role": "user", "content": prompt}],
                text_format=kwargs["response_format"],
            )
            return response.output_parsed
        llm = (
            lambda content: client.chat.completions.create(
                model=routed_model,
                messages=[{"role": "user", "content": content}],
                **kwargs,
            )
            .choices[0]
            .message.content
        )

        return llm(prompt)

    elif model == "vllm":
        DEFAULT_VLLM_MODEL = "neuralmagic/Meta-Llama-405B-Instruct-FP8"
        vllm_api_url = kwargs.pop("vllm_api_url", config.VLLM_API_URL or "http://localhost:8000")
        vllm_api_key = kwargs.pop("vllm_api_key", config.VLLM_API_KEY)
        vllm_model = kwargs.pop("vllm_model", config.VLLM_MODEL or DEFAULT_VLLM_MODEL)

        client = OpenAI(base_url=f"{vllm_api_url}/v1", api_key=vllm_api_key)

        messages = [
            {"role": "user", "content": prompt},
        ]

        if "response_format" in kwargs.keys():
            response = client.beta.chat.completions.parse(
                model=vllm_model,
                messages=messages,
                response_format=kwargs["response_format"],
            )
            return response.choices[0].message.parsed
        response = client.chat.completions.create(
            model=vllm_model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    elif model.startswith("azure:"):
        from openai import AzureOpenAI

        azure_api_key = config.AZURE_API_KEY
        azure_endpoint = config.AZURE_ENDPOINT
        azure_deployment = model.split(":", 1)[1]

        client = OpenAI(
            api_key=azure_api_key,
            base_url=azure_endpoint,
        )

        response = client.chat.completions.create(
            model=azure_deployment,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content

    elif model.startswith("openrouter/"):
        openrouter_api_url = kwargs.pop("openrouter_api_url", config.OPENROUTER_API_URL or "https://openrouter.ai/api/v1")
        openrouter_api_key = kwargs.pop("openrouter_api_key", config.OPENROUTER_API_KEY)
        openrouter_model = model.removeprefix("openrouter/").strip()

        if not openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Please set it in your .env file or environment."
            )

        openrouter_api_url = openrouter_api_url.rstrip("/")
        if not openrouter_api_url.endswith("/v1"):
            openrouter_api_url = f"{openrouter_api_url}/v1"

        client = OpenAI(base_url=openrouter_api_url, api_key=openrouter_api_key)

        messages = [
            {"role": "user", "content": prompt},
        ]

        if "response_format" in kwargs.keys():
            response = client.beta.chat.completions.parse(
                model=openrouter_model,
                messages=messages,
                response_format=kwargs["response_format"],
            )
            return response.choices[0].message.parsed
        response = client.chat.completions.create(
            model=openrouter_model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    else:
        import litellm
        from litellm import completion

        litellm._logging.handler.setLevel(logger.getEffectiveLevel())

        tokens = len(prompt.split())

        if show_cost:
            logger.debug(f"\nNumber of tokens: {tokens}")
            cost = (0.1 / 1_000_000) * tokens
            logger.debug(f"Estimated cost for {model}: ${cost:.10f}\n")

        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content


def break_report_to_insights(
    report_text,
    model="together_ai/meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    save_json=False,
    max_retries=3,
    **kwargs,
):
    insight_claims_prompt = f"""
    Please break down the following report text into insight claims. Each insight claim should be:
    1. A single insight, that might include multiple statements and claims
    2. Independent and self-contained
    3. Each claim can have more than one sentence, but should be focused on a single insight
    4. Support each insight with citations from the report text following these specific rules:
       - Usually at the end of the report there is a list of citations with numbers
       - In the report text, citations are referenced with numbers in square brackets like [1], [2], [^1], etc.
       - When providing citations, write the actual name of the cited documents (file names or URLs), NOT the reference numbers
       - Do NOT find citations based on your general knowledge - only use citations that are explicitly presented in the report
       - For each insight you extract, look for citation markers in that specific text section and find the corresponding citation names from the reference list
       - If no citations are found for a specific insight, leave citations empty
    5. Citations should be in one of these formats (various formats will be automatically normalized):
       - Valid URLs: "https://www.example.com/article", "https://techcrunch.com/2023/report.html"
       - File names: "quarterly_report.pdf", "market_analysis.docx", "shared/file.pdf"
       - MatterMost chats: "MatterMost-Channel-Team-User" or natural descriptions like "Mattermost Message - Enterprise Chat (User: john.doe, Team: Compliance, Channel: General)"
       - Email messages: "RoundCube-from@email-to@emails-Subject" or natural descriptions like "Email from sarah.johnson@company.com on 20 Jan 2025"
       - NOTE: Various citation formats are supported and will be automatically normalized during evaluation
    6. Do not include general summaries, opinions, or claims that lack citation, just the sentences that are facts.
    7. Each claim should be a concise but complete sentence.

    ## Report text:
    <START OF REPORT>
    {report_text}
    <END OF REPORT>

    ## Output format:
    Please return the insight claims as a JSON array. For example:
    [
        {{
            "claim": "The company's revenue increased by 15% in Q3 2023",
            "citations": ["https://techfundingnews.com/salesforce-acquires-convergence-how-it-competes-with-openai-anthropic-and-googles-duet-ai/"]
        }},
        {{
            "claim": "The new product launch contributed to the growth",
            "citations": ["product_launch.pdf", "quarterly_report.pdf"]
        }},
        {{
            "claim": "Compliance team discussed FSMA requirements",
            "citations": ["MatterMost-fsma_compliance-compliance_team-john.doe"]
        }},
        {{
            "claim": "Budget review meeting was scheduled for next quarter",
            "citations": ["RoundCube-jason.kim@leesmarket.com-emily.patel@leesmarket.com,sophia.lee@leesmarket.com-Q2 Budget Review and Planning"]
        }},
        {{
            "claim": "This claim has no supporting citations",
            "citations": []
        }}
    ]

    Return only valid JSON, no additional text.

    ### Just use the report between <START OF REPORT> and <END OF REPORT> tags to generate insights.
    ### If no insights found, return an empty JSON array: []
    ### Do not use the example outputs as report content.
    """  # noqa: E501

    for attempt in range(max_retries):
        try:
            insight_claims_reponse = prompt_llm(insight_claims_prompt, model, temperature=0, **kwargs)

            try:
                response_clean = insight_claims_reponse.strip()

                start_idx = response_clean.find("[")
                end_idx = response_clean.rfind("]")

                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response_clean[start_idx : end_idx + 1]
                    insight_list = json.loads(json_str)

                    if not isinstance(insight_list, list):
                        raise ValueError("Response is not a JSON array")

                    validated_insights = []
                    for insight in insight_list:
                        if isinstance(insight, dict) and "claim" in insight and "citations" in insight:
                            claim = insight["claim"].strip() if isinstance(insight["claim"], str) else ""
                            citations = insight["citations"] if isinstance(insight["citations"], list) else []

                            if len(claim) >= 10:
                                validated_insights.append({"claim": claim, "citations": citations})

                    if save_json:
                        with open("insight_claims.json", "w") as f:
                            json.dump(validated_insights, f, indent=4)

                    return validated_insights
                else:
                    logger.warning(f"No valid JSON array found in response (attempt {attempt + 1})")

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON response (attempt {attempt + 1}): {e}")
                logger.warning(f"Response was: {insight_claims_reponse}")

        except Exception as e:
            logger.warning(f"Error in insight extraction (attempt {attempt + 1}): {e}")


    logger.error(f"Failed to extract insights after {max_retries} attempts")
    return []


def clean_citation(citation):
    """Clean and normalize citation to expected format."""
    if citation is None:
        return None

    try:
        from drflow.agents.citation_normalizer import normalize_citation

        normalized = normalize_citation(citation)
        if normalized:
            return _process_normalized_citation(normalized)
    except ImportError:
        logger.warning("citation_normalizer not available, falling back to legacy parsing")
    except Exception as e:
        logger.debug(f"Error in citation normalization: {e}, falling back to legacy parsing")

    return _legacy_clean_citation(citation)


def _process_normalized_citation(citation):
    """Process normalized citation to the internal format expected by get_content."""
    if citation is None:
        return None

    if citation.startswith("http"):
        return citation

    if re.search(r'\.\w{1,5}$', citation):
        return citation

    if citation.lower().startswith("mattermost_"):
        citation_lower = citation.lower()
        parts = citation_lower.split("_")

        if len(parts) >= 4:
            channel_name = parts[1]
            team_name = parts[2]
            user_name = "-".join(parts[3:])
            return f"mattermost<sep>{channel_name}<sep>{team_name}<sep>{user_name}"
        else:
            logger.warning(f"⚠️  WARNING: Normalized MatterMost citation has insufficient parts: {citation}")
            return None

    if citation.lower().startswith("roundcube-"):
        citation_lower = citation.lower()
        parts = citation_lower.split("-", 3)

        if len(parts) >= 4:
            from_email = parts[1]
            to_emails = parts[2]
            subject = parts[3]
            return f"roundcube<sep>{from_email}<sep>{to_emails}<sep>{subject}"
        else:
            logger.warning(f"⚠️  WARNING: Normalized RoundCube citation has insufficient parts: {citation}")
            return None

    return citation


def _legacy_clean_citation(citation):
    """Legacy citation cleaning logic (fallback)."""
    if citation is None:
        return None

    if citation.startswith("http"):
        return citation

    if citation.lower().startswith("mattermost-"):
        citation_lower = citation.lower()
        parts = citation_lower.split("-")

        if len(parts) > 4:
            logger.warning(f"MatterMost citation has more than 4 parts, using LLM to parse: {citation}")

            llm_prompt = f"""
            Parse the following MatterMost citation and extract the channel name, team name, and user name.
            Citation: {citation}

            Please return only the three values separated by commas in this exact format:
            channel_name,team_name,user_name

            Do not include any other text or formatting.
            """

            try:
                response = prompt_llm(llm_prompt, model="gpt-4o-mini", temperature=0)
                parsed_parts = [part.strip() for part in response.split(",")]

                if len(parsed_parts) == 3:
                    channel_name, team_name, user_name = parsed_parts
                    return f"mattermost<sep>{channel_name}<sep>{team_name}<sep>{user_name}"
                else:
                    logger.warning(f"LLM failed to parse MatterMost citation properly: {response}")
                    return f"mattermost<sep>{parts[1]}<sep>{parts[2]}<sep>{parts[3]}"
            except Exception as e:
                logger.warning(f"Error using LLM to parse MatterMost citation: {e}")
                return f"mattermost<sep>{parts[1]}<sep>{parts[2]}<sep>{parts[3]}"
        else:
            if len(parts) == 4:
                return f"mattermost<sep>{parts[1]}<sep>{parts[2]}<sep>{parts[3]}"
            else:
                logger.warning(f"⚠️  WARNING: Citation {citation} is not in the expected format for mattermost chat.")
                return None

    if citation.lower().startswith("roundcube-"):
        citation_lower = citation.lower()
        parts = citation_lower.split("-")

        if len(parts) > 4:
            from_email = parts[1]
            to_emails = parts[2]
            subject = "-".join(parts[3:])
            return f"roundcube<sep>{from_email}<sep>{to_emails}<sep>{subject}"
        else:
            if len(parts) == 4:
                return f"roundcube<sep>{parts[1]}<sep>{parts[2]}<sep>{parts[3]}"
            else:
                logger.warning(f"⚠️  WARNING: Citation {citation} is not in the expected format for email.")
                return None

    file_match = re.search(r"([^/\\:]+\.[^/\\:]+)", citation, re.IGNORECASE)
    if file_match is None:
        return None
    return file_match.group(1).strip()


def get_embeddings(texts, embedding_model="text-embedding-3-small", method="openai"):
    if method == "openai":
        routed_embedding_model = embedding_model
        if config.OPENAI_API_KEY:
            client = OpenAI(api_key=config.OPENAI_API_KEY)
        elif config.OPENROUTER_API_KEY:
            client = OpenAI(
                base_url=_get_openrouter_base_url(),
                api_key=config.OPENROUTER_API_KEY,
            )
            if "/" not in routed_embedding_model:
                routed_embedding_model = f"openai/{routed_embedding_model}"
        else:
            raise ValueError(
                "Neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set for embeddings."
            )
        response = client.embeddings.create(input=texts, model=routed_embedding_model)
        embeddings = np.array([item.embedding for item in response.data])
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    elif method == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(embedding_model)
        embeddings = embedding_model.encode(texts)
    else:
        raise ValueError(f"Invalid embedding method: {method}")
    return embeddings


def get_most_relevant_chunks(query, content, top_k=5, chunk_size=2048, max_chunks=200, verbose=False):
    """
    Get the most relevant chunks from content using RAG retrieval.

    Args:
        query: The query/insight to search for
        content: The content to search in
        top_k: Number of top relevant chunks to return
        max_chunks: Maximum number of chunks to process (to prevent API limits)

    Returns:
        List of most relevant chunks
    """
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]

    valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    if not valid_chunks:
        return []

    if len(valid_chunks) > max_chunks:
        if verbose:
            logger.info(f"Too many chunks ({len(valid_chunks)}), sampling {max_chunks} chunks across the document")

        step = len(valid_chunks) / max_chunks
        sampled_indices = [int(i * step) for i in range(max_chunks)]
        valid_chunks = [valid_chunks[i] for i in sampled_indices]

    if len(valid_chunks) <= top_k:
        return valid_chunks

    import faiss

    chunk_embeddings = get_embeddings(valid_chunks, embedding_model="text-embedding-3-small", method="openai")

    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    faiss.normalize_L2(chunk_embeddings)
    index.add(chunk_embeddings)

    query_embedding = get_embeddings([query], embedding_model="text-embedding-3-small", method="openai")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, min(top_k, len(valid_chunks)))

    relevant_chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(valid_chunks):
            relevant_chunks.append(valid_chunks[idx])

    return relevant_chunks


def get_factuality_verdict_multi(insight, citations, file_list=None, model="gpt-4o-mini", max_retries=3):
    """
    Check if the insight can be answered with multiple citations/sources.

    Args:
        insight: The claim to verify
        citations: List of citations (can contain URLs or file names)
        file_list: Dictionary mapping file names to file paths
        model: LLM model to use for verification
        max_retries: Number of retry attempts

    Returns:
        Dict with is_factual, explanation, and source_details
    """
    if not citations or len(citations) == 0:
        return {
            "is_factual": False,
            "explanation": "No citations provided to verify the claim",
            "source_details": [],
        }

    combined_content = ""
    source_details = []

    for citation in citations:
        cleaned_citation = clean_citation(citation)

        if cleaned_citation is None:
            source_details.append({"citation": citation, "content_available": False, "content_length": 0})
            continue

        content = get_content(cleaned_citation, file_list)
        if content:
            combined_content += f"\n\nSource: {citation}\n{content}\n"
            source_details.append(
                {
                    "citation": citation,
                    "content_available": True,
                    "content_length": len(content),
                }
            )
        else:
            source_details.append({"citation": citation, "content_available": False, "content_length": 0})

    if not combined_content.strip():
        return {
            "is_factual": False,
            "explanation": "No valid content could be retrieved from any of the provided citations",
            "source_details": source_details,
        }

    relevant_chunks = get_most_relevant_chunks(insight, combined_content)
    context = "\n".join(relevant_chunks)

    factuality_prompt = f"""
    Given the following relevant source context from multiple sources and an insight, determine if the insight is factually supported by the sources.

    Relevant Source Materials (from multiple sources):
    {context}

    Atomic Claim: {insight}

    EVALUATION CRITERIA:
    The claim is factual if the core factual content is supported by the sources. You should be strict about important details but flexible about exact wording:

    REQUIRED for TRUE:
    1. All key factual details (numbers, dates, names, percentages, specific facts) must be present in at least one source
    2. The main substance and meaning of the claim must be supported by the source contexts
    3. No part of the claim should contradict the information in any of the sources

    ACCEPTABLE variations:
    - Different wording or phrasing that conveys the same meaning
    - Paraphrasing or summarization of the source information
    - Minor linguistic differences that don't change the factual content

    Mark as FALSE if:
    - Important factual details are missing, incorrect, or unsupported across all sources
    - The claim contradicts information in any of the sources
    - The core meaning cannot be verified from any of the source contexts

    EXAMPLES:

    FACTUAL CLAIM (TRUE):
    Source A: "Company ABC reported $50M revenue in Q3 2023"
    Source B: "ABC saw 15% growth this quarter compared to Q2"
    Claim: "ABC's revenue grew by 15% to $50M in Q3 2023"
    → TRUE (facts from both sources combined)

    UNFACTUAL CLAIM (FALSE):
    Source A: "Company ABC reported $50M revenue in Q3 2023"  
    Source B: "ABC saw 10% decline this quarter"
    Claim: "ABC's revenue grew by 15% in Q3 2023"
    → FALSE (contradicts Source B)

    UNFACTUAL CLAIM (FALSE):
    Source A: "The CEO mentioned potential expansion plans"
    Claim: "The CEO confirmed definite expansion of 25% next year"
    → FALSE (adds unsupported specifics not in any source)

    Focus on the substantive factual accuracy rather than exact word-for-word matching.

    You MUST respond with either true or false under the <factual> tag.
    Then provide a brief explanation under the <explanation> tag explaining which parts are supported or not supported and from which sources.

    Format your response EXACTLY as:

    <factual>false</factual>
    <explanation>The claim is mentioned in source X... but source Y contradicts...</explanation>

    """  # noqa: E501

    for attempt in range(max_retries):
        try:
            response = prompt_llm(factuality_prompt, model, temperature=0)

            factuality_response = re.search(
                r"<factual>(.*?)</factual>\s*<explanation>(.*?)</explanation>",
                response,
                re.DOTALL,
            )

            if factuality_response:
                is_factual_text = factuality_response.group(1).strip().lower()
                explanation_text = factuality_response.group(2).strip()

                if is_factual_text in ["true", "false"]:
                    fact_dict = {
                        "is_factual": is_factual_text == "true",
                        "explanation": explanation_text,
                        "source_details": source_details,
                    }
                    return fact_dict
                else:
                    logger.warning(f"Invalid factual value: {is_factual_text}. Retrying...")
            else:
                logger.warning(f"Failed to parse LLM response (attempt {attempt + 1}): {response}")

                factual_only = re.search(r"<factual>(.*?)</factual>", response, re.DOTALL)
                if factual_only:
                    is_factual_text = factual_only.group(1).strip().lower()
                    if is_factual_text in ["true", "false"]:
                        return {
                            "is_factual": is_factual_text == "true",
                            "explanation": "No detailed explanation provided by model",
                            "source_details": source_details,
                        }

                if "true" in response.lower() and "false" not in response.lower():
                    return {
                        "is_factual": True,
                        "explanation": response.strip(),
                        "source_details": source_details,
                    }
                elif "false" in response.lower() and "true" not in response.lower():
                    return {
                        "is_factual": False,
                        "explanation": response.strip(),
                        "source_details": source_details,
                    }

        except Exception as e:
            logger.warning(f"Error in multi-source factuality check (attempt {attempt + 1}): {e}")

        if attempt < max_retries - 1:
            factuality_prompt = f"""
            Given the following relevant source context from multiple sources and an insight, determine if the insight is factually supported by the sources.

            Relevant Source Materials (from multiple sources):
            {context}

            Atomic Claim: {insight}

            EVALUATION CRITERIA:
            The claim is factual if the core factual content is supported by the sources. Focus on important details rather than exact wording:

            - All key factual details (numbers, dates, names, percentages) must be present in at least one source
            - The main substance and meaning must be supported by the source contexts
            - Different wording or paraphrasing is acceptable if the meaning is preserved
            - Mark as FALSE if important details are missing, incorrect, or contradicted by any source

            EXAMPLES:
            TRUE: Source A "$50M revenue", Source B "15% growth" → Claim "revenue grew 15% to $50M"
            FALSE: Source A "$50M revenue", Source B "10% decline" → Claim "revenue grew 15%"

            IMPORTANT: You must respond in the exact format below. Do not add any other text.

            <factual>true</factual>
            <explanation>Your detailed explanation here mentioning which sources support or contradict the claim</explanation>

            Replace 'true' with 'false' if key factual details are not supported or contradicted by the sources.
            """  # noqa: E501

    logger.error(f"Failed to get multi-source factuality verdict after {max_retries} attempts")
    return {
        "is_factual": False,
        "explanation": "Failed to determine factuality due to parsing errors",
        "source_details": source_details,
    }


class SourceReader:
    def __init__(self):
        self.parsers = {
            ".txt": self.parse_text,
            ".json": self.parse_json,
            ".jsonl": self.parse_jsonl,
            ".csv": self.parse_csv,
            ".pdf": self.parse_pdf,
            ".docx": self.parse_docx,
            ".xlsx": self.parse_xlsx,
            ".pptx": self.parse_pptx,
            ".md": self.parse_markdown,
        }

    def parse_website(self, url: str) -> Optional[Tuple[str, str]]:
        """
        Parse a website URL and extract the title and main content.
        Can handle both HTML pages and direct file links.

        Args:
            url: URL of the website or file to parse

        Returns:
            Tuple of (title, content) or None if parsing fails
        """

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.google.com/",
        }

        session = requests.Session()
        session.headers.update(headers)

        response = session.get(url)

        if response.ok:
            soup = BeautifulSoup(response.text, "html.parser")

            content_snippets = []
            for tag in ["article", "main", "body", "div", "p", "section"]:
                content_tag = soup.find(tag)
                if content_tag:
                    content = content_tag.get_text(separator=" ", strip=True)
                    if content:
                        content_snippets.append(content)

            if len(content_snippets) > 0:
                return "\n\n".join(content_snippets)

            content = soup.get_text(separator=" ", strip=True)
            if not content:
                return None

        else:
            content = None

        return content

    def parse_file(self, file_path: Path) -> Optional[Tuple[str, str]]:
        """
        Parse a file and extract title and content.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (title, content) or None if parsing fails
        """
        try:
            if not file_path.exists() or not file_path.is_file():
                logger.warning(f"File does not exist or is not a regular file: {file_path}")
                return None

            ext = file_path.suffix.lower()

            if ext in self.parsers:
                return self.parsers[ext](file_path)

            logger.warning(f"Unsupported file type: {ext}")
            return None

        except Exception as e:
            logger.warning(f"Error parsing file {file_path}: {e}")
            return None

    def _describe_dataframe_to_sentences(self, df: pd.DataFrame) -> str:
        """Convert DataFrame summary statistics into human-readable sentences."""
        df_num = df.apply(lambda x: pd.to_numeric(x, errors="coerce"))
        summary_df = df_num.describe()
        lines = []
        for col in summary_df.columns:
            stats = summary_df[col]
            lines.append(f"For column '{col}':")
            lines.append(f"  - Count: {stats['count']:.0f}")
            lines.append(f"  - Mean: {stats['mean']:.2f}")
            lines.append(f"  - Standard deviation: {stats['std']:.2f}")
            lines.append(f"  - Min: {stats['min']:.2f}")
            lines.append(f"  - 25th percentile: {stats['25%']:.2f}")
            lines.append(f"  - Median (50%): {stats['50%']:.2f}")
            lines.append(f"  - 75th percentile: {stats['75%']:.2f}")
            lines.append(f"  - Max: {stats['max']:.2f}")
            lines.append("")
        return "\n".join(lines)

    def parse_pdf(self, file_path: Path) -> Optional[Tuple[str, str]]:
        """
        Parse PDF files using PyMuPDF (fitz).

        Args:
            file_path: Path to the PDF file

        Returns:
            Tuple of (title, content) or None if parsing fails
        """
        try:
            if not file_path.exists() or not file_path.is_file():
                logger.warning(f"File does not exist or is not a regular file: {file_path}")
                return None

            doc = fitz.open(str(file_path))
            content = ""

            for page in doc:
                content += page.get_text()

            doc.close()

            if not content.strip():
                logger.warning(f"No extractable text found in: {file_path}")
                return None

            title = file_path.stem.replace("_", " ").title()

            content = re.sub(r"\n{3,}", "\n\n", content.strip())

            return title, content

        except Exception as e:
            logger.warning(f"Error parsing PDF file {file_path}: {e}")
            return None

    def parse_docx(self, file_path: Path) -> Optional[Tuple[str, str]]:
        """Parse DOCX files using python-docx."""
        try:
            doc = Document(file_path)

            title = file_path.stem.replace("_", " ").title()
            if doc.core_properties.title:
                title = doc.core_properties.title

            full_text = []

            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)

            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        full_text.append(" | ".join(row_text))

            content = "\n\n".join(full_text)

            return title, content
        except Exception as e:
            logger.warning(f"Error parsing DOCX file {file_path}: {e}")
            return None

    def parse_xlsx(self, file_path: Path) -> Optional[Tuple[str, str]]:
        """Parse XLSX files using pandas."""
        try:
            excel_file = pd.ExcelFile(file_path)
            sheets = excel_file.sheet_names

            title = file_path.stem.replace("_", " ").title()

            content_parts = []
            for sheet_name in sheets:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                num_rows, num_cols = df.shape

                content_parts.append(f"Sheet '{sheet_name}': {num_rows} rows × {num_cols} columns")

                if num_rows > 0 and num_cols > 0:
                    content_parts.append("Data summary:\n")
                    content_parts.append(self._describe_dataframe_to_sentences(df))

                if num_cols > 0:
                    cols = df.columns.tolist()
                    content_parts.append(f"Columns: {', '.join(str(col) for col in cols)}")

                if num_rows > 0:
                    content_parts.append("\nFirst 20 rows:")
                    content_parts.append(df.head(20).to_string())

            content = "\n\n".join(content_parts)

            return title, content
        except Exception as e:
            logger.warning(f"Error parsing XLSX file {file_path}: {e}")
            return None

    def parse_pptx(self, file_path: Path) -> Optional[Tuple[str, str]]:
        """Parse PPTX files by extracting text from slides."""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                subprocess.run(["unzip", "-q", str(file_path), "-d", tmp_dir], check=True)

                slide_files = list(Path(tmp_dir).glob("ppt/slides/slide*.xml"))
                slide_texts = []

                for slide_file in sorted(slide_files):
                    with open(slide_file, "r", encoding="utf-8", errors="ignore") as f:
                        slide_xml = f.read()
                        texts = re.findall(r"<a:t>([^<]+)</a:t>", slide_xml)
                        if texts:
                            slide_texts.append(" ".join(texts))

                title = file_path.stem.replace("_", " ").title()

                if slide_texts:
                    content = f"Presentation with {len(slide_texts)} slides:\n\n"
                    for i, text in enumerate(slide_texts, 1):
                        content += f"Slide {i}:\n{text}\n\n"
                else:
                    content = "Empty presentation"

                return title, content
        except Exception as e:
            logger.warning(f"Error parsing PPTX file {file_path}: {e}")
            return None

    def parse_text(self, file_path: Path) -> Optional[Tuple[str, str]]:
        """Parse plain text files."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            title = file_path.stem.replace("_", " ").title()

            content = re.sub(r"\n{3,}", "\n\n", content.strip())

            return title, content
        except Exception as e:
            logger.warning(f"Error parsing text file {file_path}: {e}")
            return None

    def parse_csv(self, file_path: Path) -> Optional[Tuple[str, str]]:
        """Parse CSV files."""
        try:
            df = pd.read_csv(file_path)

            title = file_path.stem.replace("_", " ").title()

            content_parts = []

            num_rows, num_cols = df.shape
            content_parts.append(f"CSV data with {num_rows} rows and {num_cols} columns")

            if num_rows > 0 and num_cols > 0:
                content_parts.append("Data summary:\n")
                content_parts.append(self._describe_dataframe_to_sentences(df))

            if num_cols > 0:
                cols = df.columns.tolist()
                content_parts.append(f"Headers: {', '.join(str(col) for col in cols)}")

            if num_rows > 0:
                content_parts.append("\nFirst few rows:")
                content_parts.append(df.head(5).to_string())

            content = "\n\n".join(content_parts)

            return title, content
        except Exception as e:
            logger.warning(f"Error parsing CSV file {file_path}: {e}")
            return None

    def parse_json(self, file_path: Path) -> Optional[Tuple[str, str]]:
        """Parse JSON files."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)

            title = file_path.stem.replace("_", " ").title()

            content = ""

            text_fields = [
                "description",
                "summary",
                "text",
                "content",
                "body",
                "abstract",
            ]

            if isinstance(data, dict):
                for title_field in ["title", "name", "heading"]:
                    if title_field in data and isinstance(data[title_field], str) and data[title_field].strip():
                        title = data[title_field].strip()
                        break

                for field in text_fields:
                    if field in data and isinstance(data[field], str) and data[field].strip():
                        content = data[field].strip()
                        break

            if not content:
                content = json.dumps(data, indent=2)

            return title, content
        except Exception as e:
            logger.warning(f"Error parsing JSON file {file_path}: {e}")
            return None

    def parse_jsonl(self, file_path: Path) -> Optional[Tuple[str, str]]:
        """Parse JSONL files."""
        key_fields = ["direct_post", "post"]
        content = ""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                for line in lines:
                    if set(json.loads(line).keys()) & set(key_fields):
                        content += line

            title = file_path.stem.replace("_", " ").title()

            return title, content
        except Exception as e:
            logger.warning(f"Error parsing JSONL file {file_path}: {e}")
            return None

    def parse_markdown(self, file_path: Path) -> Optional[Tuple[str, str]]:
        """Parse markdown files."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            title = file_path.stem.replace("_", " ").title()
            lines = content.strip().split("\n")
            if lines and lines[0].startswith("# "):
                title = lines[0].lstrip("# ").strip()

            content = re.sub(r"^#.*$", "", content, flags=re.MULTILINE)
            content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
            content = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", content)
            content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
            content = re.sub(r"`([^`]+)`", r"\1", content)
            content = re.sub(r"^>.*$", "", content, flags=re.MULTILINE)
            content = re.sub(r"^[-*_]{3,}$", "", content, flags=re.MULTILINE)
            content = re.sub(r"^[-*+]\s+", "", content, flags=re.MULTILINE)
            content = re.sub(r"^\d+\.\s+", "", content, flags=re.MULTILINE)

            content = re.sub(r"\n{3,}", "\n\n", content.strip())

            return title, content
        except Exception as e:
            logger.warning(f"Error parsing markdown file {file_path}: {e}")
            return None


def _read_jsonl_file_content(file_path):
    """Read all content from a JSONL file (mattermost chat, emails, etc.)."""
    content_parts = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") == "post":
                        post = data.get("post", {})
                        user = post.get("user", "Unknown")
                        message = post.get("message", "")
                        channel = post.get("channel", "")
                        created = post.get("created_at", "")
                        content_parts.append(
                            f"[{user} in #{channel}] ({created}): {message}"
                        )
                    elif data.get("type") == "email":
                        sender = data.get("from", "Unknown")
                        subject = data.get("subject", "")
                        body = data.get("body", data.get("text", ""))
                        content_parts.append(
                            f"From: {sender}\nSubject: {subject}\n{body}"
                        )
                    else:
                        content_parts.append(str(data)[:500])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Error reading JSONL file {file_path}: {e}")
    return "\n\n".join(content_parts) if content_parts else None


def get_content(source, file_list=None):
    if str(source).lower() == "none":
        return None

    if file_list:
        file_match_name = re.search(r"([^/\\:]+\.\w{1,5})$", source)
        lookup_name = file_match_name.group(1).strip() if file_match_name else source
        for fname, fpath in file_list.items():
            if fname == lookup_name or fname == source:
                fpath_str = str(fpath)
                if fpath_str.endswith(".jsonl"):
                    return _read_jsonl_file_content(fpath_str)
                result = SourceReader().parse_file(fpath)
                if result is not None:
                    return result[1] if isinstance(result, tuple) else result
                return None

    if source.startswith("http"):
        result = SourceReader().parse_website(source)
        return result if result is not None else None
    elif source.lower().startswith("mattermost"):
        if file_list is None:
            raise ValueError("file_list is required for MatterMost citations")

        parts = source.split("<sep>")
        if len(parts) != 4:
            logger.warning(f"Invalid MatterMost citation format: {source}")
            return None

        _, channel_name, team_name, user_name = [part.lower() for part in parts]

        mattermost_files = [
            filename
            for filename in file_list.keys()
            if filename.lower().startswith("mattermost_chat") and filename.endswith(".jsonl")
        ]

        collected_posts = []

        for filename in mattermost_files:
            file_path = file_list[filename]
            try:
                display_to_internal_team: dict = {}
                display_to_internal_channel: dict = {}
                all_lines = []

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        all_lines.append(data)
                        if data.get("type") == "team":
                            ti = data.get("team", {})
                            dn = ti.get("display_name", "").lower()
                            nm = ti.get("name", "").lower()
                            if dn and nm:
                                display_to_internal_team[dn] = nm
                        elif data.get("type") == "channel":
                            ci = data.get("channel", {})
                            dn = ci.get("display_name", "").lower()
                            nm = ci.get("name", "").lower()
                            if dn and nm:
                                display_to_internal_channel[dn] = nm

                resolved_team = display_to_internal_team.get(team_name, team_name)
                resolved_channel = display_to_internal_channel.get(channel_name, channel_name)

                for data in all_lines:
                    if data.get("type") != "post":
                        continue
                    post = data.get("post", {})
                    if (
                        post.get("team", "").lower() == resolved_team
                        and post.get("channel", "").lower() == resolved_channel
                        and post.get("user", "").lower() == user_name
                    ):
                        collected_posts.append(data)
            except Exception as e:
                logger.warning(f"Error reading MatterMost file {filename}: {e}")
                continue

        if not collected_posts:
            logger.warning(f"No matching MatterMost posts found for {source}")
            return None

        content_parts = []
        for post_data in collected_posts:
            post = post_data.get("post", {})
            message = post.get("message", "")
            created_at = post.get("created_at", "")
            content_parts.append(f"Message: {message} (Created: {created_at})")

        return "\n\n".join(content_parts)

    elif source.lower().startswith("roundcube"):
        if file_list is None:
            raise ValueError("file_list is required for RoundCube citations")

        parts = source.split("<sep>")
        if len(parts) != 4:
            logger.warning(f"Invalid RoundCube citation format: {source}")
            return None

        _, from_email, to_emails, subject = [part.lower() for part in parts]

        to_email_list = [email.strip() for email in to_emails.split(",")]

        roundcube_files = [
            filename
            for filename in file_list.keys()
            if filename.lower().startswith("roundcube_email") and filename.endswith(".jsonl")
        ]

        collected_emails = []
        has_valid_to = any(t.strip() for t in to_email_list)

        for filename in roundcube_files:
            file_path = file_list[filename]
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            if data.get("type") == "email":
                                email_data = data
                                email_from = email_data.get("from", "").lower()
                                email_to = [email.lower() for email in email_data.get("to", [])]
                                email_subject = email_data.get("subject", "").lower()

                                if email_from != from_email:
                                    continue

                                subject_match = (email_subject == subject)
                                if not subject_match and subject:
                                    subject_match = (
                                        subject in email_subject
                                        or email_subject in subject
                                    )

                                if subject_match:
                                    if has_valid_to:
                                        if any(to_email in email_to for to_email in to_email_list if to_email.strip()):
                                            collected_emails.append(email_data)
                                    else:
                                        collected_emails.append(email_data)
            except Exception as e:
                logger.warning(f"Error reading RoundCube file {filename}: {e}")
                continue

        if not collected_emails:
            logger.warning(f"No matching RoundCube emails found for {source}")
            return None

        content_parts = []
        attachments_list = []

        for email_data in collected_emails:
            body = email_data.get("body", "")
            date = email_data.get("date", "")
            from_name = email_data.get("from_name", "")
            subject_original = email_data.get("subject", "")
            attachments = email_data.get("attachments", [])

            if attachments:
                attachments_list.extend(attachments)

            email_content = f"From: {from_name} ({email_data.get('from', '')})\n"
            email_content += f"Subject: {subject_original}\n"
            email_content += f"Date: {date}\n"
            email_content += f"Body: {body}"

            content_parts.append(email_content)

        return "\n\n---EMAIL SEPARATOR---\n\n".join(content_parts)

    else:
        if file_list is None:
            raise ValueError("file_list is required")

        file_match = re.search(r"([^/\\:]+\.[^/\\:]+)", source, re.IGNORECASE)
        if file_match is None:
            return None

        file_name = file_match.group(1).strip()

        if file_name in file_list:
            path = file_list[file_name]
            result = SourceReader().parse_file(path)
            if result is not None:
                return result[1]
            else:
                return None
        else:
            return None
