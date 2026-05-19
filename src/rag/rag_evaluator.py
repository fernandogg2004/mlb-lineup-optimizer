"""
RAG Evaluation Pipeline — automated quality gate for the Tactical Briefing system.

Uses RAGAS to measure two critical properties of the agent's outputs:

  Faithfulness (≥ 0.88 required):
      Fraction of atomic claims in the answer that are supported by the
      retrieved context. A score < 0.88 indicates hallucination risk and
      raises FaithfulnessAlertError, blocking deployment.

  Answer Relevance (≥ 0.80 target):
      Cosine similarity between the original question and LLM-generated
      questions reconstructed from the answer. Measures on-topic quality.

Usage:
    evaluator = RAGEvaluator.from_env()
    samples = SyntheticTestDataBuilder.from_env().build(chunks, n=20)
    report = evaluator.evaluate(samples)
    print(report.summary())
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field, field_validator
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

logger = structlog.get_logger(__name__)

_FAITHFULNESS_THRESHOLD = 0.88
_ANSWER_RELEVANCE_THRESHOLD = 0.80

# Default evaluation LLM — gpt-4o-mini balances accuracy with cost
_DEFAULT_EVAL_MODEL = "gpt-4o-mini"
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"  # sufficient for cosine-sim scoring


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FaithfulnessAlertError(Exception):
    """Raised when the mean Faithfulness score falls below the deployment threshold.

    Attributes:
        score: Measured mean faithfulness score (0.0 – 1.0).
        threshold: Required minimum (default 0.88).
        failing_questions: Questions where the individual score fell below threshold.
    """

    def __init__(
        self,
        score: float,
        threshold: float,
        failing_questions: list[str],
    ) -> None:
        self.score = score
        self.threshold = threshold
        self.failing_questions = failing_questions
        super().__init__(
            f"Faithfulness score {score:.4f} is below the deployment threshold "
            f"{threshold:.2f}. Failing questions: {len(failing_questions)}. "
            "Review retrieved context and system prompt before deploying."
        )


class AnswerRelevanceWarning(UserWarning):
    """Non-blocking warning when Answer Relevance drops below target."""


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


class EvaluationSample(BaseModel):
    """One (question, answer, contexts, ground_truth) quadruple for RAGAS."""

    question: str = Field(min_length=5)
    answer: str = Field(min_length=10)
    contexts: list[str] = Field(min_length=1)
    ground_truth: str = ""   # optional; enables context_precision if provided

    @field_validator("contexts")
    @classmethod
    def _at_least_one_context(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("contexts must contain at least one string")
        return v


@dataclass
class RAGEvalConfig:
    faithfulness_threshold: float = _FAITHFULNESS_THRESHOLD
    answer_relevance_threshold: float = _ANSWER_RELEVANCE_THRESHOLD
    eval_llm_model: str = _DEFAULT_EVAL_MODEL
    eval_embed_model: str = _DEFAULT_EMBED_MODEL
    # Raise exception on faithfulness failure (set False to warn-only in dev)
    raise_on_faithfulness_failure: bool = True


@dataclass
class PerSampleScore:
    question: str
    faithfulness: float
    answer_relevancy: float


@dataclass
class EvaluationReport:
    mean_faithfulness: float
    mean_answer_relevancy: float
    per_sample: list[PerSampleScore]
    faithfulness_threshold: float
    answer_relevancy_threshold: float
    n_faithfulness_failures: int
    n_answer_relevancy_warnings: int
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def passed(self) -> bool:
        return self.mean_faithfulness >= self.faithfulness_threshold

    def summary(self) -> str:
        status = "PASS" if self.passed() else "FAIL"
        lines = [
            f"=== RAG Evaluation Report [{status}] ===",
            f"  Faithfulness:      {self.mean_faithfulness:.4f}  "
            f"(threshold ≥ {self.faithfulness_threshold:.2f})",
            f"  Answer Relevance:  {self.mean_answer_relevancy:.4f}  "
            f"(target ≥ {self.answer_relevancy_threshold:.2f})",
            f"  Samples evaluated: {len(self.per_sample)}",
            f"  Faithfulness failures: {self.n_faithfulness_failures}",
            f"  Relevance warnings:    {self.n_answer_relevancy_warnings}",
            f"  Evaluated at: {self.evaluated_at}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic test data builder
# ---------------------------------------------------------------------------


class SyntheticTestDataBuilder:
    """Generates RAGAS-compatible Q/A samples from retrieved scouting chunks.

    For each chunk, uses an LLM to produce one question whose answer is
    derivable from the chunk text. This creates a realistic evaluation dataset
    without needing human-labelled gold standards.
    """

    _GENERATION_PROMPT = (
        "You are a baseball analyst. Based ONLY on the following scouting report "
        "excerpt, write ONE specific question a manager might ask about this player. "
        "Then write a concise answer derived solely from the excerpt.\n\n"
        "Scouting report excerpt:\n{excerpt}\n\n"
        "Respond in JSON with keys: question, answer"
    )

    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini") -> None:
        self._llm = ChatOpenAI(
            model=model,
            api_key=openai_api_key,
            temperature=0.3,
        )
        logger.info("synthetic_builder_ready", model=model)

    @classmethod
    def from_env(cls, model: str = "gpt-4o-mini") -> "SyntheticTestDataBuilder":
        return cls(openai_api_key=os.environ["OPENAI_API_KEY"], model=model)

    def _generate_one(self, chunk_text: str, context: str) -> EvaluationSample | None:
        import json
        prompt = self._GENERATION_PROMPT.format(excerpt=chunk_text[:1_500])
        try:
            response = self._llm.invoke(prompt)
            raw = response.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed: dict[str, str] = json.loads(raw)
            return EvaluationSample(
                question=parsed["question"],
                answer=parsed["answer"],
                contexts=[context],
                ground_truth=parsed.get("answer", ""),
            )
        except Exception as exc:
            logger.warning("synthetic_sample_generation_failed", error=str(exc))
            return None

    def build(
        self,
        chunk_texts: list[str],
        n_samples: int = 20,
    ) -> list[EvaluationSample]:
        """Generate up to n_samples Q/A pairs from chunk_texts."""
        samples: list[EvaluationSample] = []
        for text in chunk_texts[:n_samples]:
            sample = self._generate_one(text, context=text)
            if sample is not None:
                samples.append(sample)
        logger.info(
            "synthetic_dataset_built",
            requested=n_samples,
            generated=len(samples),
        )
        return samples


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class RAGEvaluator:
    """Runs RAGAS evaluation and enforces the Faithfulness deployment gate.

    Usage:
        evaluator = RAGEvaluator.from_env()
        report = evaluator.evaluate(samples)   # raises FaithfulnessAlertError if < 0.88
        print(report.summary())
    """

    def __init__(
        self,
        openai_api_key: str,
        config: RAGEvalConfig | None = None,
    ) -> None:
        self._config = config or RAGEvalConfig()
        eval_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=self._config.eval_llm_model,
                api_key=openai_api_key,
                temperature=0,
            )
        )
        eval_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=self._config.eval_embed_model,
                api_key=openai_api_key,
            )
        )
        # Bind evaluation LLM and embeddings to RAGAS metric objects
        faithfulness.llm = eval_llm
        answer_relevancy.llm = eval_llm
        answer_relevancy.embeddings = eval_embeddings

        logger.info(
            "rag_evaluator_ready",
            eval_model=self._config.eval_llm_model,
            faithfulness_threshold=self._config.faithfulness_threshold,
            answer_relevance_threshold=self._config.answer_relevance_threshold,
        )

    @classmethod
    def from_env(cls, config: RAGEvalConfig | None = None) -> "RAGEvaluator":
        return cls(openai_api_key=os.environ["OPENAI_API_KEY"], config=config)

    def _build_dataset(self, samples: list[EvaluationSample]) -> Dataset:
        return Dataset.from_dict(
            {
                "question": [s.question for s in samples],
                "answer": [s.answer for s in samples],
                "contexts": [s.contexts for s in samples],
                "ground_truth": [s.ground_truth for s in samples],
            }
        )

    def evaluate(self, samples: list[EvaluationSample]) -> EvaluationReport:
        """Run full RAGAS evaluation. Raises FaithfulnessAlertError if gate fails.

        Args:
            samples: List of EvaluationSample (question, answer, contexts, ground_truth).

        Returns:
            EvaluationReport with per-sample scores and aggregated metrics.

        Raises:
            FaithfulnessAlertError: When mean faithfulness < config.faithfulness_threshold
                                    and raise_on_faithfulness_failure is True.
        """
        if not samples:
            raise ValueError("Cannot evaluate an empty sample list")

        logger.info("ragas_evaluation_started", n_samples=len(samples))

        dataset = self._build_dataset(samples)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
        )

        # Extract per-sample scores from result DataFrame
        scores_df = result.to_pandas()
        per_sample: list[PerSampleScore] = []
        for _, row in scores_df.iterrows():
            per_sample.append(
                PerSampleScore(
                    question=str(row.get("question", "")),
                    faithfulness=float(row.get("faithfulness", 0.0)),
                    answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                )
            )

        mean_faithfulness: float = float(result["faithfulness"])
        mean_answer_relevancy: float = float(result["answer_relevancy"])

        failing_faithfulness = [
            s.question
            for s in per_sample
            if s.faithfulness < self._config.faithfulness_threshold
        ]
        answer_relevancy_warnings = [
            s.question
            for s in per_sample
            if s.answer_relevancy < self._config.answer_relevance_threshold
        ]

        report = EvaluationReport(
            mean_faithfulness=mean_faithfulness,
            mean_answer_relevancy=mean_answer_relevancy,
            per_sample=per_sample,
            faithfulness_threshold=self._config.faithfulness_threshold,
            answer_relevancy_threshold=self._config.answer_relevance_threshold,
            n_faithfulness_failures=len(failing_faithfulness),
            n_answer_relevancy_warnings=len(answer_relevancy_warnings),
        )

        logger.info(
            "ragas_evaluation_complete",
            mean_faithfulness=f"{mean_faithfulness:.4f}",
            mean_answer_relevancy=f"{mean_answer_relevancy:.4f}",
            faithfulness_failures=len(failing_faithfulness),
            relevancy_warnings=len(answer_relevancy_warnings),
        )

        if answer_relevancy_warnings:
            import warnings
            warnings.warn(
                f"Answer Relevance {mean_answer_relevancy:.4f} below target "
                f"{self._config.answer_relevance_threshold:.2f} "
                f"({len(answer_relevancy_warnings)} samples affected).",
                AnswerRelevanceWarning,
                stacklevel=2,
            )

        if mean_faithfulness < self._config.faithfulness_threshold:
            if self._config.raise_on_faithfulness_failure:
                raise FaithfulnessAlertError(
                    score=mean_faithfulness,
                    threshold=self._config.faithfulness_threshold,
                    failing_questions=failing_faithfulness,
                )
            else:
                logger.error(
                    "faithfulness_below_threshold",
                    score=mean_faithfulness,
                    threshold=self._config.faithfulness_threshold,
                    failing_questions=failing_faithfulness,
                )

        return report

    def evaluate_live_response(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, float]:
        """Evaluate a single live agent response. Returns per-metric scores.

        Designed for real-time monitoring hooks — does NOT raise on failure,
        only returns the score dict for the caller to decide.
        """
        sample = EvaluationSample(question=question, answer=answer, contexts=contexts)
        config_no_raise = RAGEvalConfig(
            faithfulness_threshold=self._config.faithfulness_threshold,
            answer_relevance_threshold=self._config.answer_relevance_threshold,
            raise_on_faithfulness_failure=False,
        )
        # Temporarily disable raise to get the report
        original_raise = self._config.raise_on_faithfulness_failure
        self._config.raise_on_faithfulness_failure = False
        try:
            report = self.evaluate([sample])
        finally:
            self._config.raise_on_faithfulness_failure = original_raise

        scores = {
            "faithfulness": report.mean_faithfulness,
            "answer_relevancy": report.mean_answer_relevancy,
            "passed": report.passed(),
        }
        logger.info("live_response_evaluated", **{k: v for k, v in scores.items()})
        return scores


# ---------------------------------------------------------------------------
# End-to-end pipeline evaluation helper
# ---------------------------------------------------------------------------


def run_evaluation_pipeline(
    chunk_texts: list[str],
    n_synthetic_samples: int = 20,
    config: RAGEvalConfig | None = None,
) -> EvaluationReport:
    """Convenience function: build synthetic dataset → evaluate → return report.

    Raises FaithfulnessAlertError if the gate fails (< 0.88 by default).

    Args:
        chunk_texts: Raw text chunks from the scouting vector store (sample).
        n_synthetic_samples: How many synthetic Q/A pairs to generate.
        config: Override evaluation thresholds and model settings.

    Returns:
        EvaluationReport with full metrics.
    """
    builder = SyntheticTestDataBuilder.from_env()
    samples = builder.build(chunk_texts, n_samples=n_synthetic_samples)

    evaluator = RAGEvaluator.from_env(config=config)
    report = evaluator.evaluate(samples)

    print(report.summary())
    return report
