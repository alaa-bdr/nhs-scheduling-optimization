from nbt_pipeline.config import (
    EXTRACTION_CREW_MAX_ITER,
    EXTRACTION_CREW_MAX_RPM,
    EXTRACTION_EMBEDDING_MODEL,
    EXTRACTION_MODEL,
    EXTRACTION_USE_PDF_KNOWLEDGE,
)
from nbt_pipeline.extraction.schemas import TheatreNoteExtraction

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class ExtractorCrew:
    """Crew that extracts structured fields from a single theatre note."""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    theatre_note_knowledge = TextFileKnowledgeSource(
        file_paths=["extraction/nhs_theatre_note_reference.txt"],
        chunk_size=2000,
        chunk_overlap=100,
        collection_name="nhs_theatre_note_reference",
    )

    def agent_runtime_config(self) -> dict:
        return {
            "llm": EXTRACTION_MODEL,
            "max_iter": EXTRACTION_CREW_MAX_ITER,
            "max_rpm": EXTRACTION_CREW_MAX_RPM,
        }

    @agent
    def extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["extractor"],
            **self.agent_runtime_config(),
        )

    @agent
    def reviewer(self) -> Agent:
        knowledge_sources = [self.theatre_note_knowledge]
        if EXTRACTION_USE_PDF_KNOWLEDGE:
            knowledge_sources.append(
                PDFKnowledgeSource(
                    file_paths=["extraction/OPCS-4.11_NCCS-2026.pdf"],
                    chunk_size=3000,
                    chunk_overlap=300,
                    collection_name="opcs_4_11_nccs_2026",
                )
            )

        return Agent(
            config=self.agents_config["reviewer"],
            knowledge_sources=knowledge_sources,
            **self.agent_runtime_config(),
        )

    @task
    def extract_task(self) -> Task:
        return Task(
            config=self.tasks_config["extract_task"],
            output_pydantic=TheatreNoteExtraction,
        )

    @task
    def review_task(self) -> Task:
        return Task(
            config=self.tasks_config["review_task"],
            context=[self.extract_task()],
            output_pydantic=TheatreNoteExtraction,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            embedder={
                "provider": "google-generativeai",
                "config": {
                    "model_name": EXTRACTION_EMBEDDING_MODEL,
                    "task_type": "RETRIEVAL_DOCUMENT",
                },
            },
        )
