from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.project import CrewBase, agent, crew, task

from nbt_pipeline.config import EXTRACTION_MODEL
from nbt_pipeline.extraction.schemas import TheatreNoteExtraction
from nbt_pipeline.extraction.tools.opcs_lookup import opcs_lookup_tool


@CrewBase
class ExtractorCrew:
    """Crew that extracts structured fields from a single theatre note."""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    theatre_note_knowledge = TextFileKnowledgeSource(
        file_paths=["nhs_theatre_note_reference.txt"],
        chunk_size=2000,
        chunk_overlap=100,
        collection_name="nhs_theatre_note_reference",
    )

    @agent
    def extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["extractor"],
            tools=[opcs_lookup_tool],
            llm=EXTRACTION_MODEL,
        )

    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["reviewer"],
            tools=[opcs_lookup_tool],
            llm=EXTRACTION_MODEL,
            knowledge_sources=[self.theatre_note_knowledge],
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
        )
