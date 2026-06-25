from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from nbt_pipeline.extraction.schemas import TheatreNoteExtraction
from nbt_pipeline.extraction.tools.opcs_lookup import opcs_lookup_tool


@CrewBase
class ExtractorCrew:
    """Crew that extracts structured fields from a single theatre note."""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["extractor"],
            tools=[opcs_lookup_tool],
        )

    @task
    def extract_task(self) -> Task:
        return Task(
            config=self.tasks_config["extract_task"],
            output_pydantic=TheatreNoteExtraction,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
        )
