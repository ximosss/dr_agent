import os
import asyncio

from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
import weave
from utils.helpers import strip_think_block

from tools import web_search, paper_search, local_docs_lookup, summarize_sources
from prompt import SYSTEM_PROMPT, LOCAL_FILES_SUMMARY_PROMPT, WEB_TEMPLATE_INSTRUCTIONS, PAPER_TEMPLATE_INSTRUCTIONS \

os.environ["HTTP_PROXY"] = "http://localhost:8081"
os.environ["HTTPS_PROXY"] = "http://localhost:8081"

set_tracing_disabled(disabled=True)

model = os.getenv("MODEL_NAME_AT_ENDPOINT")
api_key = os.getenv("BASE_KEY")
base_url = os.getenv("BASE_URL")

# weave.init("deep research agent")

async def main():

    # TODO: human in the loop
    while True:

        user_input = input("what can i do?")
        # two stage agent, concat agent context
        if user_input == "ok":
            break

    # TODO: Agent loop
    while True:
        agent = Agent(
            name="Assistant",
            instructions=SYSTEM_PROMPT,
            model=LitellmModel(model="hosted_vllm/"+ model, base_url=base_url, api_key=api_key),
            tools=[web_search, paper_search, local_docs_lookup, summarize_sources],
        )
        
        break

    # TODO: Refinement
    result = await Runner.run(agent, "What's the Agentic RL")
    print(strip_think_block(result.final_output))

if __name__ == "__main__":
    asyncio.run(main())

    