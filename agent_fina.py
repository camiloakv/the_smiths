from smolagents import ToolCallingAgent, ToolCollection, LiteLLMModel
from mcp import StdioServerParameters

#model = LiteLLMModel(model_id="ollama_chat/qwen2.5:14b", num_ctx=4096)
model = LiteLLMModel(model_id="ollama_chat/gemma4:31b", num_ctx=4096)

server_parameters = StdioServerParameters(
    command="uv",
    args=["run", "server_fina.py"],
    env=None,
)

with ToolCollection.from_mcp(
    server_parameters=server_parameters,
    trust_remote_code=True
) as tool_collection:
    agent = ToolCallingAgent(tools=[*tool_collection.tools], model=model)
    agent.run("Who are the core leaders at NVIDIA?")
