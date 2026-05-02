import asyncio
from pydantic_ai import Agent
from pydantic_ai.messages import (
    PartDeltaEvent,
    TextPartDelta,
    TextPart,
    ThinkingPart,
    PartStartEvent,
    ThinkingPartDelta,
)
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

model = OllamaModel(
    "gemma4:26b",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

agent = Agent(
    model,
    instructions="You are a clever artificial construct",
)


async def main():
    async with agent.iter("Explain MoE in one sentence.") as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        if isinstance(event, PartStartEvent):
                            part = event.part
                            if isinstance(part, ThinkingPart):
                                print(f"\n💭 {part.content}", end="", flush=True)
                            elif isinstance(part, TextPart):
                                print(f"\n💬 {part.content}", end="", flush=True)
                        elif isinstance(event, PartDeltaEvent):
                            if isinstance(
                                event.delta, (ThinkingPartDelta, TextPartDelta)
                            ):
                                print(event.delta.content_delta, end="", flush=True)


asyncio.run(main())
