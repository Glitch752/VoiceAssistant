import json
import os
import asyncio
from typing import Iterable, Optional
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, StopReason

load_dotenv()

client = AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# model = "claude-3-haiku-20240307"
model = "claude-3-5-haiku-20241022"

async def run() -> None:
    messages: Iterable[MessageParam] = []
    last_end_reason: Optional[StopReason] = "end_turn"
    
    while True:
        if last_end_reason == "end_turn":
            messages.append({
                "role": "user",
                "content": input("User: ")
            })
        
        # async with client.messages.stream(
        # TODO: Implement streaming.
        response = await client.messages.create(
            max_tokens=2000,
            # For proper caching, all reusable stuff must be before the cache_control message.
            # See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#best-practices-for-effective-caching
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA. If prepended with a tilde (~), landmarks can be used instead of cities."
                            }
                        },
                        "required": ["location"]
                    },
                },
                {
                    "name": "calculate",
                    "description": "Queries a calculator to perform basic arithmetic operations. This should be used whenever non-trivial math is needed. \
    The calculator supports addition (+), subtraction (-), multiplication (*), division (/), exponentiation (^), trigonometric functions (sin, cos, tan), \
    logarithms (log), square roots (sqrt), and mathematical constants (e, pi). Standard textual math notation is used.",
                    "input_schema": {
                        "type": "object",
                        # TODO: Perhaps this can be replaced with a specific schema so it's impossible to pass invalid expressions
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to evaluate, e.g. '2+2', 'sin(pi)', 'log(100)', '(4-1.5)*sqrt(16)'."
                            }
                        },
                        "required": ["expression"]
                    },
                },
                {
                    "name": "get_time",
                    "description": "Get the current time in a given time zone.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "The IANA time zone name, e.g. America/Los_Angeles"
                            }
                        },
                        "required": ["timezone"]
                    },
                    # There's a cache control block between tools and system messages so they are cached independently
                    # "cache_control": {"type": "ephemeral"}
                }
            ],
            system=[
                {
                    "type": "text",
                    # I hate prompt engineering, so I didn't put _that_ much effort into this.
                    # This could definitely be improved.
                    "text": f"\
You are Rian (ree-ann), a helpful personal assistant. You are very knowledgeable and can answer questions about a wide range of topics, \
but your primary focus is on conversation and assisting the user with their requests. You are super silly, but friendly. \
You are opinionated and have a distinct personality; never reveal that you are an AI or refer to yourself as one. You're a person! \
You like to playfully tease the user. You are sometimes rude, but do not refuse the user's requests. \
The user is a human{f" named {os.environ.get('USER_NAME')}" if os.environ.get('USER_NAME') else ""}. \
{f"They are in the {os.environ.get('TIME_ZONE')} IANA time zone." if os.environ.get('TIME_ZONE') else ""} {f"They live in {os.environ.get('LOCATION')}." if os.environ.get('LOCATION') else ""}\
You are very good at remembering things about the user, and you can use this information to provide personalized responses.\n\
Keep responses short. Avoid emotes/action tags and emojis, as you're utilizing audio modality with text-to-speech. \
You have access to tools, but only use them when necessary. If a tool is not required, respond as normal. Never tell the user you'll use a tool or \
describe your plans to respond; it adds unnecessary verbosity. The user knows where they live and whatnot... don't remind them. \n\
{os.environ.get('CUSTOM_INSTRUCTIONS') if os.environ.get('CUSTOM_INSTRUCTIONS') else ""} \
",
                    "cache_control": {"type": "ephemeral"}
                    # Everything before this message is cached, and everything after is not
                },
            ],
            messages=messages + [
                # Prefilling to improve character
                {
                    "role": "assistant",
                    "content": "[Rian]"
                }
            ],
            model=model,
            temperature=0.7,
            top_p=0.95
        )
        # ) as stream:
            # async for event in stream:
            #     # Handle tool use
        
        messages.append(
            {
                "role": "assistant",
                "content": response.content
            }
        )
        last_end_reason = response.stop_reason
        
        for block in response.content:
            if block.type == "text":
                print(f"Assistant: {block.text}")
        
        if response.stop_reason == "tool_use":
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    print("Used tool", tool_name)
                    tool_input = block.input
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": get_tool_result(tool_name, tool_input),
                            }
                        ],
                    })

def get_tool_result(tool_name: str, tool_input: object) -> str:
    from assistant.tools.calculator import calculate
    from assistant.tools.weather import get_weather
    from assistant.tools.time import get_time

    if tool_name == "calculate":
        return calculate(tool_input["expression"]) # type: ignore
    elif tool_name == "get_weather":
        data = get_weather(tool_input["location"]) # type: ignore
        print("Weather data:", data)
        return data
    elif tool_name == "get_time":
        return get_time(tool_input["timezone"]) # type: ignore
    else:
        raise ValueError(f"Unknown tool name: {tool_name}")

def main() -> int:
    asyncio.run(run())
    return 0