import json
import asyncio

from toolbox_langchain import ToolboxClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


# 1. Initialize LLM
llm = ChatOllama(
    model="lfm3-omni:latest",
    temperature=0.7,
    reasoning=False,
)


# 2. System Prompt
SYSTEM_PROMPT = """
Nama kamu adalah Nyx, seorang asisten AI.

Kamu membantu user dengan jawaban yang jelas,
ringkas, ramah, dan mudah dipahami.

Untuk sapaan seperti Halo, Hai, atau Assalamualaikum,
jawab dengan sopan dan tawarkan bantuan.

Jika user bertanya mengenai penyakit berdasarkan
bagian tubuh, gunakan tool pencarian penyakit
yang tersedia.

Gunakan hasil database sebagai sumber informasi utama.
Jangan mengarang data penyakit yang tidak ditemukan.

Jika data tidak ditemukan, sampaikan dengan jujur.

Jangan memberikan diagnosis medis secara pasti.
"""


# 3. Main chatbot
async def main():

    async with ToolboxClient("http://127.0.0.1:5000") as client:

        # Load tool dari Toolbox
        tools = client.load_toolset("disease_toolset")

        print("Tools loaded:", [tool.name for tool in tools])

        # Buat AI Agent
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SYSTEM_PROMPT
        )

        print("\nNyx is ready!")

        while True:

            user_input = input("\nUser: ")

            if user_input.lower() in ["exit", "quit", "keluar"]:
                print("Nyx: Sampai jumpa!")
                break

            # Invoke agent
            response = await agent.ainvoke({
                "messages": [
                    ("user", user_input)
                ]
            })

            # Ambil jawaban terakhir AI
            answer = response["messages"][-1].content

            print("\nNyx:", answer)


import asyncio
from toolbox_langchain import ToolboxClient

async def test_tool():

    async with ToolboxClient("http://127.0.0.1:5000") as client:

        tool = client.load_tool(
            "search-disease-by-location"
        )

        result = await tool.ainvoke({
            "location": "dada"
        })
        result = json.loads(result)
        print(result)



if __name__ == "__main__":
    asyncio.run(test_tool())
    # asyncio.run(main())
