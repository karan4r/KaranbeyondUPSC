import asyncio
import json
import sys
from main import app, QueryRequest, chat_endpoint

async def test():
    req = QueryRequest(query="I want to prepare for UPSC along with college, suggest a course")
    res = await chat_endpoint(req)
    print("-----ANSWER-----")
    print(res["answer"])
    print("----------------")

if __name__ == "__main__":
    asyncio.run(test())
