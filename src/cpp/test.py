import asyncio
import json

import aiohttp
import requests

url = "http://localhost:8080/tts"

payload = [json.dumps({
    "modelPath": "../models/Kristin.onnx",
    "outputPath": "../audio/",
    "output_file": f"output_file_{i}",
    "sentence": f"Test sentence {i}",
    # "outputType": "OUTPUT_RAW"
}) for i in range(30)]
headers = {
    'Content-Type': 'application/json'
}

async def make_request(session, element):
    try:
        async with session.post(url, data=payload[element], headers=headers) as response:
            # Optionally, log or process the response
            print(f"Request sent with status: {response.status} - {element}")
    except Exception as e:
        print(f"An error occurred: {e}")

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, i) for i in range(30)]
        # Await tasks to ensure they complete before closing the session
        await asyncio.gather(*tasks)

# Run the event loop
asyncio.run(main())


# call the API 10 times in a row with requests
# for i in range(10):
#     response = requests.request("POST", url, headers=headers, data=payload)
#     print(f"Request sent with status: {response.status_code}")
