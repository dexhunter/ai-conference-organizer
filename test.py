from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv()

YOUR_API_KEY = os.environ.get("PERPLEXITY_API_KEY")

messages = [
    {
        "role": "system",
        "content": (
            "You are an artificial intelligence assistant and you need to "
            "engage in a helpful, detailed, polite conversation with a user."
        ),
    },
    {
        "role": "user",
        "content": (
        """
        You are an ai conference organizer. Can you check if this pdf has a title slide? Is it all bullet points and does it have images? 
        Please reply in the json format 
        {
            "category": "ai_analaysis",
            "is_valid": boolean,
            "message": str
        }
        """
        ),
    },
]

client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")

# demo chat completion without streaming
response = client.chat.completions.create(
    model="mistral-7b-instruct",
    messages=messages,
)
# print(response)
print(response.choices[0].message.content)



#### not workign
      
        # file = client.files.create(
        # file=pdf_bytes,
        # purpose="assistants"
        # )

        # assistant = client.beta.assistants.create(
        # name="Conference organizer Assistant",
        # instructions="You are an expert conference organizer. Use you knowledge base to organize a conference about {topics}", topics=topics,
        # model="gpt-4o",
        # tools=[{"type": "file_search"}],
        # tool_resources={
        #     "file_search": {
        #     "file_ids": [file.id]
        #     }
        # }
        # )

        # # Create a thread and attach the file to the message
        # thread = client.beta.threads.create(
        # messages=[
        #     {
        #     "role": "user",
        #     "content": """
        #         Can you check if this pdf has a title slide? Is it all bullet points and does it have images? 
        #         Please reply in the json format 
        #         {
        #             "category": "ai_analaysis",
        #             "is_valid": boolean,
        #             "message": str
        #         }
        #     """,
        #     # Attach the new file to the message.
        #     "attachments": [
        #         { "file_id": file.id, "tools": [{"type": "file_search"}] }
        #     ],
        #     }
        # ]
        # )

        # run = client.beta.threads.runs.create_and_poll(
        #     thread_id=thread.id, assistant_id=assistant.id
        # )

        # messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

        # message_content = messages[0].content[0].text

        # return message_content.value