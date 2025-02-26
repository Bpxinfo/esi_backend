from dotenv import load_dotenv
import os
load_dotenv()

key = os.getenv("phi_key")
endpoint = os.getenv("phi_endpoint")

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def chat_phi(text):
    user_message = f"Context:{text}"
    system_message = """
            Extract key phrases and concepts from the provided text, focusing on nouns, noun phrases, and significant action verbs that convey core ideas.

            Guidelines:
            Include key actions and their objects (e.g., "screen patients," "assess outcomes").
            Focus on meaningful terms such as processes, challenges, goals, or outcomes.
            Exclude filler words, full sentences, and generic terms.
            Retain context-specific phrases that highlight main ideas.
            
            Output Format:
            Provide the extracted key phrases as a comma-separated list.

            Examples:

            Input:
            A new initiative for the Health Systems Team is working upstream in Primary Care - help Primary Care screen, assess, even diagnose - and only send patients who are 'outside the box' to the Cognitive Specialists. We are encouraging reimbursement thru GUIDE. The biggest challenge we are hearing with LEQ is PET being covered. This has been a barrier when access to CSF is a challenge.

            Output:
            screen, assess, diagnose, send patients, outside the box, cognitive specialists, GUIDE, reimbursement, LEQ, PET coverage, lack of PET coverage, barrier, access to CSF, challenge

    """

    response = client.complete(
        messages=[
            SystemMessage(content=system_message),
            UserMessage(content=user_message),
        ]
    )
    output_triple = response.choices[0].message.content
    print(output_triple)
    return output_triple