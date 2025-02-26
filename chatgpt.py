import os
import base64
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv()

endpoint = os.getenv("ENDPOINT_URL", "https://llmmodal.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

prompt= """
Extract the key phrases and concepts from the given text strings, focusing on nouns, noun phrases, and significant action verbs that represent important ideas or core themes.

### Guidelines for Extraction:
- Include key actions (verbs) and associated objects (nouns) or significant segments of the text.
- Focus on meaningful fragments such as processes, challenges, goals, outcomes, or constraints.
- Exclude filler words, entire sentences, and overly generic terms.
- Retain context-specific phrases that contribute to understanding the main ideas.

# Steps

1. Identify key actions or activities and their related objects (e.g., "screen patients," "assess outcomes").
2. Extract specific terms or challenges referenced (e.g., "lack of PET coverage," "reimbursement issues").
3. Isolate important nouns, phrases, or specific terminology mentioned (e.g., "GUIDE," "cognitive specialists").
4. Filter out non-essential words (e.g., conjunctions, prepositions, or explanatory content).
   
# Output Format
Provide the extracted key phrases as a **comma-separated list**.

# Examples

### Example 1:
**Input:**  
A new initiative for the Health Systems Team is working upstream in Primary Care - help Primary Care screen, assess, even diagnose - and only send patients who are 'outside the box' to the Cognitive Specialists. We are encouraging reimbursement thru GUIDE. Now there is an incentive with Primary Care due to a treatment being available. The biggest challenge we are hearing with LEQ is PET being covered. This has been a barrier when access to CSF is a challenge.  

**Output:**  
screen, assess, diagnose, send patients, outside the box, cognitive specialists, GUIDE, reimbursement, LEQ, PET coverage, lack of PET coverage, barrier, access to CSF, challenge  

---

### Example 2:  
**Input:**  
We need to figure out how to capture the revenue for the in-between work. Referencing the steps required to start and monitor therapy.  

**Output:**  
revenue, in-between work, start and monitor therapy

# Notes
- Focus on extracting meaningful terms and avoid noise.
- Ensure the extracted phrases sufficiently summarize the text content.  

"""
def chatwith_openai(userinput):
# Initialize Azure OpenAI Service client with key-based authentication
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-05-01-preview",
    )

    # IMAGE_PATH = "YOUR_IMAGE_PATH"
    # encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

    # Prepare the chat prompt
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        },
        {
            "role": "user",
            "content": f"{userinput}"
        }
    ]

    # Include speech result if speech is enabled
    messages = chat_prompt

    # Generate the completion
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=1000,
        temperature=0.4,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    # Print the AI response
    # print(completion.to_json())
    # print(completion.choices[0].message.content)
    data = completion.choices[0].message.content
    return data
