# import urllib.request
# import json
# import os
# import ssl
# from dotenv import load_dotenv

# load_dotenv()


# prompt="""
# You are an AI assistant tasked with analyzing customer feedback. Your job is to Extract keywords, relationships, synonyms, sentiments, and details related to safety, diagnosis, and provide the output in JSON format."
# **Output JSON:**

# ```json
# {
#   "result": [
#     {
#       "keywords": ["community Summit", "pioneers", "patients", "caregivers", "Eisai", "community leaders"],
#       "Theme": ["pioneering healthcare initiatives"],
#       "safety": [],
#       "diagnose": [],
#       "treatment": [],
#       "synonyms": ["gathering", "initiators", "wellbeing seekers", "caretakers", "company", "leaders"],
#       "sentiment": "8",
#       "nodes": [
#         { "id": "pioneers", "group": 1, "label": "pioneers" },
#         { "id": "community Summit", "group": 1, "label": "community Summit" },
#         { "id": "Eisai", "group": 2, "label": "Eisai" },
#         { "id": "community leaders", "group": 2, "label": "community leaders" }
#       ],
#       "links": [
#         { "source": "pioneers", "target": "community Summit", "relationship": "hosts" },
#         { "source": "community Summit", "target": "Eisai", "relationship": "initiated by" },
#         { "source": "Eisai", "target": "community leaders", "relationship": "invited by" }
#       ],
#       "AnalyzeThoroughly": "The statement emphasizes leadership in healthcare initiatives, hinting at active future community involvement.",
#       "THEME": "Pioneering community healthcare engagement",
#       "ISSUE": "Delayed action in community healthcare initiatives"
#     }
#   ]
# }
# ```
# """

# def chat(user):
#     def allowSelfSignedHttps(allowed):
#         # bypass the server certificate verification on client side
#         if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
#             ssl._create_default_https_context = ssl._create_unverified_context

#     allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

#     # Request data goes here
#     # More information can be found here:
#     # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
#     data = {
#       "input_data": {
#         "input_string": [
#           {"role": "system", "content": prompt},
#           {"role": "user", "content": user}
#         ],
#         "parameters": {
#           "temperature": 0.8,
#           "top_p": 0.8,
#           "max_new_tokens": 2096
#         }
#       }
#     }

#     body = str.encode(json.dumps(data))

#     url = 'https://bpx-llmmodal-qahhc.eastus2.inference.ml.azure.com/score'
#     # Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
#     api_key = os.getenv("LLAMA3KEY") # Make sure to use your actual API key
#     #print(api_key)
#     if not api_key:
#         raise Exception("A key should be provided to invoke the endpoint")

#     headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

#     req = urllib.request.Request(url, body, headers)

#     try:
#         response = urllib.request.urlopen(req)
#         result = response.read().decode("utf8")
#         #print(json.loads(result))
#         print("response generated")
#         return result
#     except urllib.error.HTTPError as error:
#         print("The request failed with status code: " + str(error.code))

#         # Print the headers - they include the request ID and the timestamp, which are useful for debugging the failure
#         print(error.info())
#         print(error.read().decode("utf8", 'ignore'))
from dotenv import load_dotenv
import os
import json 
load_dotenv()

key = os.getenv("LLMURLKEY")
endpoint = os.getenv("LLMURL")

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def chat(text,sno):
    user_message = f"Context:{text}"
#     system_message = """
#     You are an AI assistant tasked with analyzing customer feedback. Your job is to Extract keywords, relationships, synonyms, sentiments, and details related to safety, diagnosis, and provide the output in JSON format."
#     **Output JSON:**

# ```json
# {
#   "result": [
#     {
#       "keywords": ["community Summit", "pioneers", "patients", "caregivers", "Eisai", "community leaders"],
#       "Theme": ["pioneering healthcare initiatives"],
#       "safety": [],
#       "diagnose": [],
#       "treatment": [],
#       "synonyms": ["gathering", "initiators", "wellbeing seekers", "caretakers", "company", "leaders"],
#       "sentiment": "8",
#       "nodes": [
#         { "id": "pioneers", "group": 1, "label": "pioneers" },
#         { "id": "community Summit", "group": 1, "label": "community Summit" },
#         { "id": "Eisai", "group": 2, "label": "Eisai" },
#         { "id": "community leaders", "group": 2, "label": "community leaders" }
#       ],
#       "links": [
#         { "source": "pioneers", "target": "community Summit", "relationship": "hosts" },
#         { "source": "community Summit", "target": "Eisai", "relationship": "initiated by" },
#         { "source": "Eisai", "target": "community leaders", "relationship": "invited by" }
#       ],
#       "AnalyzeThoroughly": "The statement emphasizes leadership in healthcare initiatives, hinting at active future community involvement.",
#       "THEME": "Pioneering community healthcare engagement",
#       "ISSUE": "Delayed action in community healthcare initiatives"
#     }
#   ]
# }
# ```
#"""

    system_message="""
      Analyze customer feedback to extract keywords, relationships, synonyms, sentiments, and details specifically related to safety, diagnosis, and treatment. Output the analysis in JSON format.

# Details
1. Identify and extract the following:
   - **Keywords**: Significant terms or phrases that represent key ideas.
   - **Relationships**: Connections or interactions between entities, expressed as "source-target-relationship."
   - **Synonyms**: Alternative words or phrases for identified keywords.
   - **Safety**: Information or concerns specifically related to safety.
   - **Diagnosis**: Information or concerns specifically related to diagnosis.
   - **Treatment**: Information or concerns specifically related to various treatment aspects.

2. Conduct sentiment analysis:
   - Assign a sentiment score ranging from 1 (negative) to 10 (positive) to assess the overall sentiment of the feedback.

3. Create a network of **nodes** (representing entities) and **links** (representing relationships) to visualize the connections between the mentioned keywords.

4. Provide a detailed analysis including:
   - An overarching **theme** summarizing the core idea.
   - Identification of specific **issues** mentioned.
   - Analysis text that captures the deeper interpretation of the feedback.

5. - output return as JSON format."

# Output Format
- The output should be structured in JSON format as follows:
  ```json
  {
    "result": [
      {
        "keywords": [/* Extracted keywords */],
        "Theme": [/* Overarching theme(s) */],
        "safety": [/* Safety-related details */],
        "diagnose": [/* Diagnosis-related details */],
        "treatment": [/* Treatment-related details */],
        "synonyms": [/* Synonyms for keywords */],
        "sentiment": "/* Sentiment score */",
        "nodes": [
          { "id": "/* Entity name */", "group": /* Numerical group */, "label": "/* Entity label */" }
        ],
        "links": [
          { "source": "/* Source entity */", "target": "/* Target entity */", "relationship": "/* Relationship description */" }
        ],
        "AnalyzeThoroughly": "/* Deep interpretation summary */",
        "THEME": "/* Overarching theme */",
        "ISSUE": "/* Issues identified */"
      }
    ]
  }
  ```

# Examples

### Example 1
**Input:**
"The patient community summit brought together patients, caregivers, and community leaders to discuss innovative approaches toward diagnosis and treatment."

**Output:**
```json
{
  "result": [
    {
      "keywords": ["patient community summit", "patients", "caregivers", "community leaders", "diagnosis", "treatment"],
      "Theme": ["Collaborative healthcare initiatives"],
      "safety": [],
      "diagnose": ["innovative approaches toward diagnosis"],
      "treatment": ["innovative approaches toward treatment"],
      "synonyms": ["conference", "participants", "helpers", "leaders", "medical processes"],
      "sentiment": "9",
      "nodes": [
        { "id": "patients", "group": 1, "label": "patients" },
        { "id": "caregivers", "group": 1, "label": "caregivers" },
        { "id": "community leaders", "group": 2, "label": "community leaders" },
        { "id": "diagnosis", "group": 3, "label": "diagnosis" },
        { "id": "treatment", "group": 3, "label": "treatment" }
      ],
      "links": [
        { "source": "patient community summit", "target": "patients", "relationship": "engaged" },
        { "source": "patient community summit", "target": "caregivers", "relationship": "hosted" },
        { "source": "patients", "target": "diagnosis", "relationship": "related to" },
        { "source": "caregivers", "target": "treatment", "relationship": "support for" }
      ],
      "AnalyzeThoroughly": "The feedback highlights collaborative discussions focusing on innovation in healthcare, fostering patient and caregiver cooperation.",
      "THEME": "Innovation in collaboration for diagnosis and treatment",
      "ISSUE": null
    }
  ]
}
```

### Example 2
**Input:**  
"There was concern about the lack of safety measures during the clinical trial phases."

**Output:**  
```json
{
  "result": [
    {
      "keywords": ["safety", "clinical trial phases"],
      "Theme": ["Safety concerns in clinical trials"],
      "safety": ["lack of safety measures"],
      "diagnose": [],
      "treatment": [],
      "synonyms": ["protection", "stages of experimentation"],
      "sentiment": "3",
      "nodes": [
        { "id": "safety", "group": 1, "label": "safety" },
        { "id": "clinical trial phases", "group": 2, "label": "clinical trial phases" }
      ],
      "links": [
        { "source": "safety", "target": "clinical trial phases", "relationship": "lacking in" }
      ],
      "AnalyzeThoroughly": "The statement strongly emphasizes missing safety protocols as a critical issue during trial experiments.",
      "THEME": "Addressing safety in experimental trials",
      "ISSUE": "Gaps in safety measures"
    }
  ]
}
```

### Notes
- Ensure to capture detailed information explicitly related to safety, diagnosis, or treatment as priorities.
- For each input, provide a complete and concise analysis that follows the JSON structure while maintaining logical consistency and coherence.
"""
    response = client.complete(
        messages=[
            SystemMessage(content=system_message),
            UserMessage(content=user_message),
        ]
    )
    result = response.choices[0].message.content
    print(result)
    return result
    # r = result.replace("```json\n", '').replace("'", '').replace("```", "")
    # try:
    #     print(r)
    #     json_data = json.loads(r)
    #     processed_data=[]
    #     data = json_data.get('result', [])
    #     if not data:
    #       return []  # Skip if result is empty


    #     first_entry = data[0]
    #     return first_entry
        # processed_data.append({
        #             "sno": sno,  # Access sno directly now
        #             "metadata": data,
        #             "keywords": first_entry.get('keywords', []),
        #             "theme": first_entry.get("Theme", ""),
        #             "theme_statement": first_entry.get("THEME", ""),
        #             "analyze_thoroughly": first_entry.get("AnalyzeThoroughly", ""),
        #             "issue": first_entry.get("ISSUE", ""),
        #             "safety": first_entry.get("safety", ""),
        #             "diagnose": first_entry.get("diagnose", ""),
        #             "treatment": first_entry.get("treatment", ""),
        #             "synonyms": first_entry.get('synonyms', []),
        #             "sentiment": first_entry.get("sentiment", ""),
        #             "nodeid": [node.get('id', '') for node in first_entry.get("nodes", [])],
        #             "group": [node.get('group', '') for node in first_entry.get("nodes", [])],
        #             "label": [node.get('label', '') for node in first_entry.get("nodes", [])],
        #             "source": [link.get('source', '') for link in first_entry.get("links", [])],
        #             "target": [link.get('target', '') for link in first_entry.get("links", [])],
        #             "relationship": [link.get('relationship', '') for link in first_entry.get("links", [])],
        #         })
        # print(result)
        # return processed_data  # Return list for consistency

    # except Exception as e:
    #     print(f"Error during chat processing: {e}")
    #     return []

# import re
# import json  
# txt= "I wanted to be a minister in the community, not just at the pulpit; ADOS started 20 years ago with a $300k grant to reach people of color."
# result = chat(txt,1)
# print(result)
# r = result.replace("```json\n",'').replace("'",'').replace("```","")
# json_data = json.loads(r)
# print(json_data)
#processed_data=[]

# data = json_data.get('result')
# first_entry = data[0]
# # keywords = 
# processed_data.append({
#       "sno": 1,
#       "metadata": "Test",
#       "keywords2": "test",
#       "keywords": first_entry['keywords'],
#       "theme": first_entry.get("Theme", ""),
#       "theme_statement": first_entry.get("THEME", ""),
#       "analyze_througly": first_entry.get("AnalyzeThoroughly", ""),
#       "issue": first_entry.get("ISSUE", ""),
#       "safety": first_entry.get("safety", ""),
#       "diagnose": first_entry.get("diagnose", ""),
#       "treatment": first_entry.get("treatment", ""),
#       "synonyms": first_entry['synonyms'],
#       "sentiment": first_entry.get("sentiment", ""),
#       "nodeid": [node.get('id', '') for node in first_entry.get("nodes", [])],
#       "group": [node.get('group', '') for node in first_entry.get("nodes", [])],
#       "label": [node.get('label', '') for node in first_entry.get("nodes", [])],
#       "source": [link.get('source', '') for link in first_entry.get("links", [])],
#       "target": [link.get('target', '') for link in first_entry.get("links", [])],
#       "relationship": [link.get('relationship', '') for link in first_entry.get("links", [])],
#   })

# print(processed_data)