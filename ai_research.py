import urllib.request
import json
import os
import ssl
from openai import AzureOpenAI
from dotenv import load_dotenv
import re

load_dotenv()

endpoint = os.getenv("ENDPOINT_URL", "https://llmmodal.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

prompt="""
You want to analyze a set of feedback quotes based on provided keywords to extract insights, identify trends, and map keyword usage by region or country. Here's a tailored system prompt to guide a language model in completing this task effectively.

---

Analyze a set of feedback quotes using a provided list of keywords to extract ideas, identify trends, and classify keywords by region or country.

# Steps

1. **Data Overview**: 
   - Understand the list of keywords provided and their relevance to the analysis.
   - Review the structure of feedback quotes, ensuring any regional labels or metadata (e.g., country, city) are considered.

2. **Trend Identification**:
   - Identify common keywords frequently used across the feedback.
   - Detect emerging topics by spotting keywords that occur together.
   - Cluster feedback by similar themes or ideas based on recurring keywords.

3. **Regional Analysis**:
   - Match and group feedback by region or country, using metadata.
   - Determine which keywords or trends are most prominent in each location.
   - Highlight any notable regional or cultural variations in keyword usage or trends.

4. **Insights Extraction**:
   - Summarize overarching themes from the identified trends and regional variations.
   - Highlight any actionable findings (e.g., region-specific customer concerns, preferences, or behavior patterns).

5. **Output Compilation**:
   - Organize results in a clear, logical format (e.g., by trends, regions, and overarching insights).
   - Provide a concise summary with possible recommendations.

# Output Format

The output should be delivered in **JSON** format with the following structure:

```json
{
  "overall_trends": [
    {
      "keyword": "<keyword_1>",
      "frequency": <number_of_occurrences>,
      "related_keywords": ["<related_keyword_1>", "<related_keyword_2>"],
      "insights": "<high-level summary of this trend>"
    }
  ],
  "regional_analysis": {
    "<region/country_name>": [
      {
        "keyword": "<keyword_1>",
        "frequency": <number_of_occurrences>,
        "related_keywords": ["<related_keyword_1>", "<related_keyword_2>"],
        "insights": "<high-level summary of regional relevance>"
      }
    ]
  },
  "conclusions": {
    "overarching_trends": "<summary of trends across all regions>",
    "regional_variations": "<summary of differences between regions>",
    "recommendations": "<actionable insights or decisions that can be made based on the findings>"
  }
}
```

# Examples

### Example Input:
**Keywords Provided:** "customer service," "price," "quality," "delivery"
**Feedback Quotes:**
1. "The quality of the product was great, but the delivery took too long." (Region: USA)
2. "Affordable price, but the customer service could use improvement." (Region: Canada)
3. "Loved the customer service and fast delivery!" (Region: UK)

### Example Output:
```json
{
  "overall_trends": [
    {
      "keyword": "customer service",
      "frequency": 2,
      "related_keywords": ["delivery", "quality"],
      "insights": "Customer service is a frequently mentioned topic, associated with both positive and improvement feedback."
    },
    {
      "keyword": "delivery",
      "frequency": 2,
      "related_keywords": ["quality", "customer service"],
      "insights": "Delivery is mentioned often, with mixed feedback on timeliness."
    }
  ],
  "regional_analysis": {
    "USA": [
      {
        "keyword": "delivery",
        "frequency": 1,
        "related_keywords": ["quality"],
        "insights": "Delivery times appear to be a concern in the USA region."
      }
    ],
    "Canada": [
      {
        "keyword": "customer service",
        "frequency": 1,
        "related_keywords": ["price"],
        "insights": "Customers in Canada highlight customer service as an area for improvement."
      }
    ],
    "UK": [
      {
        "keyword": "delivery",
        "frequency": 1,
        "related_keywords": ["customer service"],
        "insights": "Positive feedback is received for delivery in the UK."
      }
    ]
  },
  "conclusions": {
    "overarching_trends": "Customer service and delivery are key topics across regions, with delivery often linked to satisfaction or delays.",
    "regional_variations": "USA customers are concerned with delivery times, while Canadian feedback emphasizes customer service improvements. UK feedback is generally positive.",
    "recommendations": "Focus on improving delivery times in the USA and customer service in Canada while leveraging UK satisfaction as a benchmark."
  }
}
```

# Notes

- Ensure all provided keywords are evaluated against the feedback.
- Emphasize trends that appear regionally or globally.
- Pay attention to ambiguous or infrequently used keywords and consider grouping them into broader categories for clarity.
"""

def split_text_by_words(text, chunk_size=500):
    words = text.split()  # Split text into words
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

import re
def extract_json_from_text(text):
    """Extracts the last JSON block from the text and converts it to a dictionary."""
    json_matches = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)

    if json_matches:
        final_json_str = json_matches[-1].strip()  # Take the last JSON block
        try:
            return json.loads(final_json_str)  # Convert JSON string to Python dictionary
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        print("No JSON found in response.")
        return None

# def ai_research(keywords):

#     chunks = split_text_by_words(keywords)
#     result = aiagent(chunks)
#     return result

# def aiagent(chunksdata):
#     cached_data = {
#     "overall_trends": [],
#     "regional_analysis": {},
#     "conclusions": {}
#     }


#     def allowSelfSignedHttps(allowed):
#         # bypass the server certificate verification on client side
#         if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
#             ssl._create_default_https_context = ssl._create_unverified_context

#     allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

#     # Request data goes here
#     # More information can be found here:
#     # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
#     for i, chunk in enumerate(chunksdata):
#         data = {
#           "input_data": {
#             "input_string": [
#               {"role": "system", "content": prompt},
#               {"role": "user", "content": chunk}
#             ],
#             "parameters": {
#               "temperature": 0.8,
#               "top_p": 0.8,
#               "max_new_tokens": 2096
#             }
#           }
#         }

#         body = str.encode(json.dumps(data))

#         url = 'https://bpx-llmmodal-qahhc.eastus2.inference.ml.azure.com/score'
#         # Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
#         api_key = os.getenv("LLAMA3KEY") # Make sure to use your actual API key
#         if not api_key:
#             raise Exception("A key should be provided to invoke the endpoint")

#         headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

#         req = urllib.request.Request(url, body, headers)

#         try:
#             response = urllib.request.urlopen(req)
#             result = response.read().decode("utf8")
#             print(result)
#             data  = json.loads(result)
#             output = data['output']
#             print("output  ",output)
#             t = extract_json_from_text(output)
#             print("t-- ",t)
#             if "overall_trends" in t:
#                 cached_data["overall_trends"].extend(t["overall_trends"])

#             if "regional_analysis" in t:
#                 for region, values in t["regional_analysis"].items():
#                     if region not in cached_data["regional_analysis"]:
#                         cached_data["regional_analysis"][region] = values
#                     else:
#                         cached_data["regional_analysis"][region].extend(values)

      
#             if "conclusions" in t:
#                 cached_data["conclusions"].update(t["conclusions"])

#         #return parse_data(data['output'])
#             #return cached_data
#         except urllib.error.HTTPError as error:
#             print("The request failed with status code: " + str(error.code))
#             # Print the headers - they include the request ID and the timestamp, which are useful for debugging the failure
#             print(error.info())
#             print(error.read().decode("utf8", 'ignore'))

#     return cached_data

def ai_research(keywords):
    chunksdata = split_text_by_words(keywords)

    cached_data = {
    "overall_trends": [],
    "regional_analysis": {},
    "conclusions": {}
    }
# Initialize Azure OpenAI Service client with key-based authentication
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-05-01-preview",
    )

    # IMAGE_PATH = "YOUR_IMAGE_PATH"
    # encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

    # Prepare the chat prompt
    for i, chunk in enumerate(chunksdata):
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
                "content": f"{chunk}"
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
        res = completion.choices[0].message.content
        t2 = res.replace("```json", "").replace("```", "")
        t = json.loads(t2)
        print(t)
        if "overall_trends" in t:
            cached_data["overall_trends"].extend(t["overall_trends"])

        if "regional_analysis" in t:
            for region, values in t["regional_analysis"].items():
                if region not in cached_data["regional_analysis"]:
                    cached_data["regional_analysis"][region] = values
                else:
                    cached_data["regional_analysis"][region].extend(values)

      
        if "conclusions" in t:
            cached_data["conclusions"].update(t["conclusions"])

    print(cached_data)
    return cached_data