import urllib.request
import json
import ssl
import os
import base64
import threading
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

endpoint = os.getenv("ENDPOINT_URL", "https://llmmodal.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

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

def classifykeywords(user_input, results):
    prompt="""
Determine trending topics in the market based on a provided list of input keywords, classify the keywords by region, and provide a relevance score for each keyword.

---

**Task Explanation**

You are a research AI agent tasked with analyzing a list of input keywords to identify trending topics in the market. Your tasks include:

1. **Research Trends:** Identify which topics/trends the keywords belong to based on online search popularity, media mentions, or other indicators.
2. **Regional Classification:** Classify each keyword by region(s) where it is trending.
3. **Scoring:** Assign a relevance score (between 0–100) for each keyword, where the score represents how strong or relevant the trend is for that keyword in the identified region(s).

---

### Steps

1. Parse the provided list of keywords.
2. Research relevant trending topics and align them with the input keywords.
3. Identify which regions the keywords are trending in. Mention specific regions (e.g., "North America", "Europe", "Asia-Pacific") or countries if precise data is identifiable.
4. Assign a relevance score (0–100) to each keyword indicating strength of the trend. Scores closer to 100 indicate highly trending keywords.
5. Return a structured output per keyword, including:
    - The keyword
    - Its associated trend or topic
    - The trending region(s)
    - The score

---

### Output Format

The output should be a structured JSON object where each keyword includes the following attributes:
- `keyword`: The input keyword being analyzed.
- `trend`: The topic or trend related to the keyword.
- `regions`: A list of regions or countries where the keyword is trending.
- `score`: A relevance score (0–100) indicating the strength of the trend.

Example output:

```json
[
  {
    "keyword": "AI in healthcare",
    "trend": "Artificial intelligence applications in health",
    "regions": ["North America", "Europe"],
    "score": 85
  },
  {
    "keyword": "EV batteries",
    "trend": "Electric vehicle advancements",
    "regions": ["Asia-Pacific", "Europe"],
    "score": 78
  },
  {
    "keyword": "Sustainable fashion",
    "trend": "Eco-friendly consumer goods",
    "regions": ["North America", "Europe"],
    "score": 90
  }
]
```

---

### Notes

- Be as specific as possible regarding trends and regions but avoid adding speculative or unsupported claims.
- If specific data for a region or trend is unavailable, clearly indicate this with a placeholder (e.g., `"regions": ["Unknown"]`).
- Ensure scores are aligned with general popularity metrics (e.g., search volume, social media trends) whenever applicable.
- If certain keywords are not trending or data is unavailable, you may still include them with `"score": 0`. 

--- 

### Assumptions

- Keywords are provided as input, and you will research the trends based on current market and industry data up to the most recent time.
    """
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
                "content": f"{user_input}"
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
    data2 = completion.choices[0].message.content
    print(data2)
    cleaned_data =data2.replace("```json", "").replace("```", "")
    # Parse the cleaned JSON data
    data = json.loads(cleaned_data)

    results["classify"] = data
    # def allowSelfSignedHttps(allowed):
    #     # bypass the server certificate verification on client side
    #     if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    #         ssl._create_default_https_context = ssl._create_unverified_context

    # allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

    # # Request data goes here
    # # More information can be found here:
    # # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
    # data = {
    #   "input_data": {
    #     "input_string": [
    #       {"role": "system", "content": prompt},
    #       {"role": "user", "content": user_input}
    #     ],
    #     "parameters": {
    #       "temperature": 0.2,
    #       "top_p": 0.8,
    #       "max_new_tokens": 2096
    #     }
    #   }
    # }

    # body = str.encode(json.dumps(data))

    # url = 'https://bpx-llmmodal-qahhc.eastus2.inference.ml.azure.com/score'
    # # Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
    # api_key = os.getenv("LLAMA3KEY")  # Make sure to use your actual API key
    # #print(api_key)
    # if not api_key:
    #     raise Exception("A key should be provided to invoke the endpoint")

    # headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    # req = urllib.request.Request(url, body, headers)

    # try:
    #     response = urllib.request.urlopen(req)
    #     result = response.read().decode("utf8")

    #     try:
    #         data = json.loads(result)
    #         output = data['output']
    #         print(output)
    #         classify_data = extract_json_from_text(output)

    #         if classify_data is None:
    #             print("Warning: classify_data is None")
    #         else:
    #             print("Extracted Classification Data:", classify_data)

    #         results["classify"] = classify_data

    #     except json.JSONDecodeError as e:
    #         print(f"Error decoding API response JSON: {e}")

    # except urllib.error.HTTPError as error:
    #     print("The request failed with status code:", error.code)
    #     print(error.info())
    #     print(error.read().decode("utf8", 'ignore'))
#################################################################
def findtrend_topic_basedonRegion(user_input, results):



    #print(subscription_key)

    prompt= """
   You are a market research agent specializing in identifying trending topics. Using the input list of keywords, your task is to find the trending topics associated with those keywords, specify the regions where these trends are prominent, and provide key insights on how these trends can impact the market. Prioritize providing actionable and data-driven insights.

## Instructions:

1. Accept a list of keywords as input.
2. Analyze and identify trending topics related to these keywords. Contextualize the trends when possible.
3. Specify the regions where these trends are most prominent based on available data or known patterns.
4. Provide insights into how these trends can impact the market, focusing on opportunities, challenges, and potential shifts in consumer behavior.
5. Be concise and prioritize relevance in your analysis.

## Steps:

1. Identify input keywords from the user.
2. Cross-reference keywords with currently trending data (e.g., social media, search volumes, and other market indicators).
3. Gather regional insights, ensuring the specified regions are clearly associated with each trend.
4. Analyze the market impact:
   - How the trend may create new opportunities for businesses.
   - Potential challenges the trend may introduce.
   - Possible shifts in consumer behavior as a result of these trends.

## Output Format:

The output should be presented in the following structure:

### 1. Keywords Analyzed:
[List the input keywords used.]

### 2. Trending Topics and Region:
Provide the list of trending topics with their associated regions as a sub-point. Example format:
- **Trend Name**:
    - **Region:** [Region where the trend is prominent]
    - **Trend Insight:** [Brief explanation of the trend and its significance]

### 3. Market Impact Analysis:
Provide a detailed breakdown of the market impact for each trend. Example format:
- **Trend Name**:
    - **Opportunities:** [How this affects businesses and opens new opportunities]
    - **Challenges:** [Challenges this trend may create]
    - **Consumer Behavior Shift:** [Potential shifts in audience behavior]

### 4. Summary of Insights:
Conclude with a brief summary of how the identified trends, overall, could affect the broader market.

## Examples:

### Example Input:
**Keywords Analyzed:** ["Electric Vehicles", "Sustainable Fashion", "Artificial Intelligence"]

### Example Output:

#### 1. Keywords Analyzed:
- Electric Vehicles
- Sustainable Fashion
- Artificial Intelligence

#### 2. Trending Topics and Region:
- **Adoption of EVs in Urban Areas**:
    - **Region:** North America, Europe
    - **Trend Insight:** Growing interest in electric vehicles driven by environmental concerns and government subsidies.
- **Eco-Friendly Materials in Clothing**:
    - **Region:** Asia-Pacific, North America
    - **Trend Insight:** Rise in sustainable fashion with an emphasis on circular economy and biodegradable materials.
- **Generative AI in Marketing**:
    - **Region:** Global
    - **Trend Insight:** Surging adoption of AI tools to automate and personalize marketing strategies.

#### 3. Market Impact Analysis:
- **Adoption of EVs in Urban Areas**:
    - **Opportunities:** Increased demand for EV infrastructure, such as charging stations and battery technology.
    - **Challenges:** High entry costs for smaller market players. Infrastructure rollout may lag behind demand.
    - **Consumer Behavior Shift:** Growing preference for eco-friendly vehicles and reduced reliance on traditional combustion engines.

- **Eco-Friendly Materials in Clothing**:
    - **Opportunities:** New market for sustainable textiles and increased brand loyalty for eco-conscious companies.
    - **Challenges:** Higher production costs, scaling challenges for smaller businesses.
    - **Consumer Behavior Shift:** Consumers prioritizing sustainability over fast fashion, leading to slower consumption cycles.

- **Generative AI in Marketing**:
    - **Opportunities:** Cost efficiency, increased customer engagement, and streamlined campaigns.
    - **Challenges:** Data privacy concerns and difficulty staying unique in an AI-saturated market.
    - **Consumer Behavior Shift:** Expectations for hyper-personalized experiences from brands.

#### 4. Summary of Insights:
The identified trends highlight significant shifts toward sustainability, technological advancement, and personalization. Businesses should adapt by investing in innovation, aligning with eco-conscious movements, and leveraging AI to stay competitive in a rapidly changing market.

## Notes:
- Focus on providing trends that are actionable and relevant to the specified keywords.
- If no trends are found for specific keywords, mention this explicitly and suggest potential areas the user may explore further.
- Avoid vague insights; ensure points are grounded in market realities.

    """
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
                "content": f"{user_input}"
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
    results["trends"] = data
    #print(data)
    #return data

##########################################
##########################################

def trend_classify(user_input):
    """
    Run both classifykeywords and findtrend_topic_basedonRegion concurrently.
    """
    results = {}

    # Define threads for concurrent execution
    threads = [
        threading.Thread(target=classifykeywords, args=(user_input, results)),
        threading.Thread(target=findtrend_topic_basedonRegion, args=(user_input, results)),
    ]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print(results)
    return results