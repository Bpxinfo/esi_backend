import urllib.request
import json
import os
import ssl
from dotenv import load_dotenv

load_dotenv()



def extract_output(json_response):
    """
    Extract the 'output' field from a JSON string.

    Args:
        json_response (str): A JSON-formatted string containing the response.

    Returns:
        str: The extracted output field or an error message if the field is missing.
    """
    try:
        data = json.loads(json_response)  # Parse the JSON string
        return data.get("output", "Output field not found in the response.")  # Extract 'output'
    except json.JSONDecodeError as e:
        return f"Invalid JSON format: {e}"

prompt="""
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

def chat(user):
    def allowSelfSignedHttps(allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context

    allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

    # Request data goes here
    # More information can be found here:
    # https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
    data = {
      "input_data": {
        "input_string": [
          {"role": "system", "content": prompt},
          {"role": "user", "content": user}
        ],
        "parameters": {
          "temperature": 0.4,
          "top_p": 0.8,
          "max_new_tokens": 2096
        }
      }
    }

    body = str.encode(json.dumps(data))

    #url = 'https://bpx-llmmodal-qahhc.eastus2.inference.ml.azure.com/score'
    # Replace this with the primary/secondary key, AMLToken, or Microsoft Entra ID token for the endpoint
    url = os.getenv("LLMURL")
    api_key = os.getenv("LLMURKKEY") # Make sure to use your actual API key
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    print("API URL -----",url)
    print("API KEY ------",api_key)
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf8")
        #print(json.loads(result))
        output = extract_output(result)
        return output
    
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the request ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

