from flask import Flask, request, jsonify, render_template
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from flask_cors import CORS
import spacy
from llama31  import chat
from concurrent.futures import ThreadPoolExecutor
from azure_table import *
from reinforcement_learn import *
from azure_blob import *
from ai_research import *
from trend_topic import *
from chatai import *
import logging
import traceback
# from chatmodal import sample_chat_completions

# ThreadPoolExecutor for managing threads
executor = ThreadPoolExecutor(max_workers=5)
task_status = {}

# Helper function to handle task errors
def task_wrapper(func, df, column_names, metadata, name):
    try:
        func(df, column_names, metadata, name)
        task_status[name] = "completed"  # Update status when done
    except Exception as e:
        task_status[name] = f"error: {str(e)}"

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize global variables
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize global transformers
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
cv = CountVectorizer(max_features=6000, ngram_range=(1, 2))

# Helper Functions
def preprocess_text(txt):
    """Preprocess text: lowercasing, tokenization, stopword removal, lemmatization."""
    if not isinstance(txt, str) or not txt.strip():
        return ""
    
    txt = txt.lower()
    txt = re.sub(r"<.*?>", " ", txt)  # Remove HTML tags
    txt = re.sub(r"[^a-zA-Z]", " ", txt)  # Remove special characters and digits
    tokens = nltk.word_tokenize(txt)
    tokens = [word for word in tokens if word not in stop_words and len(word) >= 3]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def extract_entities(text):
    """Extract named entities using spaCy."""
    doc = nlp(text)
    entities = {"text": [ent.text for ent in doc.ents], "labels": [ent.label_ for ent in doc.ents]}
    return entities

def get_sentiment(feedback):
    """Analyze sentiment using TextBlob."""
    analysis = TextBlob(feedback)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    return "Neutral"

def sort_coo(coo_matrix):
    """Sort COO matrix by score."""
    return sorted(zip(coo_matrix.col, coo_matrix.data), key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """Extract top n keywords from a TF-IDF vector."""
    sorted_items = sorted_items[:topn]
    results = {feature_names[idx]: round(score, 3) for idx, score in sorted_items}
    return results

@app.route('/')
def health():
    print("Working Fine")  # Debug message printed in the console
    return "Working Fine!"  # Response sent to the client
#############################################
# @app.route('/cluster2/<string:table_id>', methods=['GET'])
# def cluster2(table_id):
#     """
#     Route to fetch table data for a given table ID.
#     """
#     #test = "Region"
#     all_tablenames = getAllGraphName()  
#     #print(table_id)
#     if table_id in all_tablenames:
#         data = get_cluster2_data(table_id)  
#         return jsonify({'data': data}), 200
#     else:
#         # Return 404 if table name not found
#         return jsonify({'error': "Table not found"}), 404
##################################################
@app.route('/networkgraphData/<string:table_id>', methods=['GET'])
def networkgraphData(table_id):
    """
    Route to fetch table data for a given table ID.
    """
    all_tablenames = getAllGraphName() 

    #print(table_id)
    data = get_networkgraphdata_by_id(table_id)  
    return jsonify({'data': data}), 200
    # if table_id in all_tablenames:
    #     # data = get_networkgraphdata_by_id(table_id)  
    #     # return jsonify({'data': data}), 200
    # else:
    #     # Return 404 if table name not found
    #     return jsonify({'error': "Table not found"}), 404

  ####################################################
@app.route('/wordclusterData/<string:table_id>', methods=['GET'])
def wordclusterData(table_id):
    """
    Route to fetch table data for a given table ID.
    """
    all_tablenames = getAllGraphName()  
    data = get_wordclusterdata(table_id)  
    return jsonify({'data': data}), 200
    #print(table_id)
    # if table_id in all_tablenames:
    #     data = get_wordclusterdata(table_id)  
    #     return jsonify({'data': data}), 200
    # else:
    #     # Return 404 if table name not found
    #     return jsonify({'error': "Table not found"}), 404

#################################################
@app.route('/reinforcementlearn_data/<string:table_id>', methods=['GET'])
def reinforcementlearn_data(table_id):
    """
    Route to fetch table data for a given table ID.
    """
    data = get_reinforcementlearn_data(table_id)
    if data:
        return jsonify({'data': data})
    else:
        return jsonify({'error': 'No data found for the given table ID or an error occurred'}), 404
###############################################
@app.route('/reinforcementlearn_chat', methods=['POST'])
def reinforcementlearn_chat():
    row = request.json.get('selectedRow')
    if not row:
        return jsonify({'error': 'No row provided'}), 400
    data = reinforcement_learn_modal(row)
    # if not row_id:
    #     return jsonify({'error': 'No id provided'}), 400
    # # Continue with processing here...
    return jsonify({'data':data}), 200
#####################################################
@app.route('/airesearcher/<string:table_id>', methods=['GET'])
def airesearcher(table_id):
    """
    Route to fetch table data for a given table ID.
    """
    # Define the file name
    file_name = f"{table_id}keywordsdata.txt"

    # Check if the file exists in Azure Blob Storage
    if check_files_in_blob(file_name):
        print(f"File {file_name} exists in Azure Blob Storage. Fetching data...")
        data = fetch_data_from_blob(file_name)
        print(data)
        if data:
            return jsonify({'data': data})  # Remove {"data": [...]}
        else:
            return jsonify({'error': 'Error occurred while fetching data from the file'}), 500
    else:
        print(f"File {file_name} does not exist. Fetching data via API and saving to blob storage...")
        # Fetch the data using the API
        data = get_all_keywords(table_id)
        if data:
            # Convert data to string for saving
            data_str = ','.join(data)
            # Save the data as a file in Azure Blob Storage
            aidata = ai_research(data_str)
            #print(data_str)
            #aidata = trend_classify(data_str)
            ai_data = json.dumps(aidata, indent=2)
            save_data_to_blob(file_name, ai_data)
            return jsonify({'data': aidata})
        else:
            return jsonify({'error': 'No data found for the given table ID or an error occurred'}), 404

##########################################################

@app.route('/chatwithopenai', methods=['POST'])
def chatwithopenai():
    # Get the message from the request
    row = request.json.get('message')
    print("Received message:", row)  # For debugging

    # Validate the input
    if not row:
        return jsonify({'error': 'No message provided'}), 400

    # Mock response for testing
    #response = f"Received your message: {row}"
    response = chat_with_models(row)
    print(response)


    # Return the response
    return jsonify(response), 200
################################################
def replace_nan(value):
    """
    Replace NaN or None values with 'N/A'.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return value

@app.route('/timelinedata/<string:table_id>', methods=['GET'])
def timelinedata(table_id):
    """
    Route to fetch table data for a given table ID.
    """
    try:
        all_tablenames = getAllGraphName()  # Fetch all available table names
        data = timeline_chatdata(table_id)
        json_str = json.dumps(data)
        cleaned_data = json.loads(json_str, object_hook=lambda d: {k: replace_nan(v) for k, v in d.items()})
        return jsonify({'data': cleaned_data}), 200
        # if table_id in all_tablenames:
        #     print("Table name found:", table_id)
            
        #     # Fetch the data for the given table ID
        #     data = timeline_chatdata(table_id)

        #     # Replace NaN/None values in the data
        #     json_str = json.dumps(data)
        #     cleaned_data = json.loads(json_str, object_hook=lambda d: {k: replace_nan(v) for k, v in d.items()})
            
        #     #print("Cleaned Data:", cleaned_data)

        #     return jsonify({'data': cleaned_data}), 200
        # else:
        #     # Return 404 if table name is not found
        #     return jsonify({'error': "Table not found"}), 404
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500
#####################################################
@app.route('/heatmapdata/<string:table_id>', methods=['GET'])
def heatmapdata(table_id):
    """
    Route to fetch table data for a given table ID.
    """
    try:
        all_tablenames = getAllGraphName()  # Fetch all available table names
        data = heatmap_chatdata(table_id)
        json_str = json.dumps(data)
        cleaned_data = json.loads(json_str, object_hook=lambda d: {k: replace_nan(v) for k, v in d.items()})
        return jsonify({'data': cleaned_data}), 200
        # if table_id in all_tablenames:
        #     print("Table name found:", table_id)
            
        #     # Fetch the data for the given table ID
        #     data = heatmap_chatdata(table_id)

        #     # Replace NaN/None values in the data
        #     json_str = json.dumps(data)
        #     cleaned_data = json.loads(json_str, object_hook=lambda d: {k: replace_nan(v) for k, v in d.items()})
            
        #     #print("Cleaned Data:", cleaned_data)

        #     return jsonify({'data': cleaned_data}), 200
        # else:
        #     # Return 404 if table name is not found
        #     return jsonify({'error': "Table not found"}), 404
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500
############################################################
def replace_nan(value):
    return value if value is not None else "N/A"

@app.route('/geomap/<string:table_id>', methods=['GET'])
def geomap(table_id):
    print("--------------",table_id)
    """
    Route to fetch table data for a given table ID.
    """
    try:
        all_tablenames = getAllGraphName()  # Fetch all available table names
        data = geomap_data(table_id)
        if data is None:
            return jsonify({'error': "No data found for the specified table ID"}), 404

        return jsonify(json.loads(data)), 200
        # if table_id in all_tablenames:
        #     print("Table name found:", table_id)
            
        #     # Fetch the data for the given table ID
        #     data = geomap_data(table_id)
        #     print(data)
            
        #     if data is None:
        #         return jsonify({'error': "No data found for the specified table ID"}), 404

        #     # Replace NaN/None values in the data
        #     json_str = json.dumps(data)
        #     cleaned_data = json.loads(json_str, object_hook=lambda d: {k: replace_nan(v) for k, v in d.items()})
            
        #     #print("Cleaned Data:", cleaned_data)

        #     return jsonify({'data': cleaned_data}), 200
        # else:
        #     # Return 404 if table name is not found
        #     return jsonify({'error': "Table not found"}), 404
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500
####################################################
def clean_nan_values(data):
    """
    Recursively clean NaN values in a JSON-like dictionary by replacing them with "Other".
    """
    if isinstance(data, dict):
        return {k: clean_nan_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_nan_values(item) for item in data]
    elif pd.isna(data) or data in [None, "NaN", np.nan, float("nan")]:
        return "Other"
    else:
        return data
        
@app.route('/sunbrust/<string:table_id>', methods=['GET'])
def sunbrust(table_id):
    print("--------------", table_id)
    """
    Route to fetch table data for a given table ID.
    """
    try:
        all_tablenames = getAllGraphName()  # Fetch all available table names
        data = sunbrust_data(table_id)
        if data is None:
            return jsonify({'error': "No data found for the specified table ID"}), 404

        # Ensure all NaN values are replaced properly
        cleaned_data = clean_nan_values(data)

        return jsonify({"data": cleaned_data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

###############################################
import math
def sanitize_data(data):
    def replace_nan(value):
        return "N/A" if value is None or (isinstance(value, float) and math.isnan(value)) else value
    
    if isinstance(data, list):
        return [sanitize_data(item) for item in data]
    elif isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    else:
        return replace_nan(data)

@app.route('/getTableData/<string:table_id>', methods=['GET'])
def get_tableData(table_id):
    # Get all graph names
    data = get_TableData(table_id) 
    print(data)
    s_data = sanitize_data(data) 
    return jsonify({'data': s_data}), 200
    #all_tablenames = getAllGraphName()  
    #print(table_id)
    # if table_id in all_tablenames:
    #     data = get_TableData(table_id) 
    #     s_data = sanitize_data(data) 
    #     return jsonify({'data': s_data}), 200
    # else:
    #     # Return 404 if table name not found
    #     return jsonify({'error': "Table not found"}), 404

#############################################
## used to duplicate name not used on graphData table
@app.route('/get_allgraphname', methods=['GET'])
def get_allGraphName():
    #get all graph name ## used same name not used
    data = getAllGraphName()  
    return jsonify({'data': data})

###############################################
## get all data from grapgDate table from azure ## call on home page
@app.route('/totaluploadData', methods=['GET'])
def totaluploadData():
    data = getGraphData()
    print(data)
    return jsonify({'data': data})


#########################################
## UPLOAD FILE FUNCTION
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and extract sheet names."""
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No valid file provided'}), 400

    file = request.files['file']
    try:
        df = pd.read_excel(file, sheet_name=None)
        sheet_names = list(df.keys())
        return jsonify({'sheets': sheet_names})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_columns', methods=['POST'])
def get_columns():
    """Get column names of a specific sheet."""
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No valid file provided'}), 400
    sheet_name = request.form.get('sheet_name')
    file = request.files['file']
    try:
        df = pd.read_excel(file, sheet_name=sheet_name)
        return jsonify({'columns': df.columns.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def clean_sheetname(text):
    text = re.sub(r"[0-9.,;/:'\"']", '', text) 
    # Replace extra spaces with a single underscore
    text = re.sub(r'\s+', '', text.strip())  
    return text

def get_keywords(idx, docs, feature_names):
    """Generate keywords for a specific document index."""
    # Generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform([docs[idx]]))

    # Sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # Extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

    return keywords

def task1(metadata, tablename,metadata2,sheet_name,name):


    #logging.basicConfig(level=logging.INFO)  # Configure logging
    sourcenode = []

    for data in metadata:
        try:
            # Skip processing if data is NaN or empty
            if pd.isna(data) or not str(data).strip():
                print('=================================')
                logging.warning(f"Skipping empty or NaN data: {data}")
                print('=================================')
                continue

            cleaned_text = preprocess_text(data)
            result = chat(cleaned_text)  # Ensure this function is robust and well-tested

            # Extract JSON substring from result
            pattern = r"```json\n(.*?)\n```"
            match = re.search(pattern, json.loads(result).get('output', ''), re.DOTALL)
            if match:
                raw_json = match.group(1).strip()
                try:
                    result = json.loads(raw_json)
                    result_json = result.get('result', [])[0]  # Safely get first item
                    if result_json:  # Ensure result_json is not None
                        json_object = {
                            "metadata":data,
                            "keywords": result_json.get("keywords", []),
                            "theme": result_json.get("Theme", ""),
                            "safety": result_json.get("safety", ""),
                            "diagnose": result_json.get("diagnose", ""),
                            "treatment": result_json.get("treatment", ""),
                            "synonyms": result_json.get("synonyms", []),
                            "sentiment": result_json.get("sentiment", ""),
                            "nodeid": [node.get('id') for node in result_json.get("nodes", [])],
                            "group": [node.get('group') for node in result_json.get("nodes", [])],
                            "label": [node.get('label') for node in result_json.get("nodes", [])],
                            "source": [link.get('source') for link in result_json.get("links", [])],
                            "target": [link.get('target') for link in result_json.get("links", [])],
                            "relationship": [link.get('relationship') for link in result_json.get("links", [])],
                            "analyze_thoroughly": result_json.get("AnalyzeThoroughly", ""),
                            "theme_statement": result_json.get("THEME", ""),
                            "issue": result_json.get("ISSUE", ""),
                        }
                        #logging.info(f"Processed JSON object: {json_object}")
                        sourcenode.append(json_object)
                    else:
                        logging.warning("No valid result found in JSON data.")
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON: {e}")
                    #logging.debug(f"Raw JSON: {raw_json}")
            else:
                logging.warning("No JSON found in the input string.")
        except Exception as e:
            logging.error(f"Error processing metadata '{data}': {e}")
            #logging.debug(traceback.format_exc())

    logging.info(f"Inserting into table '{tablename}' with {len(sourcenode)} entries.")
    try:
        get_table_client(tablename)
        insert_json_data_list(sourcenode, tablename)
        insert_data_graphtable(clean_sheetname(metadata2), sheet_name, name)
    except Exception as e:
        logging.error(f"Error inserting data into table '{tablename}': {e}")
        #logging.debug(traceback.format_exc())

    #print(sourcenode)

def keywords_extract(data):
    result = chat(data)  # Assuming chat() returns a JSON response

    pattern = r"```json\n(.*?)\n```"
    match = re.search(pattern, json.loads(result).get('output', ''), re.DOTALL)

    if match:
        raw_json = match.group(1).strip()
        try:
            result = json.loads(raw_json)
            result_json = result.get('result', [{}])[0]  # Default to empty dict if list is empty

            return {
                "metadata": data,
                "keywords": result_json.get("keywords", []),
                "theme": result_json.get("Theme", ""),
                "safety": result_json.get("safety", ""),
                "diagnose": result_json.get("diagnose", ""),
                "treatment": result_json.get("treatment", ""),
                "synonyms": result_json.get("synonyms", []),
                "sentiment": result_json.get("sentiment", ""),
                "nodeid": [node.get('id') for node in result_json.get("nodes", [])],
                "group": [node.get('group') for node in result_json.get("nodes", [])],
                "label": [node.get('label') for node in result_json.get("nodes", [])],
                "source": [link.get('source') for link in result_json.get("links", [])],
                "target": [link.get('target') for link in result_json.get("links", [])],
                "relationship": [link.get('relationship') for link in result_json.get("links", [])],
                "analyze_thoroughly": result_json.get("AnalyzeThoroughly", ""),
                "theme_statement": result_json.get("THEME", ""),
                "issue": result_json.get("ISSUE", ""),
            }
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            return {}
    return {}

# Function to clean quotes
def clean_quotes(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'^[",\s\\/]+|[",\s\\/]+$', '', text)  # Remove unwanted characters
    text = re.sub(r'\s+/n\s+', ' ', text)  # Fix newline issues
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    return text

# Function to check if a column has at least 5 words
def has_min_words(text, min_words=5):
    if pd.isna(text):
        return False
    words = str(text).strip().split()
    return len(words) >= min_words

@app.route('/get_column_data', methods=['POST'])
def get_column_data():
    """Process column data and return cleaned and transformed content."""
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No valid file provided'}), 400

    metadata = request.form.get('metadata', '')
    sheet_name = request.form.get('sheet_name', '')
    name = request.form.get('name', '')
    column_names = json.loads(request.form.get('column_names', '[]'))
    file = request.files['file']

    table_name = re.sub(r"[.,;:'\"']", '', name)
    table_name = re.sub(r'\s+', '', table_name.strip())

    cleaned_name = clean_sheetname(metadata)  # Assuming you have this function

    try:
        df = pd.read_excel(file, sheet_name=sheet_name)

        if metadata in df.columns:
            df[metadata] = df[metadata].apply(clean_quotes)

        # Only process if at least one row has 5+ words
        if metadata in df.columns and df[metadata].apply(has_min_words).any():
            print("Processing data...")

            # Remove duplicates and empty rows/columns
            df = df.drop_duplicates()
            df = df.dropna(how='all').dropna(axis=1, how='all')

        else:
            return jsonify({"message": "Skipping processing: 'quotes' column has insufficient data."}), 400

    except Exception as e:
        return jsonify({"error": f"Error reading sheet '{sheet_name}': {str(e)}"}), 400

    results = {}
    cleantext = []
    processed_rows = []
    table1 = table_name
    get_table_client(table1)
    # Process rows and extract keywords
    for _, row in df.iterrows():
        result2={}
        row_data = {col: row[col] for col in column_names if col in df.columns}

        if metadata in row_data and pd.notna(row_data[metadata]):
            feedback = row_data[metadata]
            preprocessed_text = preprocess_text(feedback)  # Assuming this function exists
            chatresult = keywords_extract(feedback) or {}  # Ensure no None errors

            row_data.update({
                "metadata":feedback,
                "keywords": chatresult.get("keywords", []),
                "theme": chatresult.get("theme", ""),
                "safety": chatresult.get("safety", ""),
                "diagnose": chatresult.get("diagnose", ""),
                "treatment": chatresult.get("treatment", ""),
                "synonyms": chatresult.get("synonyms", []),
                "sentiment": chatresult.get("sentiment", ""),
                "nodeid": chatresult.get("nodeid", []),
                "group": chatresult.get("group", []),
                "label": chatresult.get("label", []),
                "source": chatresult.get("source", []),
                "target": chatresult.get("target", []),
                "relationship": chatresult.get("relationship", []),
                "analyze_thoroughly": chatresult.get("analyze_thoroughly", ""),
                "theme_statement": chatresult.get("theme_statement", ""),
                "issue": chatresult.get("issue", ""),
            })

            cleantext.append(preprocessed_text)
            row_data["preprocessedText"] = preprocessed_text
            result2.setdefault(cleaned_name, []).append(row_data)
            insert_json_data_list(result2[cleaned_name], table1)
            processed_rows.append(row_data)

    # Vectorize text (assuming CountVectorizer and TfidfTransformer exist)
    # word_count_vectors = cv.fit_transform(cleantext)
    # tfidf_transformer.fit(word_count_vectors)
    # feature_names = cv.get_feature_names_out()

    # for idx, row_data in enumerate(processed_rows):
    #     try:
    #         keywords = get_keywords(idx, cleantext, feature_names)  # Assuming this function exists
    #         row_data["keywords_by_sklearn"] = list(keywords.keys())
    #         results.setdefault(cleaned_name, []).append(row_data)
    #     except Exception as e:
    #         row_data["keywords_by_sklearn"] = {"error": str(e)}
    #         results.setdefault(cleaned_name, []).append(row_data)

    # #print(results)
    # # Create tables and insert data
    # table1 = table_name
    # # table2 = f"{table_name}sourcenode"
    # get_table_client(table1)
    # insert_json_data_list(results[cleaned_name], table1)
    #insert_data_graphtable(clean_sheetname(metadata), sheet_name, name)

    # Run task1 in the background
    #executor.submit(task1, df[metadata].tolist(), table2,metadata,sheet_name,name)

    # Return immediate response
    return jsonify({"message": "Processing started successfully"}), 200



if __name__ == '__main__':
    app.run(debug=True)