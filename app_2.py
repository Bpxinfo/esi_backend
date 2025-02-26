from flask import Flask, request, jsonify, render_template
import pandas as pd
import json
import logging
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from flask_cors import CORS
import spacy
import traceback
from llama31  import chat
from concurrent.futures import ThreadPoolExecutor
from azure_table_2 import *
from reinforcement_learn import *
from azure_blob import *
from ai_research import *
from trend_topic import *
from chatai import *
# from flask_socketio import SocketIO, emit
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
# socketio = SocketIO(app, cors_allowed_origins="*") 

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
    #all_tablenames = getAllGraphName()  
    #print(table_id)
    #if table_id in all_tablenames:
    data = get_networkgraphdata_by_id(table_id)  
    return jsonify({'data': data}), 200
    #else:
        # Return 404 if table name not found
        #return jsonify({'error': "Table not found"}), 404

  ####################################################
@app.route('/wordclusterData/<string:table_id>', methods=['GET'])
def wordclusterData(table_id):
    """
    Route to fetch table data for a given table ID.
    """
    #all_tablenames = getAllGraphName()  
    #print(table_id)
    #if table_id in all_tablenames:
    data = get_wordclusterdata(table_id)  
    return jsonify({'data': data}), 200
    #else:
        # Return 404 if table name not found
        #return jsonify({'error': "Table not found"}), 404

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
    print('----------------------')
    print(row)
    print('----------------------')
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
        if data:
            return jsonify({'data': json.loads(data)})
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
            #aidata = ai_research(data_str)
            print(data_str)
            aidata = trend_classify(data_str)
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
        #all_tablenames = getAllGraphName()  # Fetch all available table names

        #if table_id in all_tablenames:
        print("Table name found:", table_id)
            
            # Fetch the data for the given table ID
        data = timeline_chatdata(table_id)

            # Replace NaN/None values in the data
        json_str = json.dumps(data)
        cleaned_data = json.loads(json_str, object_hook=lambda d: {k: replace_nan(v) for k, v in d.items()})
            
            #print("Cleaned Data:", cleaned_data)

        return jsonify({'data': cleaned_data}), 200
        #else:
            # Return 404 if table name is not found
            #return jsonify({'error': "Table not found"}), 404
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500
#####################################################
@app.route('/heatmapdata/<string:table_id>', methods=['GET'])
def heatmapdata(table_id):
    """
    Route to fetch table data for a given table ID.
    """
    #try:
        #all_tablenames = getAllGraphName()  # Fetch all available table names

        #if table_id in all_tablenames:
        #print("Table name found:", table_id)
            
            # Fetch the data for the given table ID
    data = heatmap_chatdata(table_id)

            # Replace NaN/None values in the data
    json_str = json.dumps(data)
    cleaned_data = json.loads(json_str, object_hook=lambda d: {k: replace_nan(v) for k, v in d.items()})
            
            #print("Cleaned Data:", cleaned_data)

    return jsonify({'data': cleaned_data}), 200
        #else:
            # Return 404 if table name is not found
            #return jsonify({'error': "Table not found"}), 404
    # except Exception as e:
    #     # Handle unexpected errors
    #     return jsonify({'error': str(e)}), 500
############################################################
@app.route('/geomap/<string:table_id>', methods=['GET'])
def geomap(table_id):
    """
    Route to fetch table data for a given table ID.
    """
    try:
        #all_tablenames = getAllGraphName()  # Fetch all available table names

        #if table_id in all_tablenames:
        #print("Table name found:", table_id)
            
            # Fetch the data for the given table ID
        data = geomap_data(table_id)

            # Replace NaN/None values in the data
        json_str = json.dumps(data)
        cleaned_data = json.loads(json_str, object_hook=lambda d: {k: replace_nan(v) for k, v in d.items()})
            
            #print("Cleaned Data:", cleaned_data)

        return jsonify({'data': cleaned_data}), 200
        # else:
        #     # Return 404 if table name is not found
        #     return jsonify({'error': "Table not found"}), 404
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500

###############################################
############################################################
@app.route('/sunbrust/<string:table_id>', methods=['GET'])
def sunbrust(table_id):
    """
    Route to fetch table data for a given table ID.
    """
    try:
        #all_tablenames = getAllGraphName()  # Fetch all available table names

        #if table_id in all_tablenames:
        #print("Table name found:", table_id)
            
            # Fetch the data for the given table ID
        data = sunbrust_data(table_id)

            # Replace NaN/None values in the data
        json_str = json.dumps(data)
        cleaned_data = json.loads(json_str, object_hook=lambda d: {k: replace_nan(v) for k, v in d.items()})
            
            #print("Cleaned Data:", cleaned_data)

        return jsonify({'data': cleaned_data}), 200
        # else:
        #     # Return 404 if table name is not found
        #     return jsonify({'error': "Table not found"}), 404
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500

################################################
@app.route('/sunbrust2/<string:table_id>', methods=['GET'])
def sunbrust2(table_id):
    """
    Route to fetch table data for a given table ID.
    """
    try:
        #all_tablenames = getAllGraphName()  # Fetch all available table names

        #if table_id in all_tablenames:
        #print("Table name found:", table_id)
            
            # Fetch the data for the given table ID
        data = sunbrust_data_2(table_id)

            # Replace NaN/None values in the data
        json_str = json.dumps(data)
        cleaned_data = json.loads(json_str, object_hook=lambda d: {k: replace_nan(v) for k, v in d.items()})
            
            #print("Cleaned Data:", cleaned_data)
        print(cleaned_data)
        return jsonify({'data': cleaned_data}), 200
        # else:
        #     # Return 404 if table name is not found
        #     return jsonify({'error': "Table not found"}), 404
    except Exception as e:
        # Handle unexpected errors
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
    #all_tablenames = getAllGraphName()  
    #print(table_id)
    #if table_id in all_tablenames:
    data = get_TableData(table_id) 
    print(data)
    s_data = sanitize_data(data) 
    return jsonify({'data': s_data}), 200
    #else:
        # Return 404 if table name not found
        #return jsonify({'error': "Table not found"}), 404

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
    text = re.sub(r"[0-9.,;/:'\"']", '', text)  # Remove unwanted characters
    text = re.sub(r'\s+', '', text.strip())  # Replace extra spaces
    return text

def get_keywords(idx, docs, feature_names):
    """Generate keywords for a specific document index."""
    tf_idf_vector = tfidf_transformer.transform(cv.transform([docs[idx]]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    return extract_topn_from_vector(feature_names, sorted_items, 10)

def extract_json_from_text(text):
    # Regular expression to extract JSON between curly braces
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(0)  # Extract matched JSON string
        
        try:
            json_data = json.loads(json_str)  # Parse JSON
            return json_data
        except json.JSONDecodeError:
            print("Error: Invalid JSON format.")
            return None
    else:
        print("Error: No JSON data found.")
        return None

def process_metadata(df, metadata_column):
    processed_data = []  # Store results here
    sno = 1  # Serial number counter

    for _, row in df.iterrows():
        data = row.get(metadata_column, None)  # Fetch metadata dynamically
        print(f"metadata - {data}")
        try:
            if pd.isna(data) or not str(data).strip():
                continue  # Skip empty or NaN data

            cleaned_text = preprocess_text(data)  # Preprocessing
            chat_result = chat(cleaned_text, sno)  # Get processed metadata
            
            logging.info("Chat Result: %s", chat_result)
            if not chat_result:
                continue  # Skip if chat returns empty
            
            sno += 1  # Increment sno after processing

            # Safely parse JSON response
            try:
                #r = chat_result.replace("```json\n", '').replace("'", '').replace("```", "")
                json_data = extract_json_from_text(chat_result)
                #json_data = json.loads(r)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding error: {e}")
                continue  # Skip this row if JSON fails

            data = json_data.get('result', [])

            if not data:
                continue  # Skip if result is empty

            first_entry = data[0] if data else {}

            # Create a new row dictionary that includes all original columns dynamically
            processed_entry = row.to_dict()  # Convert row to dictionary
            processed_entry.update({
                "sno": sno,  
                "metadata": data,
                "keywords": first_entry.get('keywords', []),
                "theme": first_entry.get("Theme", ""),
                "theme_statement": first_entry.get("THEME", ""),
                "analyze_thoroughly": first_entry.get("AnalyzeThoroughly", ""),
                "issue": first_entry.get("ISSUE", ""),
                "safety": first_entry.get("safety", ""),
                "diagnose": first_entry.get("diagnose", ""),
                "treatment": first_entry.get("treatment", ""),
                "synonyms": first_entry.get('synonyms', []),
                "sentiment": first_entry.get("sentiment", ""),
                "nodeid": [node.get('id', '') for node in first_entry.get("nodes", [])],
                "group": [node.get('group', '') for node in first_entry.get("nodes", [])],
                "label": [node.get('label', '') for node in first_entry.get("nodes", [])],
                "source": [link.get('source', '') for link in first_entry.get("links", [])],
                "target": [link.get('target', '') for link in first_entry.get("links", [])],
                "relationship": [link.get('relationship', '') for link in first_entry.get("links", [])],
            })

            processed_data.append(processed_entry)

        except Exception as e:
            logging.error(f"Error processing metadata '{data}': {e}")

    # Convert processed list into a DataFrame
    processed_metadata_df = pd.DataFrame(processed_data)
    processed_metadata_df.replace({pd.NA: None}, inplace=True)  # Handle missing values

    return processed_metadata_df


@app.route('/get_column_data', methods=['POST'])
def get_column_data():
    """Processes uploaded Excel file and returns extracted data with API response."""
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No valid file provided'}), 400

    metadata_column = request.form.get('metadata', '').strip()
    sheet_name = request.form.get('sheet_name', '').strip()
    name = request.form.get('name', '').strip()
    column_names = json.loads(request.form.get('column_names', '[]'))
    file = request.files['file']

    logging.info("Received file for processing: %s", file.filename)
    logging.info("Sheet Name: %s, Metadata Column: %s, Column Names: %s", sheet_name, metadata_column, column_names)

    try:
        excel_file = pd.ExcelFile(file)
        if sheet_name not in excel_file.sheet_names:
            return jsonify({'error': f"Sheet '{sheet_name}' not found in the uploaded file"}), 400

        df = pd.read_excel(file, sheet_name=sheet_name, usecols=column_names)
    except Exception as e:
        return jsonify({"error": f"Error reading sheet '{sheet_name}': {str(e)}"}), 400

    if df.empty:
        return jsonify({'error': 'The uploaded sheet is empty'}), 400

    # Step 1: Assign Unique Key (`sno`)
    if 'sno' not in df.columns:
        df.insert(0, 'sno', range(1, len(df) + 1))

    logging.info("DataFrame after inserting sno: \n%s", df.head())

    # Step 2: Process Metadata (Extract Keywords + Call Chat API)
    processed_metadata_df = process_metadata(df, metadata_column)
    logging.info("Processed Metadata: \n%s", processed_metadata_df)

    # Step 3: Merge All Data
    final_json_data = processed_metadata_df.to_dict(orient='records')
    print(final_json_data)
    # Insert into database
    insert_json_data_list(final_json_data, name)
    insert_data_graphtable(clean_sheetname(metadata_column), sheet_name, name)

    # Step 4: Convert to JSON and Return
    return jsonify({
        "data": final_json_data,
        "message": "Processing completed successfully"
    }), 200


if __name__ == '__main__':
    app.run(debug=True)