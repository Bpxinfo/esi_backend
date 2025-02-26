from azure.data.tables import TableServiceClient, TableEntity
from azure.core.exceptions import ResourceExistsError
import uuid
import datetime
from datetime import datetime
import pandas as pd
import os
import re
import json
import math
from collections import Counter,defaultdict
import pytz  # If you want to handle timezone-aware timestamps
from dotenv import load_dotenv
import os
import numpy as np
from us_lat_long import *

load_dotenv()

# Fetch the connection string from environment variables
CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=ragchatfiles;AccountKey=09QmkXb6fhd0Qb1vEDKAIjO8OuH2y13A7LbdmPxvrbWzpiaaCeIY5HpXjjPJk3uAwaTdLXnNjwll+ASt8ebWrw==;EndpointSuffix=core.windows.net"

# def getalltablesname():
#     tablename = []
#     service_client = TableServiceClient.from_connection_string(conn_str=CONNECTION_STRING)
#     tables = service_client.list_tables()
#     for table in tables: 
#         tablename.append(table.name)
#         #print(table.name)
#     return tablename


# Initialize TableServiceClient
def get_table_client(TABLE_NAME):
    service_client = TableServiceClient.from_connection_string(conn_str=CONNECTION_STRING)
    try:
        # Create the table if it doesn't exist
        service_client.create_table(TABLE_NAME)
        print(f"Table '{TABLE_NAME}' created successfully.")
    except ResourceExistsError:
        print(f"Table '{TABLE_NAME}' already exists.")
    return service_client.get_table_client(TABLE_NAME)

# Insert a record into the table
def insert_data_graphtable(metadata,sheet_name,name):
    try:
        table_client = get_table_client("GraphData")

        # Entity must have PartitionKey and RowKey
        entity = {
            "PartitionKey": "ExcelSheet",  # Logical grouping of rows
            "RowKey": str(uuid.uuid4()),  # Unique within PartitionKey
            "sheetname": sheet_name,
            "metadata": metadata,
            "name": name,
            "Date":  datetime.now()
        }

        # Insert or merge the entity
        table_client.create_entity(entity)
        print("Data inserted successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

## INSERT DATA

def sanitize_property_name(property_name):
    """Sanitize column names to fit Azure Table Storage naming rules."""
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", property_name)  # Replace invalid characters with _
    sanitized = re.sub(r"^[_\d]+", "Prop_", sanitized)  # Ensure it doesn't start with a number or underscore
    return sanitized


def insert_json_data_list(json_data_list, table_name):
    """Insert JSON data dynamically into Azure Table Storage."""
    table_client = get_table_client(table_name)

    for index, json_data in enumerate(json_data_list):
        # Ensure json_data is a dictionary
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)  # Convert from JSON string to dictionary
            except json.JSONDecodeError:
                print(f" Failed to parse JSON for index {index}. Skipping.")
                continue  # Skip invalid entries

        # Construct unique PartitionKey and RowKey
        partition_key = "ProcessedData"  # Logical grouping
        # row_key = str(index + 1)  # Unique row key
        row_key = str(uuid.uuid4())  # Generate unique UUID


        # Construct entity
        entity = {
            "PartitionKey": partition_key,
            "RowKey": row_key,
        }

        # Add JSON fields dynamically
        for key, value in json_data.items():
            sanitized_key = sanitize_property_name(key)  # Use sanitized key

            # Handle datetime fields
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    value = pytz.utc.localize(value)  # Convert naive datetime to UTC
                entity[sanitized_key] = value.isoformat()

            # Handle list fields
            elif isinstance(value, list):
                entity[sanitized_key] = json.dumps(value)  # Convert lists to JSON strings

            # Handle empty values
            elif value is None or value == "":
                entity[sanitized_key] = "N/A"  # Store empty fields as "N/A"

            else:
                entity[sanitized_key] = value  # Store directly

        # Insert entity into Azure Table
        try:
            table_client.create_entity(entity)
            print(f" Data inserted for RowKey '{row_key}'.")
        except Exception as e:
            print(f" Error inserting RowKey '{row_key}': {str(e)}")
########################################################
def insert_to_azure_table_2(data, table_name):
    """Insert cleaned data into Azure Table Storage."""
    table_client = get_table_client(table_name)
    try:
        table_client.upsert_entity(entity=data)  # Upsert prevents duplicate errors
        print(f" Successfully inserted into {table_name}")
    except Exception as e:
        print(f" Error inserting data: {e}")
######################################################
#get all data from graphData table from azure.
def getGraphData():
    try:
        # Get the table client
        table_client = get_table_client("GraphData") 
        # Query all entities in the table
        all_records = list(table_client.list_entities())
        # entities = table_client.query_entities(
        #     query_filter="",
        #     select=['tablename']  # Specify only the column you need
        # )      
        return all_records
    
    except HttpResponseError as e:
        print(f"An error occurred: {e}")
        return None
####################################################
#get all name from graphData table azure
def getAllGraphName():
    # Query to select specific columns
    table_client = get_table_client("GraphData")
    selected_columns = ["name"]

    # Fetch the data
    name=[]
    try:
        entities = table_client.query_entities(select=selected_columns,query_filter='')
        for entity in entities:
            name.append(entity.get('name'))
            #print(f"Name: {entity.get('name')}")
        return name
    except Exception as e:
        print(f"An error occurred: {e}")
###################################################
#####################################################
## get all column name from azure table
def getallcolumnname(tablename):
    table_client = get_table_client(tablename)
    column_names = set()  # Using a set to avoid duplicates
    try:
        # Fetch a few entities to get properties (columns)
        entities = table_client.list_entities()  # Limiting to default 100 entities
        for entity in entities:
            # Exclude PartitionKey, RowKey, and Timestamp
            for key in entity.keys():
                if key not in ['PartitionKey', 'RowKey', 'Timestamp']:
                    column_names.add(key)  # Collect column names dynamically
    except Exception as e:
        print(f"Error fetching column names from {tablename}: {e}")
    
    return list(column_names)

# Function to fetch data from a table with selected columns
def fetch_table_data(table_name, select_columns, query_filter=None):
    table_client = get_table_client(table_name)
    print("----------------tablename--------------------")
    print(table_name)
    try:
        # Fetch data with optional filter
        entities = table_client.query_entities(select=select_columns, query_filter=query_filter)
        results = []
        for entity in entities:
            results.append({col: entity.get(col) for col in select_columns})
        return results
    except Exception as e:
        print(f"Error fetching data from table {table_name}: {e}")
        return []

# Function to get metadata and keywords data
def get_TableData(tablename):
    print("tableid--",tablename)
    def fetch_table_data(table_name,rename_keywords=False):
        table_client = get_table_client(table_name)
        data = {}
        for row in table_client.list_entities():
            row_key = row["RowKey"]
            row_data = dict(row)

            # # Rename "keywords" column if fetching from Table1
            # if rename_keywords and "keywords" in row_data:
            #     row_data["keywords_by_ml_model"] = row_data.pop("keywords")

            data[row_key] = row_data
        return data
 
    table1_name = tablename
    #table2_name = f"{tablename}sourcenode"

    table1_data = fetch_table_data(table1_name,rename_keywords=True)
    print(table1_data)
    #table2_data = fetch_table_data(table2_name,rename_keywords=False)
    #merged_data = {}

    #for row_key in set(table1_data.keys()).union(table2_data.keys()):
     #   merged_data[row_key] = {**table1_data.get(row_key, {}), **table2_data.get(row_key, {})}

    # Convert merged data to list format
    #merged_list = [{"RowKey": k, **v} for k, v in merged_data.items()]


    # # # Fetch columns dynamically from both tables
    # #table2_columns = getallcolumnname(table2_name)  # Get dynamic columns for the second table
    # table2_columns = ['metadata','keywords','synonyms','sentiment','theme','issue','analyze_thoroughly']
    # keywords_data = fetch_table_data(table2_name, table2_columns)

    return table1_data

#########################################
##############################################
def transform_data(input_data):
    nodes = []
    links = []
    
    for entry in input_data:
        # Parse the JSON-like strings into actual lists
        group = json.loads(entry['group'])
        label = json.loads(entry['label'])
        nodeid = json.loads(entry['nodeid'])
        relationship = json.loads(entry['relationship'])
        source = json.loads(entry['source'])
        target = json.loads(entry['target'])
        
        # Construct nodes
        for i in range(len(nodeid)):
            nodes.append({
                "id": nodeid[i],
                "label": label[i],
                "group": group[i]
            })
        
        # Construct links
        for i in range(len(source)):
            links.append({
                "source": source[i],
                "target": target[i],
                "relationship": relationship[i]
            })
    
    return {
        "node": nodes,
        "link": links
    }
def get_networkgraphdata_by_id(tablename):
    table1_name = tablename
    #table2_name = f"{tablename}sourcenode"
    # # Fetch columns dynamically from both tables
    table1_columns = ["nodeid","label","relationship","source","target","group"]
    network_data = fetch_table_data(table1_name, table1_columns)

    return transform_data(network_data)

#########################################
from collections import Counter

def get_word_counts(api_data):
    # Initialize a Counter to accumulate counts
    total_counts = Counter()
    
    # Iterate over the "data" values in the API response
    for value in api_data:
        # Convert the string representation of the list to an actual list
        items = json.loads(value)
        # Update the Counter with the items in the list
        total_counts.update(items)
    
    # Convert the Counter object to a dictionary
    return dict(total_counts)

def get_wordclusterdata(tablename):
    table1_name = tablename
    #table2_name = f"{tablename}sourcenode"
    
    # Fetch only the "keywords" column from the table
    table1_columns = ["keywords"]
    network_data = fetch_table_data(table1_name, table1_columns)  # List of dictionaries

    # Extract keywords into a flat list
    keywords_list = []
    for row in network_data:
        # Append "keywords" field from each dictionary, if it exists
        if "keywords" in row:
            keywords = row["keywords"]
            if isinstance(keywords, list):
                keywords_list.extend(keywords)  # If it's a list, extend the list
            elif isinstance(keywords, str):
                keywords_list.append(keywords)  # If it's a string, append directly
    
    # Generate the word cluster data
    output_object = get_word_counts(keywords_list)
    return output_object

#####################################################
def get_reinforcementlearn_data(tablename):
    table1_name = tablename
    #table2_name = f"{tablename}sourcenode"
    
    # Fetch only the "keywords" column from the table
    table1_columns = ["metadata"]
    network_data = fetch_table_data(table1_name, table1_columns)  # List of dictionaries
    #print(network_data)
    return network_data

#################################################
def cleanedkeywords(keywords):
    # Extract and process the keywords
    unique_keywords = set()
    for entry in keywords:
        # Convert the string of keywords to a list
        keywords_list = json.loads(entry["keywords"])
        unique_keywords.update(keywords_list)

    # Convert back to a list for the output
    output_keywords = list(unique_keywords)
    return output_keywords

def get_all_keywords(tablename):

    table1_name = tablename
    #table2_name = f"{tablename}sourcenode"
    # # Fetch columns dynamically from both tables
    table1_columns = ["keywords"]
    network_data = fetch_table_data(table1_name, table1_columns)

    return cleanedkeywords(network_data)

#####################################################
# Function to transform data
def transform_treemap_data(data,table1_name):
    result = {}
    # columns=['Stakeholder']
    # column_name = find_columname(table1_columns,table1_name)
    print(data)
    for entry in data:
        stakeholder = entry.get("Stakeholder")  # Ensure the key is in lowercase
        keywords = entry.get("keywords")
        print(stakeholder)
        print(keywords)
        if not stakeholder or not keywords:
            continue
        
        # Replace single quotes with double quotes and handle potential malformed JSON
        try:
            keywords = keywords.replace("'", '"')
            keyword_list = json.loads(keywords)
        except json.JSONDecodeError:
            continue
        
        # Count occurrences of each keyword
        keyword_count = Counter(keyword_list)
        
        # Format the result
        result[stakeholder] = [{"stakeholder": stakeholder, "keywords": k, "count": v} for k, v in keyword_count.items()]
    
    return result

def find_columname(keywords,table_name):
    table_client = get_table_client(table_name)
    entities = table_client.list_entities()
    first_entity = next(entities, None)

    if first_entity:
        columns = list(first_entity.keys())  # Extract column names
        filtered_columns = [col for col in columns if any(keyword.lower() in col.lower() for keyword in keywords)]
        return filtered_columns

def heatmap_chatdata(tablename):
    table1_name = tablename
    #table2_name = f"{tablename}sourcenode"
    
    # Fetch only the required columns from the tables
    table1_columns = ["Stakeholder",'RTLL','Region','Speciality','keywords']
    # column_name = find_columname(table1_columns,table1_name)
    # #print(column_name)
    heatmapedata = fetch_table_data(table1_name, table1_columns)  # List of dictionaries

    #table2_columns = ["keywords","RowKey"]
    #heatmapedata2 = fetch_table_data(table2_name, table2_columns)  # List of dictionaries

    # merged_dict = {}

    # # Merge the data from both tables based on RowKey
    # for row in heatmapedata2:
    #     if row.get("keywords"):
    #         merged_dict[row["RowKey"]] = {"keywords": row["keywords"]}

    # for row in heatmapedata:
    #     if row["RowKey"] in merged_dict:
    #         merged_dict[row["RowKey"]]["Stakeholder"] = row["Stakeholder"]
    #         merged_dict[row["RowKey"]]["RTLL"] = row["RTLL"]
    #         merged_dict[row["RowKey"]]["Region"] = row["Region"]
    #         merged_dict[row["RowKey"]]["Speciality"] = row["Speciality"]

    # output_data = list(merged_dict.values())
    #print("OUTPUT DATA  -",output_data)
    #print("HEATMAP -- ",heatmapedata)
    output = transform_treemap_data(heatmapedata,table1_name)
    #print(output)
    # print("FINAL DATA  -",output)
    return output

#####################################################
from collections import defaultdict

def process_timeline_data(input_data):
    # Validate that input_data is a list
    if not isinstance(input_data, list):
        raise ValueError("Input data must be a list of entries")

    # Initialize a dictionary to store results
    result = defaultdict(list)

    # Process each entry in the input data
    for entry in input_data:
        # Validate the presence of required keys
        if not isinstance(entry, dict):
            print(f"Skipping invalid entry: {entry}")
            continue
        
        if "Date" not in entry or "keywords" not in entry:
            print(f"Skipping entry due to missing fields: {entry}")
            continue

        date = entry["Date"]
        keywords_raw = entry["keywords"]

        # Safely evaluate the keywords field
        try:
            keywords = eval(keywords_raw) if isinstance(keywords_raw, str) else keywords_raw
            if not isinstance(keywords, list):
                raise ValueError("keywords must be a list")
        except Exception as e:
            print(f"Skipping entry due to invalid keywords format: {keywords_raw}. Error: {e}")
            continue

        # Count occurrences of each keyword in the current entry
        keyword_counts = defaultdict(int)
        for keyword in keywords:
            if not isinstance(keyword, str):
                print(f"Skipping invalid keyword: {keyword} in entry: {entry}")
                continue
            keyword_counts[keyword] += 1

        # Add counts to the result dictionary
        for keyword, count in keyword_counts.items():
            # Check if date already exists for the keyword, update count if so
            found = False
            for record in result[keyword]:
                if record["Date"] == date:
                    record["count"] += count
                    found = True
                    break
            # If not found, append a new record for this keyword and date
            if not found:
                result[keyword].append({"Date": date, "count": count})

    # Convert result to a standard dictionary for JSON compatibility
    return dict(result)


def timeline_chatdata(tablename):
    table1_name = tablename
    #table2_name = f"{tablename}sourcenode"
    
    # Fetch only the required columns from the tables
    table1_columns = ["Date","keywords"]
    timelinedata = fetch_table_data(table1_name, table1_columns)  # List of dictionaries

    #table2_columns = ["keywords","RowKey"]
    #timelinedata2 = fetch_table_data(table2_name, table2_columns)  # List of dictionaries

    #merged_dict = {}

    # Merge the data from both tables based on RowKey
    # for row in timelinedata:
    #     merged_dict[row["RowKey"]] = {"Date": row["Date"]}
        
    # for row in timelinedata2:
    #     if row["RowKey"] in merged_dict:
    #         merged_dict[row["RowKey"]]["keywords"] = row["keywords"]
    #     else:
    #         print("ROW KEY =======",row['RowKey'])

    #output_data = list(merged_dict.values())
    return  process_timeline_data(timelinedata)
##########################################################
def process_keywords(data):
    result = defaultdict(lambda: {"state": set(), "count": 0, "stakeholder": set(), "name": set()})

    if isinstance(data, dict):
        data = list(data.values())

    if not isinstance(data, list):
        raise ValueError("Invalid input format: Expected a list or dictionary of records.")

    for item in data:
        if not isinstance(item, dict):
            continue  # Skip invalid records

        keywords = item.get("keywords", "[]")
        if isinstance(keywords, str):  
            try:
                keywords = json.loads(keywords)  
            except json.JSONDecodeError:
                keywords = []  

        if not isinstance(keywords, list):
            keywords = []

        state = item.get("State", None)
        stakeholder = item.get("Stakeholder", "")

        if isinstance(stakeholder, float) and pd.isna(stakeholder):
            stakeholder = ""

        stakeholder = stakeholder.strip()

        for keyword in keywords:
            result[keyword]["count"] += 1
            result[keyword]["stakeholder"].add(stakeholder)

            # Ensure state is a valid string
            if isinstance(state, float) and pd.isna(state):
                state = None
            
            if state is not None and state not in ["NaN", "None", "null", ""]:
                result[keyword]["state"].add(state)

    for key in result:
        result[key]["state"] = list(result[key]["state"])
        result[key]["stakeholder"] = list(result[key]["stakeholder"])
        result[key]["name"] = list(result[key]["name"])

    return result

def geomap_data(tablename):
    table1_name = tablename
    table2_name = f"{tablename}sourcenode"

    table1_columns = ["keywords", "State", "Region", "Stakeholder"]
    geodata = fetch_table_data(table1_name, table1_columns) or []
    df = pd.DataFrame(geodata)
    
    # Replace NaN or empty strings with None (JSON null)
    df = df.where(pd.notna(df), None)

    # Ensure State is a string, and remove leading/trailing spaces
    if "State" in df:
        df["State"] = df["State"].astype(str).str.strip()
        df["State"] = df["State"].replace("nan", None)  # Replace string 'nan' with None

    # Convert back to a JSON-safe format
    cleaned_data = df.to_dict(orient="records")

    return json.dumps({"data": cleaned_data}, default=str, allow_nan=False)

def sunbrust_data(tablid):
    table1_name = tablid

    table1_columns = ["keywords", "State", "Region", "Stakeholder","theme","issue"]
    sunbrustdata = fetch_table_data(table1_name, table1_columns) or []

    region_dict = defaultdict(lambda: defaultdict(list))

    for entry in sunbrustdata:
        region = entry["Region"]
        state = entry["State"] if entry["State"] not in ["NaN", np.nan, None] else "Other"
        keywords = json.loads(entry["keywords"]) if entry["keywords"] not in ["[]", "nan"] else ["no"]
        theme = json.loads(entry["theme"]) if entry["theme"] not in ["N/A", "NaN"] else ["no"]
        issue = entry["issue"] if entry["issue"] not in ["NaN", None] else "no"
        stakeholder = entry["Stakeholder"] if entry["Stakeholder"] not in ["NaN", None] else "no"
        
        region_dict[region][state].append({
            "keywords": keywords,
            "keywords_count": len(keywords),
            "stakeholder": stakeholder,
            "theme": theme,
            "issue": issue
        })

    # Constructing final nested dictionary
    final_structure = {"name": "Region", "children": []}

    for region, states in region_dict.items():
        region_node = {"name": region, "children": []}
        for state, entries in states.items():
            state_node = {"name": state, "children": entries}
            region_node["children"].append(state_node)
        final_structure["children"].append(region_node)

    #print(sunbrustdata)
    return final_structure
