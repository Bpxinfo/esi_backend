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

def sanitize_property_name(name):
    # Replace spaces with underscores and remove any non-alphanumeric characters
    sanitized_name = re.sub(r'[^A-Za-z0-9_]', '', name)
    return sanitized_name

# Insert a list of JSON data with sanitized property names
def insert_json_data_list(json_data_list, table_name):
    """Inserts processed JSON data into an Azure Table Storage dynamically."""
    print(json_data_list)
    table_client = get_table_client(table_name)

    for index, json_data in enumerate(json_data_list):
        if isinstance(json_data, str):
            json_data = json.loads(json_data)  # Convert from JSON string to dictionary

        partition_key = f"Partition{index + 1}"  
        row_key = str(index + 1)  

        entity = {
            "PartitionKey": partition_key,
            "RowKey": row_key,
        }

        for key, value in json_data.items():
            sanitized_key = sanitize_property_name(key)

            if isinstance(value, datetime):
                if value.tzinfo is None:
                    value = pytz.utc.localize(value)
                entity[sanitized_key] = value.isoformat()  # Convert datetime to string

            elif isinstance(value, list) or isinstance(value, dict):
                entity[sanitized_key] = json.dumps(value)  # Convert list/dict to JSON string

            elif isinstance(value, (int, float, str)):  
                entity[sanitized_key] = value  # Store directly if it's a supported type

            else:
                entity[sanitized_key] = str(value)  # Convert unsupported types to string

        # Insert entity into the table
        try:
            table_client.create_entity(entity=entity)
            print(f"Inserted RowKey '{row_key}' successfully.")
        except Exception as e:
            print(f"Failed to insert RowKey '{row_key}': {e}")

######################################################
#get all data from graphData table from azure.
def getGraphData():
    try:
        # Get the table client
        table_client = get_table_client("GraphData") 
        # Query all entities in the table
        all_records = list(table_client.list_entities())
        print(all_records)
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
            for key in entity.keys(): #,'RowKey',
                if key not in [ 'PartitionKey', 'Timestamp']:
                    column_names.add(key)  # Collect column names dynamically
    except Exception as e:
        print(f"Error fetching column names from {tablename}: {e}")
    
    return list(column_names)

# Function to fetch data from a table with selected columns
def fetch_table_data_2(table_name, select_columns, query_filter=None):
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
def fetch_table_data(table_name,rename_keywords=False):
    table_client = get_table_client(table_name)
    data = {}
    for row in table_client.list_entities():
        row_key = row["RowKey"]
        row_data = dict(row)

        # Rename "keywords" column if fetching from Table1
        if rename_keywords and "keywords" in row_data:
            row_data["keywords_by_ml_model"] = row_data.pop("keywords")

        data[row_key] = row_data
    return data

def get_TableData(tablename):

    table1_name = tablename
    #table2_name = f"{tablename}sourcenode"

    entities = getallcolumnname(table1_name)
    table1_data = fetch_table_data_2(table1_name,entities)
# #    table2_data = fetch_table_data(table2_name,rename_keywords=False)
#     merged_data = {}

#     for row_key in set(table1_data.keys()).union(table2_data.keys()):
#         merged_data[row_key] = {**table1_data.get(row_key, {}), **table2_data.get(row_key, {})}

#     # Convert merged data to list format
#     merged_list = [{"RowKey": k, **v} for k, v in merged_data.items()]


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
    #print(input_data)
    for entry in input_data.values():  # Extract dictionary values

        group = json.loads(entry['group'])  # Convert string to list
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
    table2_name = f"{tablename}sourcenode"
    # # Fetch columns dynamically from both tables
    table2_columns = ["nodeid","label","relationship","source","target","group"]
    network_data = fetch_table_data(table1_name, table2_columns)

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
    table2_name = f"{tablename}sourcenode"
    
    # Fetch only the "keywords" column from the table
    table1_columns = ["keywords"]
    network_data = fetch_table_data_2(table1_name, table1_columns)  # List of dictionaries

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
    table2_name = f"{tablename}sourcenode"
    
    # Fetch only the "keywords" column from the table
    table1_columns = ["Feedback_Quotes_From_Stakeholders"]
    network_data = fetch_table_data_2(table1_name, table1_columns)  # List of dictionaries
    print(network_data)
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
    table2_name = f"{tablename}sourcenode"
    # # Fetch columns dynamically from both tables
    table2_columns = ["keywords"]
    network_data = fetch_table_data(table2_name, table2_columns)

    return cleanedkeywords(network_data)

#####################################################
# Function to transform data
def transform_treemap_data(data):
    result = {}
    print(data)
    for entry in data:
        stakeholder = entry.get("Stakeholder")  # Ensure the key is in lowercase
        keywords = entry.get("keywords")
        
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

def heatmap_chatdata(tablename):
    table1_name = tablename
    table2_name = f"{tablename}sourcenode"
    
    # Fetch only the required columns from the tables
    table1_columns = ["State","City","theme_statement","issue","keywords","Stakeholder","RowKey",'RTLL','Region','Speciality']
    heatmapedata = fetch_table_data_2(table1_name, table1_columns)  # List of dictionaries

    # table2_columns = ["keywords","RowKey"]
    # heatmapedata2 = fetch_table_data(table2_name, table2_columns)  # List of dictionaries

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
    # #print("OUTPUT DATA  -",output_data)
    #output = transform_treemap_data(heatmapedata)
    # print("FINAL DATA  -",output)
    return heatmapedata

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
    table2_name = f"{tablename}sourcenode"
    
    # Fetch only the required columns from the tables
    table1_columns = ["Date","RowKey","keywords"]
    timelinedata = fetch_table_data_2(table1_name, table1_columns)  # List of dictionaries

    #table2_columns = ["keywords","RowKey"]
    #timelinedata2 = fetch_table_data_2(table2_name, table2_columns)  # List of dictionaries

    # merged_dict = {}

    # # Merge the data from both tables based on RowKey
    # for row in timelinedata:
    #     merged_dict[row["RowKey"]] = {"Date": row["Date"]}
        
    # for row in timelinedata2:
    #     if row["RowKey"] in merged_dict:
    #         merged_dict[row["RowKey"]]["keywords"] = row["keywords"]
    #     else:
    #         print("ROW KEY =======",row['RowKey'])

    # output_data = list(merged_dict.values())
    return  process_timeline_data(timelinedata)
##########################################################

def process_keywords(data):
    result = defaultdict(lambda: {"state": set(), "count": 0, "stakeholder": set(), "name": set()})

    # Ensure data is iterable (list of dictionaries)
    if isinstance(data, dict):
        data = data.values()  # Extract values if data is a dictionary

    if not isinstance(data, list):
        raise ValueError("Invalid input format: Expected a list or dictionary of records.")

    for item in data:
        if not isinstance(item, dict):
            continue  # Skip invalid records

        keywords = item.get("keywords", [])
        if not isinstance(keywords, list):  # Ensure keywords is a list
            keywords = []

        city = item.get("name", "Unknown City")
        state = item.get("state", None)
        stakeholder = item.get("stakeholder", "")

        # Handle NaN values properly
        if pd.isna(stakeholder):
            stakeholder = ""

        stakeholder = stakeholder.strip()

        for keyword in keywords:
            result[keyword]["count"] += 1
            result[keyword]["stakeholder"].add(stakeholder)  # Use .add() instead of .append()
            result[keyword]["state"].add(state)
            result[keyword]["name"].add(city)

    # Convert sets to lists
    for key in result:
        result[key]["state"] = list(result[key]["state"])
        result[key]["stakeholder"] = list(result[key]["stakeholder"])
        result[key]["name"] = list(result[key]["name"])

    
    return result

def geomap_data(tablename):

    table1_name = tablename
    table2_name = f"{tablename}sourcenode"
    
    table1_columns = ["sentiment","keywords","theme_statement","issue","State", "Region", "RowKey", "Stakeholder", "RTLL", "Speciality"]
    geodata = fetch_table_data_2(table1_name, table1_columns)
    
    # table2_columns = ["keywords", "RowKey"]
    # timelinedata2 = fetch_table_data(table2_name, table2_columns)
    
    # merged_dict = {}
    
    # for row in timelinedata:
    #     row_key = row.get("RowKey")
    #     if not row_key:
    #         continue  # Skip invalid rows
        
    #     state = row.get("State", None)
    #     city = row.get("City", "Unknown City")
    #     rtll = row.get("RTLL", None)
    #     stakeholder = row.get("Stakeholder", None)
    #     speciality = row.get("Speciality", None)
        
    #     # Handle NaN values
    #     for key in ["RTLL", "Stakeholder", "Speciality"]:
    #         if row.get(key) is not None and isinstance(row[key], float) and math.isnan(row[key]):
    #             row[key] = None
        
    #     # Ensure state is a string and convert to uppercase
    #     if isinstance(state, str):
    #         state = state.upper()
    #     else:
    #         state = None
        
    #     lat_long = get_coordinates(state) if state else {"lat": None, "long": None, "name": ''}
        
    #     merged_dict[row_key] = {
    #         "name": city,
    #         "rtll": rtll,
    #         "state": lat_long.get("name", " "),
    #         "stakeholder": stakeholder,
    #         "speciality": speciality,
    #         "lat": lat_long.get("lat", None),
    #         "lng": lat_long.get("long", None),
    #         "keywords": []
    #     }
    # #print(merged_dict)
    # for row in timelinedata2:
    #     row_key = row.get("RowKey")
    #     if row_key in merged_dict:
    #         keywords = row.get("keywords", "[]")
    #         if isinstance(keywords, str):
    #             try:
    #                 keywords = json.loads(keywords)
    #             except json.JSONDecodeError:
    #                 keywords = []
    #         merged_dict[row_key]["keywords"] = keywords
    #     else:
    #         print("Missing RowKey in table1:", row_key)
    
    # #
    # output_data2 = list(merged_dict.values())

    #return process_keywords(output_data2)
    return geodata

def sunbrust_data(tablename):
    table1_name = tablename
    #table2_name = f"{tablename}sourcenode"
    
    table1_columns = ["keywords","theme_statement","issue","State", "Region", "Stakeholder"]
    sunbrustdata = fetch_table_data_2(table1_name, table1_columns)
    return sunbrustdata
 
def sunbrust_data_2(tablename):
    table1_name = tablename
    #table2_name = f"{tablename}sourcenode"
    
    table1_columns = ["keywords","theme_statement","issue","State", "theme_statement", "Stakeholder"]
    sunbrustdata = fetch_table_data_2(table1_name, table1_columns)
    return sunbrustdata


