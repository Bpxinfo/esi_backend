state_coordinates = {
    "AL": {"name": "Alabama", "lat": 32.806671, "long": -86.791130},
    "AK": {"name": "Alaska", "lat": 61.370716, "long": -152.404419},
    "AZ": {"name": "Arizona", "lat": 33.729759, "long": -111.431221},
    "AR": {"name": "Arkansas", "lat": 34.969704, "long": -92.373123},
    "CA": {"name": "California", "lat": 36.116203, "long": -119.681564},
    "CO": {"name": "Colorado", "lat": 39.059811, "long": -105.311104},
    "CT": {"name": "Connecticut", "lat": 41.597782, "long": -72.755371},
    "DE": {"name": "Delaware", "lat": 39.318523, "long": -75.507141},
    "DC": {"name": "District of Columbia", "lat": 38.897438, "long": -77.026817},
    "FL": {"name": "Florida", "lat": 27.766279, "long": -81.686783},
    "GA": {"name": "Georgia", "lat": 33.040619, "long": -83.643074},
    "HI": {"name": "Hawaii", "lat": 21.094318, "long": -157.498337},
    "ID": {"name": "Idaho", "lat": 44.240459, "long": -114.478828},
    "IL": {"name": "Illinois", "lat": 40.349457, "long": -88.986137},
    "IN": {"name": "Indiana", "lat": 39.849426, "long": -86.258278},
    "IA": {"name": "Iowa", "lat": 42.011539, "long": -93.210526},
    "KS": {"name": "Kansas", "lat": 38.526600, "long": -96.726486},
    "KY": {"name": "Kentucky", "lat": 37.668140, "long": -84.670067},
    "LA": {"name": "Louisiana", "lat": 31.169546, "long": -91.867805},
    "ME": {"name": "Maine", "lat": 44.693947, "long": -69.381927},
    "MD": {"name": "Maryland", "lat": 39.063946, "long": -76.802101},
    "MA": {"name": "Massachusetts", "lat": 42.230171, "long": -71.530106},
    "MI": {"name": "Michigan", "lat": 43.326618, "long": -84.536095},
    "MN": {"name": "Minnesota", "lat": 45.694454, "long": -93.900192},
    "MS": {"name": "Mississippi", "lat": 32.741646, "long": -89.678696},
    "MO": {"name": "Missouri", "lat": 38.456085, "long": -92.288368},
    "MT": {"name": "Montana", "lat": 46.921925, "long": -110.454353},
    "NE": {"name": "Nebraska", "lat": 41.125370, "long": -98.268082},
    "NV": {"name": "Nevada", "lat": 38.313515, "long": -117.055374},
    "NH": {"name": "New Hampshire", "lat": 43.452492, "long": -71.563896},
    "NJ": {"name": "New Jersey", "lat": 40.298904, "long": -74.521011},
    "NM": {"name": "New Mexico", "lat": 34.840515, "long": -106.248482},
    "NY": {"name": "New York", "lat": 42.165726, "long": -74.948051},
    "NC": {"name": "North Carolina", "lat": 35.630066, "long": -79.806419},
    "ND": {"name": "North Dakota", "lat": 47.528912, "long": -99.784012},
    "OH": {"name": "Ohio", "lat": 40.388783, "long": -82.764915},
    "OK": {"name": "Oklahoma", "lat": 35.565342, "long": -96.928917},
    "OR": {"name": "Oregon", "lat": 44.572021, "long": -122.070938},
    "PA": {"name": "Pennsylvania", "lat": 40.590752, "long": -77.209755},
    "RI": {"name": "Rhode Island", "lat": 41.680893, "long": -71.511780},
    "SC": {"name": "South Carolina", "lat": 33.856892, "long": -80.945007},
    "SD": {"name": "South Dakota", "lat": 44.299782, "long": -99.438828},
    "TN": {"name": "Tennessee", "lat": 35.747845, "long": -86.692345},
    "TX": {"name": "Texas", "lat": 31.054487, "long": -97.563461},
    "UT": {"name": "Utah", "lat": 40.150032, "long": -111.862434},
    "VT": {"name": "Vermont", "lat": 44.045876, "long": -72.710686},
    "VA": {"name": "Virginia", "lat": 37.769337, "long": -78.169968},
    "WA": {"name": "Washington", "lat": 47.400902, "long": -121.490494},
    "WV": {"name": "West Virginia", "lat": 38.491226, "long": -80.954456},
    "WI": {"name": "Wisconsin", "lat": 44.268543, "long": -89.616508},
    "WY": {"name": "Wyoming", "lat": 42.755966, "long": -107.302490}
}

def get_coordinates(state_code):
    state_code = state_code.upper()
    if state_code in state_coordinates:
        state = state_coordinates[state_code]
        return state
    # else:
    #     return "State code not found."
    # else:
    #     return "Invalid state code."


