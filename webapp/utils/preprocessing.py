from geopy.geocoders import Nominatim
# import numpy as np

# Encode the location to latitude and longitude
geolocator = Nominatim(user_agent="Python3.9.23")
def get_coord_lat_lon(full_addr: str):
    pt = geolocator.geocode(full_addr)
    print(pt)
    return (pt.latitude, pt.longitude) if pt else (30.045965, 31.247196) # dummy default values

# Categorize the level of the house based on floor number
def categorize_level(level):
    if level in [0,1,2,3]:
        level = 0
    elif level in [4,5,6,7]:
        level = 1
    else:
        level = 2
    return level

# score based on level category and elevator presence
def accessibility_score(level_cat, elevator):
    if level_cat == 0:
        score = 0 # high score
    elif elevator == 1 and level_cat == 1:
        score = 1 # mid score
    else:
        score = 2 # low score
    return score

def bathtobed_ratio(bathrooms, bedrooms):
    if bedrooms == 0:
        return 0
    return bathrooms / bedrooms