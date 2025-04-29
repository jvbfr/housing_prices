import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
from tqdm import tqdm
import requests
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath('../'))
# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def load_data():
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    # Navigate to the project root and then into the data directory
    data_path = script_dir.parent / 'data' / 'housing_labeled.csv'
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    return df

def reverse_geocode_batch(df, batch_size=10):
    """Reverse geocode latitude/longitude to get neighborhood and city information."""
    logger.info("Starting reverse geocoding process")
    geolocator = Nominatim(user_agent="avm_model_geocoder", timeout=5)
    results = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Reverse Geocoding"):
        batch = df.iloc[i:min(i+batch_size, len(df))]
        batch_results = []
        
        for _, row in batch.iterrows():
            try:
                location = geolocator.reverse((row['latitude'], row['longitude']), exactly_one=True)
                if location and location.raw.get('address'):
                    address = location.raw['address']
                    neighborhood = address.get('neighbourhood', 
                                     address.get('suburb', 
                                     address.get('hamlet', None)))
                    city = address.get('city', 
                           address.get('town', 
                           address.get('village', None)))
                    county = address.get('county', None)
                    state = address.get('state', None)
                    postcode = address.get('postcode', None)
                    batch_results.append({
                        'neighborhood': neighborhood,
                        'city': city,
                        'county': county,
                        'state': state,
                        'postcode': postcode
                    })
                else:
                    batch_results.append({
                        'neighborhood': None,
                        'city': None,
                        'county': None,
                        'state': None,
                        'postcode': None
                    })
            except Exception as e:
                logger.warning(f"Error in reverse geocoding: {e}")
                batch_results.append({
                    'neighborhood': None,
                    'city': None,
                    'county': None,
                    'state': None,
                    'postcode': None
                })
            
            # Sleep to respect API rate limits
            time.sleep(1)
        
        results.extend(batch_results)
    
    # Convert results to DataFrame and join with original
    geo_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), geo_df], axis=1)

def get_nearby_amenities(lat, lon, radius=1000, cache_dir='amenities_cache'):
    """Get nearby amenities using Overpass API with caching to reduce API calls."""
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a cache key based on location and radius (rounded to reduce cache misses)
    lat_rounded = round(lat, 4)
    lon_rounded = round(lon, 4)
    cache_key = f"{lat_rounded}_{lon_rounded}_{radius}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    # Check if we have cached results
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    amenities = {}
    
    # Define amenity types we're interested in
    amenity_categories = {
        "education": ["school", "university", "college", "kindergarten", "library"],
        "healthcare": ["hospital", "pharmacy", "clinic", "doctors", "dentist"],
        "shopping": ["supermarket", "convenience", "department_store", "mall"],
        "food": ["restaurant", "cafe", "fast_food", "bar"],
        "transportation": ["bus_station", "train_station", "subway_station", "parking"],
        "recreation": ["park", "sports_centre", "swimming_pool", "beach"],
        "financial": ["bank", "atm"],
        "services": ["post_office", "police", "fire_station", "townhall"]
    }
    
    # Flatten the categories
    all_amenities = []
    for category, types in amenity_categories.items():
        all_amenities.extend(types)
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Build comprehensive query for amenities
    query_parts = []
    
    # Query for amenities
    query_parts.append(f"""
        node[amenity](around:{radius},{lat},{lon});
        out center;
    """)
    
    # Query for leisure areas like parks
    query_parts.append(f"""
        (
          node["leisure"](around:{radius},{lat},{lon});
          way["leisure"](around:{radius},{lat},{lon});
          relation["leisure"](around:{radius},{lat},{lon});
        );
        out center;
    """)
    
    # Query for shops
    query_parts.append(f"""
        (
          node["shop"](around:{radius},{lat},{lon});
          way["shop"](around:{radius},{lat},{lon});
        );
        out center;
    """)
    
    # Query for natural features (beaches, etc.)
    query_parts.append(f"""
        (
          node["natural"](around:{radius},{lat},{lon});
          way["natural"](around:{radius},{lat},{lon});
          relation["natural"](around:{radius},{lat},{lon});
        );
        out center;
    """)
    
    # Combine queries
    full_query = "[out:json];" + "".join(query_parts)
    
    try:
        response = requests.get(overpass_url, params={"data": full_query}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            elements = data.get("elements", [])
            
            # Process results by category and type
            for element in elements:
                if "tags" in element:
                    tags = element["tags"]
                    
                    # Get coordinates
                    if "lat" in element and "lon" in element:
                        element_lat = element["lat"]
                        element_lon = element["lon"]
                    elif "center" in element:
                        element_lat = element["center"]["lat"]
                        element_lon = element["center"]["lon"]
                    else:
                        continue
                    
                    # Calculate distance
                    distance = geodesic((lat, lon), (element_lat, element_lon)).meters
                    
                    # Process each type of amenity
                    for key in tags:
                        if key == "amenity" and tags[key] in all_amenities:
                            amenity_type = tags[key]
                            if amenity_type not in amenities:
                                amenities[amenity_type] = []
                            amenities[amenity_type].append({
                                "distance": distance,
                                "lat": element_lat,
                                "lon": element_lon,
                                "name": tags.get("name")
                            })
                        elif key == "leisure" and tags[key] == "park":
                            if "park" not in amenities:
                                amenities["park"] = []
                            amenities["park"].append({
                                "distance": distance,
                                "lat": element_lat,
                                "lon": element_lon,
                                "name": tags.get("name")
                            })
                        elif key == "shop":
                            shop_type = tags[key]
                            if shop_type not in amenities:
                                amenities[shop_type] = []
                            amenities[shop_type].append({
                                "distance": distance,
                                "lat": element_lat,
                                "lon": element_lon,
                                "name": tags.get("name")
                            })
                        elif key == "natural" and tags[key] == "beach":
                            if "beach" not in amenities:
                                amenities["beach"] = []
                            amenities["beach"].append({
                                "distance": distance,
                                "lat": element_lat,
                                "lon": element_lon,
                                "name": tags.get("name")
                            })
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(amenities, f)
            
            return amenities
        else:
            logger.warning(f"Overpass API returned status code {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error querying Overpass API: {e}")
        return {}

def calculate_nearest_amenities(df, batch_size=10, radius=2000):
    """Calculate distances to the nearest amenities for each location."""
    logger.info("Starting amenity distance calculations")
    
    # Define the amenities we're most interested in for AVM
    priority_amenities = [
        "school", "university", "college", "hospital", "supermarket", 
        "convenience", "department_store", "restaurant", "cafe", 
        "park", "bus_station", "train_station", "subway_station", 
        "beach", "bank", "police", "fire_station"
    ]
    
    # Initialize columns for nearest distances
    for amenity in priority_amenities:
        df[f'distance_to_{amenity}'] = np.nan
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing Amenities"):
        batch = df.iloc[i:min(i+batch_size, len(df))]
        
        for idx, row in batch.iterrows():
            try:
                amenities = get_nearby_amenities(row['latitude'], row['longitude'], radius=radius)
                
                # Find nearest distance for each amenity type
                for amenity_type, amenity_list in amenities.items():
                    if amenity_type in priority_amenities and amenity_list:
                        min_distance = min(item["distance"] for item in amenity_list)
                        df.at[idx, f'distance_to_{amenity_type}'] = min_distance
                
                # Add generic retail and shopping distances
                if "supermarket" in amenities or "convenience" in amenities or "department_store" in amenities:
                    all_retail = []
                    for retail_type in ["supermarket", "convenience", "department_store"]:
                        if retail_type in amenities:
                            all_retail.extend(amenities[retail_type])
                    
                    if all_retail:
                        min_retail_distance = min(item["distance"] for item in all_retail)
                        df.at[idx, 'distance_to_retail'] = min_retail_distance
                
                # Add generic education distances
                if "school" in amenities or "university" in amenities or "college" in amenities:
                    all_education = []
                    for edu_type in ["school", "university", "college"]:
                        if edu_type in amenities:
                            all_education.extend(amenities[edu_type])
                    
                    if all_education:
                        min_edu_distance = min(item["distance"] for item in all_education)
                        df.at[idx, 'distance_to_education'] = min_edu_distance
                
            except Exception as e:
                logger.error(f"Error processing amenities for index {idx}: {e}")
            
            # Sleep briefly to avoid overwhelming the API
            time.sleep(0.1)
    
    return df

def calculate_major_city_distances(df):
    """Calculate distances to major California cities."""
    logger.info("Calculating distances to major California cities")
    
    # Define major California cities with their coordinates
    major_cities = {
        'Los_Angeles': (34.0522, -118.2437),
        'San_Francisco': (37.7749, -122.4194),
        'San_Diego': (32.7157, -117.1611),
        'Sacramento': (38.5816, -121.4944),
        'San_Jose': (37.3382, -121.8863),
        'Fresno': (36.7378, -119.7871),
        'Oakland': (37.8044, -122.2711),
        'Bakersfield': (35.3733, -119.0187),
        'Irvine': (33.6846, -117.8265),
        'Riverside': (33.9806, -117.3755)
    }
    
    # Calculate distances to each city
    for city_name, (city_lat, city_lon) in tqdm(major_cities.items(), desc="City Distances"):
        df[f'distance_to_{city_name}'] = df.apply(
            lambda row: geodesic(
                (row['latitude'], row['longitude']), 
                (city_lat, city_lon)
            ).kilometers,
            axis=1
        )
    
    # Calculate distance to nearest major city
    city_distance_cols = [f'distance_to_{city}' for city in major_cities.keys()]
    df['distance_to_nearest_major_city'] = df[city_distance_cols].min(axis=1)
    df['nearest_major_city'] = df[city_distance_cols].idxmin(axis=1).apply(lambda x: x.replace('distance_to_', ''))
    
    return df

def create_additional_features(df):
    """Create additional features based on amenity distances."""
    logger.info("Creating additional features")
    
    # Get all amenity distance columns
    amenity_cols = [col for col in df.columns if col.startswith('distance_to_') and 
                   not any(city in col for city in ['Los_Angeles', 'San_Francisco', 'San_Diego', 
                                                   'Sacramento', 'San_Jose', 'Fresno', 'Oakland', 
                                                   'Bakersfield', 'Irvine', 'Riverside'])]
    
    # Fill missing values with a large number (indicating not available)
    for col in amenity_cols:
        max_dist = df[col].dropna().max() if not df[col].dropna().empty else 10000
        df[col].fillna(max_dist * 2, inplace=True)
    
    # Create binary indicators for nearby amenities
    distances = [500, 1000, 2000]  # 500m, 1km, 2km
    
    for col in amenity_cols:
        amenity_type = col.replace('distance_to_', '')
        for dist in distances:
            df[f'has_{amenity_type}_within_{dist}m'] = (df[col] <= dist).astype(int)
    
    # Count number of different amenities within certain distances
    for dist in distances:
        binary_cols = [col for col in df.columns if col.endswith(f'within_{dist}m')]
        if binary_cols:
            df[f'amenity_count_{dist}m'] = df[binary_cols].sum(axis=1)
            df[f'amenity_diversity_{dist}m'] = df[binary_cols].gt(0).sum(axis=1)
    
    # Create walkability score (simplified version)
    # Higher score means more amenities are within walking distance
    walk_weights = {
        500: 1.0,   # Full weight for amenities within 500m
        1000: 0.7,  # 70% weight for amenities within 1km
        1500: 0.5,
        2000: 0.3,   # 30% weight for amenities within 2km
        3000: 0.1
    }
    
    df['walkability_score'] = sum(
        df[f'amenity_count_{dist}m'] * weight 
        for dist, weight in walk_weights.items() if f'amenity_count_{dist}m' in df.columns
    )
    
    # Add amenity data completeness as a feature
    df['amenity_data_completeness'] = df[amenity_cols].notna().mean(axis=1)
    
    # Calculate average distance to key amenities
    key_amenities = ['school', 'park', 'supermarket', 'bank', 'restaurant']
    key_cols = [f'distance_to_{a}' for a in key_amenities if f'distance_to_{a}' in df.columns]
    if key_cols:
        df['avg_distance_to_key_amenities'] = df[key_cols].mean(axis=1)
    
    # Create a transportation accessibility score
    transport_cols = [col for col in df.columns if 'bus_station' in col or 
                                                 'train_station' in col or 
                                                 'subway_station' in col]
    if transport_cols:
        # Normalize distances and invert (closer is better)
        for col in transport_cols:
            if col.startswith('distance_to_'):
                max_val = df[col].max()
                if max_val > 0:
                    df[f'{col}_norm'] = 1 - (df[col] / max_val)
        
        norm_cols = [col for col in df.columns if col.endswith('_norm')]
        if norm_cols:
            df['transportation_accessibility'] = df[norm_cols].mean(axis=1) * 10  # Scale 0-10
    
    return df

def main():
    """Main function to process data."""
    # This assumes your data is in a CSV file with latitude and longitude columns
    file_path = "../data/housing_labeled.csv"  # Replace with your actual file path
    output_path = "housing_geo.csv"
    
    try:
        # Load data
        df = load_data()
        
        # Add reverse geocoding information
        df = reverse_geocode_batch(df, batch_size=10)
        
        # Calculate distances to amenities
        df = calculate_nearest_amenities(df, batch_size=5, radius=2000)
        
        # Calculate distances to major cities
        df = calculate_major_city_distances(df)
        
        # Create additional features
        df = create_additional_features(df)
        
        # Save the enriched dataset
        df.to_csv(output_path, index=False)
        logger.info(f"Enriched dataset saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise

if __name__ == "__main__":
    main()