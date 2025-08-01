import os
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MiniMap
from sklearn.cluster import DBSCAN
import unidecode
import webbrowser
import csv
import json

# Path constants
GENERATED_DIR = "generated_data"
CONTAMINANTS_DIR = os.path.join(GENERATED_DIR, "contaminants")
MAPS_DIR = os.path.join(GENERATED_DIR, "maps_cluster")
METADATA_FILE = "metadata.csv"
CONTAM_PATH = "contaminants data"
CACHE_FILE = "cache_geo.csv"
pattern = "tasf-thgu-"

os.makedirs(CONTAMINANTS_DIR, exist_ok=True)
os.makedirs(MAPS_DIR, exist_ok=True)

def load_metadata():
    metadata = pd.read_csv(METADATA_FILE)
    metadata['city_norm'] = metadata['city'].astype(str).str.lower().apply(unidecode.unidecode)
    return metadata

def load_city_coords():
    if os.path.exists(CACHE_FILE):
        cities = pd.read_csv(CACHE_FILE)
    else:
        cities = pd.DataFrame(columns=["ville", "lat", "lon"])
    cities['ville_norm'] = cities['ville'].astype(str).str.lower().apply(unidecode.unidecode)
    return cities.set_index('ville_norm')[['lat', 'lon']].to_dict(orient='index')

def compute_heatmap(contaminant, city_coords):
    heat_data = []
    for fn in os.listdir(CONTAM_PATH):
        if fn.startswith(pattern) and fn.endswith('.csv'):
            df = pd.read_csv(os.path.join(CONTAM_PATH, fn))
            if 'municipi' in df.columns and 'contaminant' in df.columns:
                df_c = df[df['contaminant'] == contaminant]
                hour_cols = [f"h{i}" for i in range(1, 25) if f"h{i}" in df_c.columns]
                for _, row_c in df_c.iterrows():
                    city_c = str(row_c['municipi']).strip().lower()
                    city_c_norm = unidecode.unidecode(city_c)
                    if city_c_norm in city_coords and hour_cols:
                        lat_c = city_coords[city_c_norm]['lat']
                        lon_c = city_coords[city_c_norm]['lon']
                        concs = [row_c[h] for h in hour_cols if not pd.isna(row_c[h])]
                        if concs:
                            avg_conc = sum(concs) / len(concs)
                            if all(isinstance(v, (int, float)) and not pd.isna(v) for v in [lat_c, lon_c, avg_conc]):
                                heat_data.append([float(lat_c), float(lon_c), float(avg_conc)])
    heatmap = heat_data
    return heatmap

def is_red_zone(center_lat, center_lon, heatmap_points, threshold=0.6):
    if not heatmap_points:
        return False
    max_conc = max(pt[2] for pt in heatmap_points)
    for pt in heatmap_points:
        lat, lon, conc = pt
        dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
        if dist < 0.05 and conc >= threshold * max_conc:
            return True
    return False

def find_best_match(city_name, available_cities, threshold=0.8):
    """Find the best match for a city name"""
    from difflib import SequenceMatcher
    
    best_match = None
    best_score = 0
    
    for available_city in available_cities:
        # Similarity score
        score = SequenceMatcher(None, city_name, available_city).ratio()
        
        # Bonus if one contains the other
        if city_name in available_city or available_city in city_name:
            score += 0.1
            
        # Bonus for exact word matches
        city_words = set(city_name.replace("'", "").replace("-", " ").split())
        available_words = set(available_city.replace("'", "").replace("-", " ").split())
        if city_words & available_words:  # non-empty intersection
            score += 0.1
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = available_city
    
    return best_match, best_score

def generate_map_and_html_1(contaminant, disease, open_browser=False):
    print(f"[INFO] Génération de la carte pour contaminant='{contaminant}', maladie='{disease}'")

    # Chargement des métadonnées
    metadata = load_metadata()
    patients = metadata[metadata['cod'] == disease]

    if patients.empty:
        print(f"[WARN] Aucun patient trouvé pour la maladie '{disease}'")
        return None

    # Chargement et normalisation des données géographiques
    geo_cache_df = pd.read_csv("data-csv/cache_geo.csv")

    def normalize_city_name(city_name):
        city_name = str(city_name).strip()
        if ', ' in city_name:
            parts = city_name.split(', ')
            if len(parts) == 2:
                main_name = parts[0].strip()
                determiner = parts[1].strip().lower()
                city_name = f"{determiner} {main_name}"
        return unidecode.unidecode(city_name.lower())

    city_coords = {}
    for _, row in geo_cache_df.iterrows():
        city = normalize_city_name(row['city_name'] if 'city_name' in row else row[0])
        lat = row['lat'] if 'lat' in row else row[1]
        lon = row['lon'] if 'lon' in row else row[2]
        if not pd.isna(lat) and not pd.isna(lon):
            city_coords[city] = {'lat': lat, 'lon': lon}

    print(f"[INFO] {len(city_coords)} villes avec coordonnées chargées.")

    # Calcul de la heatmap
    heatmap = compute_heatmap(contaminant, city_coords)
    print(f"[INFO] {len(heatmap)} points pour la heatmap '{contaminant}'")

    # Extraction des coordonnées des patients
    coords = []
    city_col = next((c for c in patients.columns if "city" in c.lower()), None)
    if city_col is None:
        print("[ERROR] Aucune colonne contenant 'city' dans les métadonnées.")
        return None

    for _, row in patients.iterrows():
        city = normalize_city_name(row[city_col])
        if city in city_coords:
            lat = city_coords[city]['lat']
            lon = city_coords[city]['lon']
            if not pd.isna(lat) and not pd.isna(lon):
                coords.append([lat, lon])
        else:
            print(f"[WARN] Ville non trouvée dans les coordonnées : {city}")

    coords = np.array(coords)
    clusters = []

    if len(coords) > 0:
        clustering = DBSCAN(eps=0.1, min_samples=5).fit(coords)
        labels = clustering.labels_
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = coords[labels == label]
            center_lat = np.mean(cluster_points[:, 0])
            center_lon = np.mean(cluster_points[:, 1])
            count = len(cluster_points)
            dists = np.sqrt((cluster_points[:, 0] - center_lat)**2 + (cluster_points[:, 1] - center_lon)**2)
            max_dist = dists.max()
            base_radius_m = max_dist * 111_000
            radius_m = min(base_radius_m + 200 * count, 25_000)
            clusters.append((center_lat, center_lon, count, radius_m))

    # Calcul du score de contamination
    contaminant_score = 0
    for center_lat, center_lon, count, radius_m in clusters:
        if count >= 8 and is_red_zone(center_lat, center_lon, heatmap):
            contaminant_score += 1

    print(f"[INFO] Score de contamination '{contaminant}': {contaminant_score}")

    # Création de la carte
    lat0, lon0 = coords[0] if len(coords) > 0 else (41.39, 2.15)
    m = folium.Map(location=[lat0, lon0], zoom_start=8, tiles="CartoDB positron")

    # Add comarca boundaries
    with open("comarques-compressed.geojson", "r", encoding="utf-8") as f:
        comarques_geojson = json.load(f)

    def style_comarques(feature):
        return {
            'fillOpacity': 0,
            'color': 'grey',
            'weight': 1
        }

    folium.GeoJson(
        comarques_geojson,
        name="Comarques",
        style_function=style_comarques
    ).add_to(m)

    if len(heatmap) > 0:
        HeatMap(
            heatmap,
            min_opacity=0.3,
            max_opacity=0.8,
            radius=40,
            blur=25,
            gradient={'0.2': 'green', '0.5': 'yellow', '0.8': 'red'}
        ).add_to(m)

    for center_lat, center_lon, count, radius_m in clusters:
        if count >= 8:
            folium.Circle(
                location=[center_lat, center_lon],
                radius=radius_m,
                color='black',
                fill=True,
                fill_color='black',
                fill_opacity=0.2,
                popup=f"Cluster: {count} patients avec {disease}"
            ).add_to(m)

    m.add_child(MiniMap(toggle_display=True))

    # Sauvegarde
    map_file = os.path.join(MAPS_DIR, f"map_{disease}_{contaminant}.html")
    m.save(map_file)
    print(f"[INFO] Carte sauvegardée : {map_file}")

    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(map_file)}")

    return map_file


def generate_map_and_html_2(contaminant, disease):
    metadata = load_metadata()
    
    # Load city -> comarca correspondences
    # Function to correctly parse the cities_catalunya file with commas in names
    def parse_cities_catalunya(filepath):
        """
        Parse the cities_catalunya file handling commas in city names.
        Expected format: code,city_name,comarca_code,comarca_name
        Problem: city names can contain commas (e.g. "Saus, Camallera i Llampaies")
        """
        cities_data = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Split by commas
                parts = line.split(',')
                
                if len(parts) < 4:
                    continue
                
                # First element is the city code
                city_code = parts[0].strip()
                
                # Last element is the comarca name
                comarca_name = parts[-1].strip()
                
                # Second to last is the comarca code
                comarca_code = parts[-2].strip()
                
                # Everything in the middle constitutes the city name
                city_name_parts = parts[1:-2]
                city_name = ','.join(city_name_parts).strip()
                
                cities_data.append({
                    'city_code': city_code,
                    'city_name': city_name,
                    'comarca_code': comarca_code,
                    'comarca_name': comarca_name
                })
                
        return pd.DataFrame(cities_data)
    
    cities_df = parse_cities_catalunya("data-csv/cities_catalunya_modified.csv")

    # Load city coordinates
    geo_cache_df = pd.read_csv("data-csv/cache_geo.csv")
    
    # Load postal codes
    postal_df = pd.read_csv("data-csv/Codis_postals.csv")
    
    # Function to normalize city names (handle "Garriga, la" -> "la Garriga")
    def normalize_city_name(city_name):
        """
        Normalize city names handling determinants.
        Example: "Garriga, la" -> "la Garriga"
        """
        city_name = str(city_name).strip()
        
        # Handle "Name, determiner" format
        if ', ' in city_name:
            parts = city_name.split(', ')
            if len(parts) == 2:
                main_name = parts[0].strip()
                determiner = parts[1].strip().lower()
                # Put determiner in front
                city_name = f"{determiner} {main_name}"
        
        # Normalize and lowercase
        city_normalized = unidecode.unidecode(city_name.lower())
        return city_normalized
    
    # Create postal code -> city dictionary
    postal_to_city = {}
    for _, row in postal_df.iterrows():
        postal_code = str(row['Codi postal']).strip()
        city_name = normalize_city_name(row['Nom municipi'])
        postal_to_city[postal_code] = city_name
    
    for i, (postal, city) in enumerate(list(postal_to_city.items())[:10]):
        print(f"  - '{postal}' -> '{city}'")
    
    # Create city -> comarca dictionary
    city_to_comarca = {}
    
    for _, row in cities_df.iterrows():
        city = normalize_city_name(row['city_name'])
        comarca = str(row['comarca_name']).strip()
        city_to_comarca[city] = comarca
    
    for i, (city, comarca) in enumerate(list(city_to_comarca.items())[:5]):
        print(f"  - '{city}' -> '{comarca}'")
    
    # Create city -> coordinates dictionary
    city_coords = {}
    if 'city_name' in geo_cache_df.columns and 'lat' in geo_cache_df.columns and 'lon' in geo_cache_df.columns:
        city_col_geo = 'city_name'
        lat_col = 'lat'
        lon_col = 'lon'
    else:
        # Use available columns
        city_col_geo = geo_cache_df.columns[0]
        lat_col = 'lat' if 'lat' in geo_cache_df.columns else geo_cache_df.columns[1]
        lon_col = 'lon' if 'lon' in geo_cache_df.columns else geo_cache_df.columns[2]
    
    for _, row in geo_cache_df.iterrows():
        city = normalize_city_name(row[city_col_geo])
        lat = row[lat_col]
        lon = row[lon_col]
        if not pd.isna(lat) and not pd.isna(lon):
            city_coords[city] = {'lat': lat, 'lon': lon}
    
    for i, (city, coords) in enumerate(list(city_coords.items())[:5]):
        print(f"  - '{city}': ({coords['lat']:.3f}, {coords['lon']:.3f})")
    
    # Load population data by city
    pop_df = pd.read_csv("data-csv/2861-3.csv", encoding="latin1", sep=";")
    pop_df["City"] = pop_df["Municipios"].apply(lambda x: normalize_city_name(x.split(" ", 1)[1].split(",")[0].strip()))
    pop_df["Population"] = pop_df["Total"].astype(str).str.replace(".", "", regex=False).astype(int)
    population_dict = dict(zip(pop_df["City"], pop_df["Population"]))
    
    # Calculate population by comarca
    comarca_population = {}
    for city, pop in population_dict.items():
        if city in city_to_comarca:
            comarca = city_to_comarca[city]
            comarca_population[comarca] = comarca_population.get(comarca, 0) + pop
    
    for i, (comarca, pop) in enumerate(list(comarca_population.items())[:5]):
        print(f"  - '{comarca}': {pop:,} inhabitants")
    
    # Create heatmap based on city coordinates
    heatmap = compute_heatmap(contaminant, city_coords)
    
    # Collect patients by postal code
    patients = metadata[metadata['cod'] == disease]
    
    # Check that postal_code column exists
    if 'postal_code' not in patients.columns:
        return None
    
    # Dictionaries to store data by comarca
    comarca_counts = {}
    comarca_coords = {}
    comarca_cities = {}  # To track cities in each comarca
    
    postal_counts = {}  # Also count by postal code for debug
    city_counts = {}    # Count by city
    unmatched_postals = set()  # Postal codes not found
    missing_in_comarca = set()  # Cities not found in city_to_comarca
    missing_in_coords = set()   # Cities not found in city_coords
    
    # Process each patient
    for idx, row in patients.iterrows():
        postal_code = str(row['postal_code']).strip()
        
        # Find city from postal code
        if postal_code not in postal_to_city:
            unmatched_postals.add(postal_code)
            continue
        
        city_norm = postal_to_city[postal_code]
        
        # Check that city exists in our data
        if city_norm not in city_to_comarca:
            missing_in_comarca.add(city_norm)
            continue
            
        if city_norm not in city_coords:
            missing_in_coords.add(city_norm)
            continue
        
        coords = city_coords[city_norm]
        lat, lon = coords['lat'], coords['lon']
        
        # Count patients
        postal_counts[postal_code] = postal_counts.get(postal_code, 0) + 1
        city_counts[city_norm] = city_counts.get(city_norm, 0) + 1
        
        # Get comarca for this city
        comarca = city_to_comarca[city_norm]
        
        # Count patients by comarca
        comarca_counts[comarca] = comarca_counts.get(comarca, 0) + 1
        
        # Initialize structures for this comarca if necessary
        if comarca not in comarca_cities:
            comarca_cities[comarca] = set()  # Use set to avoid duplicates
        
        # Add this city to the comarca
        comarca_cities[comarca].add(city_norm)
    
    # Calculate central coordinates for each comarca
    for comarca, cities in comarca_cities.items():
        if cities:
            # Calculate geographic center of comarca (average of city coordinates)
            lats = []
            lons = []
            for city in cities:
                if city in city_coords:
                    coords = city_coords[city]
                    lat, lon = coords['lat'], coords['lon']
                    if not pd.isna(lat) and not pd.isna(lon):
                        lats.append(lat)
                        lons.append(lon)
            
            if lats and lons:
                center_lat = sum(lats)/len(lats)
                center_lon = sum(lons)/len(lons)
                comarca_coords[comarca] = [center_lat, center_lon]

    for comarca, count in sorted(list(comarca_counts.items())[:10]):
        print(f"  - {comarca}: {count} patients")
    
    # Build clusters by comarca
    clusters = []
    rejected_clusters = []
    
    for comarca, count in comarca_counts.items():
        
        if comarca not in comarca_coords:
            rejected_clusters.append((comarca, count, "no coordinates"))
            continue
            
        if comarca not in comarca_population or comarca_population[comarca] == 0:
            rejected_clusters.append((comarca, count, "no population data"))
            continue
            
        lat, lon = comarca_coords[comarca]
        pop = comarca_population[comarca]
        
        # Filter comarques with too few cases
        if count < 3:
            rejected_clusters.append((comarca, count, f"less than 3 patients ({count})"))
            continue
        
        # Calculate ratio and radius
        ratio = count / pop
        scale = 3_000_000
        radius = min(ratio * scale, 30_000)
        
        clusters.append((lat, lon, count, radius, pop, comarca))

    for comarca, count, reason in rejected_clusters[:5]:  # First 5 rejected
        print(f"  - {comarca} ({count} patients): {reason}")
    
    # Create map
    if len(clusters) > 0:
        lat0, lon0 = clusters[0][0], clusters[0][1]
    else:
        lat0, lon0 = 41.39, 2.15  # Default coordinates for Catalonia
    
    m = folium.Map(location=[lat0, lon0], zoom_start=8, tiles="CartoDB positron")
    
    # Add comarca boundaries
    with open("comarques-compressed.geojson", "r", encoding="utf-8") as f:
        comarques_geojson = json.load(f)

    def style_comarques(feature):
        return {
            'fillOpacity': 0,
            'color': 'grey',
            'weight': 1
        }

    folium.GeoJson(
        comarques_geojson,
        name="Comarques",
        style_function=style_comarques
    ).add_to(m)

    # Add heatmap
    HeatMap(heatmap, min_opacity=0.3, max_opacity=0.8, radius=40, blur=25,
            gradient={'0.2': 'green', '0.5': 'yellow', '0.8': 'red'}).add_to(m)
    
    # Add cluster circles by comarca
    for lat, lon, count, radius, pop, comarca in clusters:
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius * count/pop * 10,  # conversion since CircleMarker uses pixels, not meters
            color='black',
            fill=True,
            fill_color='black',
            fill_opacity=0.5,
            tooltip=f"Comarca: {comarca}<br>Patients: {count}<br>Population: {pop:,}<br>Ratio: {count/pop:.6f}",
        ).add_to(m)
    
    # Add mini-map
    m.add_child(MiniMap(toggle_display=True))
    
    # Save map
    map_file = os.path.join(MAPS_DIR, f"map_{disease}_{contaminant}.html")
    m.save(map_file)
    
    return map_file


def generate_map_and_html_3(contaminant, disease):
    metadata = load_metadata()
    
    # Load city -> comarca correspondences (from function 2)
    def parse_cities_catalunya(filepath):
        """
        Parse the cities_catalunya file handling commas in city names.
        Expected format: code,city_name,comarca_code,comarca_name
        Problem: city names can contain commas (e.g. "Saus, Camallera i Llampaies")
        """
        cities_data = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Split by commas
                parts = line.split(',')
                
                if len(parts) < 4:
                    continue
                
                # First element is the city code
                city_code = parts[0].strip()
                
                # Last element is the comarca name
                comarca_name = parts[-1].strip()
                
                # Second to last is the comarca code
                comarca_code = parts[-2].strip()
                
                # Everything in the middle constitutes the city name
                city_name_parts = parts[1:-2]
                city_name = ','.join(city_name_parts).strip()
                
                cities_data.append({
                    'city_code': city_code,
                    'city_name': city_name,
                    'comarca_code': comarca_code,
                    'comarca_name': comarca_name
                })
                
        return pd.DataFrame(cities_data)
    
    cities_df = parse_cities_catalunya("data-csv/cities_catalunya_modified.csv")

    # Load city coordinates (from function 2)
    geo_cache_df = pd.read_csv("data-csv/cache_geo.csv")
    
    # Load postal codes (from function 2)
    postal_df = pd.read_csv("data-csv/Codis_postals.csv")
    
    # Function to normalize city names (from function 2)
    def normalize_city_name(city_name):
        """
        Normalize city names handling determinants.
        Example: "Garriga, la" -> "la Garriga"
        """
        city_name = str(city_name).strip()
        
        # Handle "Name, determiner" format
        if ', ' in city_name:
            parts = city_name.split(', ')
            if len(parts) == 2:
                main_name = parts[0].strip()
                determiner = parts[1].strip().lower()
                # Put determiner in front
                city_name = f"{determiner} {main_name}"
        
        # Normalize and lowercase
        city_normalized = unidecode.unidecode(city_name.lower())
        return city_normalized
    
    # Create postal code -> city dictionary (from function 2)
    postal_to_city = {}
    for _, row in postal_df.iterrows():
        postal_code = str(row['Codi postal']).strip()
        city_name = normalize_city_name(row['Nom municipi'])
        postal_to_city[postal_code] = city_name
    
    # Create city -> comarca dictionary (from function 2)
    city_to_comarca = {}
    for _, row in cities_df.iterrows():
        city = normalize_city_name(row['city_name'])
        comarca = str(row['comarca_name']).strip()
        city_to_comarca[city] = comarca
    
    # Create city -> coordinates dictionary (from function 2)
    city_coords = {}
    if 'city_name' in geo_cache_df.columns and 'lat' in geo_cache_df.columns and 'lon' in geo_cache_df.columns:
        city_col_geo = 'city_name'
        lat_col = 'lat'
        lon_col = 'lon'
    else:
        # Use available columns
        city_col_geo = geo_cache_df.columns[0]
        lat_col = 'lat' if 'lat' in geo_cache_df.columns else geo_cache_df.columns[1]
        lon_col = 'lon' if 'lon' in geo_cache_df.columns else geo_cache_df.columns[2]
    
    for _, row in geo_cache_df.iterrows():
        city = normalize_city_name(row[city_col_geo])
        lat = row[lat_col]
        lon = row[lon_col]
        if not pd.isna(lat) and not pd.isna(lon):
            city_coords[city] = {'lat': lat, 'lon': lon}
    
    # Load population data by city (from function 2)
    pop_df = pd.read_csv("data-csv/2861-3.csv", encoding="latin1", sep=";")
    pop_df["City"] = pop_df["Municipios"].apply(lambda x: normalize_city_name(x.split(" ", 1)[1].split(",")[0].strip()))
    pop_df["Population"] = pop_df["Total"].astype(str).str.replace(".", "", regex=False).astype(int)
    population_dict = dict(zip(pop_df["City"], pop_df["Population"]))
    
    # Calculate population by comarca (from function 2)
    comarca_population = {}
    for city, pop in population_dict.items():
        if city in city_to_comarca:
            comarca = city_to_comarca[city]
            comarca_population[comarca] = comarca_population.get(comarca, 0) + pop
    
    # Create heatmap
    heatmap = compute_heatmap(contaminant, city_coords)

    # Get patients data
    patients = metadata[metadata['cod'] == disease]
    
    # PART 1: Clustering by patient coordinates (from function 1)
    city_col = next((c for c in metadata.columns if "city" in c.lower()), None)
    coords_individual = []
    
    if city_col:  # Use city column if available
        for _, row in patients.iterrows():
            city = str(row[city_col]).strip().lower()
            city_norm = unidecode.unidecode(city)
            if city_norm in city_coords:
                lat = city_coords[city_norm]['lat']
                lon = city_coords[city_norm]['lon']
                if not pd.isna(lat) and not pd.isna(lon):
                    coords_individual.append([lat, lon])
    
    clusters_individual = []
    if len(coords_individual) > 0:
        coords_array = np.array(coords_individual)
        clustering = DBSCAN(eps=0.1, min_samples=5).fit(coords_array)
        labels = clustering.labels_
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = coords_array[labels == label]
            center_lat = np.mean(cluster_points[:, 0])
            center_lon = np.mean(cluster_points[:, 1])
            count = len(cluster_points)
            dists = np.sqrt((cluster_points[:, 0] - center_lat)**2 + (cluster_points[:, 1] - center_lon)**2)
            max_dist = dists.max()
            base_radius_m = max_dist * 111_000
            radius_m = min(base_radius_m + 200 * count, 25_000)
            clusters_individual.append((center_lat, center_lon, count, radius_m))

    # PART 2: Clustering by comarca (from function 2)
    comarca_counts = {}
    comarca_coords = {}
    comarca_cities = {}
    
    # Check if we have postal_code column for comarca-based analysis
    if 'postal_code' in patients.columns:
        for idx, row in patients.iterrows():
            postal_code = str(row['postal_code']).strip()
            
            # Find city from postal code
            if postal_code not in postal_to_city:
                continue
            
            city_norm = postal_to_city[postal_code]
            
            # Check that city exists in our data
            if city_norm not in city_to_comarca or city_norm not in city_coords:
                continue
            
            coords = city_coords[city_norm]
            lat, lon = coords['lat'], coords['lon']
            
            # Get comarca for this city
            comarca = city_to_comarca[city_norm]
            
            # Count patients by comarca
            comarca_counts[comarca] = comarca_counts.get(comarca, 0) + 1
            
            # Initialize structures for this comarca if necessary
            if comarca not in comarca_cities:
                comarca_cities[comarca] = set()
            
            # Add this city to the comarca
            comarca_cities[comarca].add(city_norm)
        
        # Calculate central coordinates for each comarca
        for comarca, cities in comarca_cities.items():
            if cities:
                # Calculate geographic center of comarca
                lats = []
                lons = []
                for city in cities:
                    if city in city_coords:
                        coords = city_coords[city]
                        lat, lon = coords['lat'], coords['lon']
                        if not pd.isna(lat) and not pd.isna(lon):
                            lats.append(lat)
                            lons.append(lon)
                
                if lats and lons:
                    center_lat = sum(lats)/len(lats)
                    center_lon = sum(lons)/len(lons)
                    comarca_coords[comarca] = [center_lat, center_lon]

    # Build clusters by comarca
    clusters_comarca = []
    for comarca, count in comarca_counts.items():
        if comarca not in comarca_coords:
            continue
            
        if comarca not in comarca_population or comarca_population[comarca] == 0:
            continue
            
        lat, lon = comarca_coords[comarca]
        pop = comarca_population[comarca]
        
        # Filter comarques with too few cases
        if count < 3:
            continue
        
        # Calculate ratio and radius
        ratio = count / pop
        scale = 3_000_000
        radius = min(ratio * scale, 30_000)
        
        clusters_comarca.append((lat, lon, count, radius, pop, comarca))

    # PART 3: Red zone analysis (from function 1)
    heatmap_points = heatmap
    contaminant_score = 0
    for center_lat, center_lon, count, radius_m in clusters_individual:
        if count >= 8:
            if is_red_zone(center_lat, center_lon, heatmap_points):
                contaminant_score += 1

    # Create map
    if len(coords_individual) > 0:
        lat0, lon0 = coords_individual[0]
    elif len(clusters_comarca) > 0:
        lat0, lon0 = clusters_comarca[0][0], clusters_comarca[0][1]
    else:
        lat0, lon0 = 41.39, 2.15  # Default coordinates for Catalonia
    
    m = folium.Map(location=[lat0, lon0], zoom_start=8, tiles="CartoDB positron")

    # Add comarca boundaries (from function 2)
    try:
        with open("comarques-compressed.geojson", "r", encoding="utf-8") as f:
            comarques_geojson = json.load(f)

        def style_comarques(feature):
            return {
                'fillOpacity': 0,
                'color': 'grey',
                'weight': 1
            }

        folium.GeoJson(
            comarques_geojson,
            name="Comarques",
            style_function=style_comarques
        ).add_to(m)
    except FileNotFoundError:
        print("Warning: comarques-compressed.geojson not found, skipping comarca boundaries")

    # Add heatmap
    print(f"Heatmap size: {len(heatmap)} | Example: {heatmap[:3]}")
    HeatMap(heatmap, min_opacity=0.3, max_opacity=0.8, radius=40, blur=25,
            gradient={'0.2': 'green', '0.5': 'yellow', '0.8': 'red'}).add_to(m)
    
    # Add individual coordinate clusters (blue circles)
    for center_lat, center_lon, count, radius_m in clusters_individual:
        if count >= 8:
            folium.Circle(
                location=[center_lat, center_lon],
                radius=radius_m,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.25,
                popup=f"[Individual] Cluster: {count} patients with {disease}"
            ).add_to(m)

    # Add comarca clusters (red circles)
    for lat, lon, count, radius, pop, comarca in clusters_comarca:
        folium.Circle(
            location=[lat, lon],
            radius=radius,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.25,
            popup=f"[Comarca] {comarca}<br>Patients: {count}<br>Population: {pop:,}<br>Ratio: {count/pop:.6f}"
        ).add_to(m)
    
    # Add mini-map
    m.add_child(MiniMap(toggle_display=True))
    
    # Save map
    map_file = os.path.join(MAPS_DIR, f"map3_{disease}_{contaminant}.html")
    m.save(map_file)
    
    print(f"Map saved: {map_file}")
    print(f"Individual clusters: {len(clusters_individual)}")
    print(f"Comarca clusters: {len(clusters_comarca)}")
    print(f"Contaminant score: {contaminant_score}")
    
    return map_file