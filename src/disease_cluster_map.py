import os
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MiniMap
from sklearn.cluster import DBSCAN
import unidecode
import webbrowser

# Constantes des chemins
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
        villes = pd.read_csv(CACHE_FILE)
    else:
        villes = pd.DataFrame(columns=["ville", "lat", "lon"])
    villes['ville_norm'] = villes['ville'].astype(str).str.lower().apply(unidecode.unidecode)
    return villes.set_index('ville_norm')[['lat', 'lon']].to_dict(orient='index')


def compute_heatmap(contaminant, city_coords):
    heat_data = []
    for fn in os.listdir(CONTAM_PATH):
        if fn.startswith(pattern) and fn.endswith('.csv'):
            df = pd.read_csv(os.path.join(CONTAM_PATH, fn))
            if 'municipi' in df.columns and 'contaminant' in df.columns:
                df_c = df[df['contaminant'] == contaminant]
                hour_cols = [f"h{i}" for i in range(1, 25) if f"h{i}" in df_c.columns]
                for _, row_c in df_c.iterrows():
                    ville_c = str(row_c['municipi']).strip().lower()
                    ville_c_norm = unidecode.unidecode(ville_c)
                    if ville_c_norm in city_coords and hour_cols:
                        lat_c = city_coords[ville_c_norm]['lat']
                        lon_c = city_coords[ville_c_norm]['lon']
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

def generate_map_and_html_1(contaminant, disease):
    metadata = load_metadata()
    city_coords = load_city_coords()
    city_col = next((c for c in metadata.columns if "city" in c.lower()), None)
    heatmap = compute_heatmap(contaminant, city_coords)

    index_path = os.path.join(GENERATED_DIR, "index_disease_cluster.html")


    patients = metadata[metadata['cod'] == disease]
    coords = []
    for _, row in patients.iterrows():
        ville = str(row[city_col]).strip().lower()
        ville_norm = unidecode.unidecode(ville)
        if ville_norm in city_coords:
            lat = city_coords[ville_norm]['lat']
            lon = city_coords[ville_norm]['lon']
            if not pd.isna(lat) and not pd.isna(lon):
                coords.append([lat, lon])
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

    contaminant_scores = []

    heatmap_points = heatmap
    score = 0
    for center_lat, center_lon, count, radius_m in clusters:
        if count >= 8:
            if is_red_zone(center_lat, center_lon, heatmap_points):
                score += 1
    contaminant_scores.append((contaminant, score))
    contaminant_scores.sort(key=lambda x: x[1], reverse=True)

    lat0, lon0 = coords[0] if len(coords) > 0 else (41.39, 2.15)
    m = folium.Map(location=[lat0, lon0], zoom_start=8, tiles="CartoDB positron")

    print(f"Heatmap size: {len(heatmap)} | Exemple: {heatmap[:3]}")
    HeatMap(heatmap, min_opacity=0.3, max_opacity=0.8, radius=40, blur=25,
            gradient={'0.2': 'green', '0.5': 'yellow', '0.8': 'red'}).add_to(m)
    # for lat_pt, lon_pt in coords:
    #     folium.CircleMarker(
    #         location=[lat_pt, lon_pt],
    #         radius=3,
    #         color='blue',
    #         fill=True,
    #         fill_color='blue',
    #         fill_opacity=0.7,
    #         popup="Patient"
    #     ).add_to(m)
    for center_lat, center_lon, count, radius_m in clusters:
        if count >= 8:
            folium.Circle(
                location=[center_lat, center_lon],
                radius=radius_m,
                color='black',
                fill=True,
                fill_color='black',
                fill_opacity=0.2,
                popup=f"Cluster: {count} patients with {disease}"
            ).add_to(m)
    m.add_child(MiniMap(toggle_display=True))
    map_file = os.path.join(MAPS_DIR, f"map_{disease}_{contaminant}.html")
    m.save(map_file)
    return map_file

def generate_map_and_html_2(contaminant, disease):
    metadata = load_metadata()
    city_coords = load_city_coords()
    city_col = next((c for c in metadata.columns if "city" in c.lower()), None)
    heatmap = compute_heatmap(contaminant, city_coords)

    # Charger les populations
    pop_df = pd.read_csv("data-csv/2861-3.csv", encoding="latin1", sep=";")
    pop_df["Ville"] = pop_df["Municipios"].apply(lambda x: unidecode.unidecode(x.split(" ", 1)[1].split(",")[0].strip().lower()))
    pop_df["Population"] = pop_df["Total"].astype(str).str.replace(".", "", regex=False).astype(int)
    population_dict = dict(zip(pop_df["Ville"], pop_df["Population"]))

    # Collecter les malades par ville
    patients = metadata[metadata['cod'] == disease]
    ville_counts = {}
    coords_by_city = {}

    for _, row in patients.iterrows():
        ville = str(row[city_col]).strip().lower()
        ville_norm = unidecode.unidecode(ville)
        if ville_norm in city_coords:
            lat = city_coords[ville_norm]['lat']
            lon = city_coords[ville_norm]['lon']
            if not pd.isna(lat) and not pd.isna(lon):
                ville_counts[ville_norm] = ville_counts.get(ville_norm, 0) + 1
                coords_by_city[ville_norm] = [lat, lon]

    # Construire le tableau des villes malades
    cities_data = []
    for ville, count in ville_counts.items():
        if ville in coords_by_city and ville in population_dict:
            lat, lon = coords_by_city[ville]
            pop = population_dict[ville]
            cities_data.append({
                "ville": ville,
                "lat": lat,
                "lon": lon,
                "count": count,
                "pop": pop
            })

    # Clustering des villes
    coords = np.array([[c["lat"], c["lon"]] for c in cities_data])
    if len(coords) > 0:
        clustering = DBSCAN(eps=0.12, min_samples=1).fit(coords)
        labels = clustering.labels_
    else:
        labels = []

    # Construire les clusters de villes
    clusters = []
    for label in set(labels):
        cluster_points = [cities_data[i] for i in range(len(cities_data)) if labels[i] == label]
        total_malades = sum(p["count"] for p in cluster_points)
        total_habitants = sum(p["pop"] for p in cluster_points)
        if total_malades < 5 or total_habitants == 0:
            continue  # ignorer les petits clusters

        center_lat = np.mean([p["lat"] for p in cluster_points])
        center_lon = np.mean([p["lon"] for p in cluster_points])
        ratio = total_malades / total_habitants
        scale = 3_000_000  # facteur de mise à l’échelle
        radius = min(ratio * scale, 30_000)
        clusters.append((center_lat, center_lon, total_malades, radius, total_habitants, [p["ville"] for p in cluster_points]))

    # Créer la carte
    lat0, lon0 = (coords[0] if len(coords) > 0 else (41.39, 2.15))
    m = folium.Map(location=[lat0, lon0], zoom_start=8, tiles="CartoDB positron")

    HeatMap(heatmap, min_opacity=0.3, max_opacity=0.8, radius=40, blur=25,
            gradient={'0.2': 'green', '0.5': 'yellow', '0.8': 'red'}).add_to(m)

    for lat, lon, count, radius, pop, villes in clusters:
        folium.Circle(
            location=[lat, lon],
            radius=radius,
            color='black',
            fill=True,
            fill_color='black',
            fill_opacity=0.2,
            popup=f"Cluster: {count} malades / {pop} hab.\n({round(100*count/pop, 2)}%)\n{', '.join(villes)}"
        ).add_to(m)

    m.add_child(MiniMap(toggle_display=True))
    map_file = os.path.join(MAPS_DIR, f"map_{disease}_{contaminant}.html")
    m.save(map_file)
    return map_file

def generate_map_and_html_3(contaminant, disease):
    metadata = load_metadata()
    city_coords = load_city_coords()
    city_col = next((c for c in metadata.columns if "city" in c.lower()), None)
    heatmap = compute_heatmap(contaminant, city_coords)

    # Partie 1 : clustering par coordonnées des patients
    patients = metadata[metadata['cod'] == disease]
    coords = []
    for _, row in patients.iterrows():
        ville = str(row[city_col]).strip().lower()
        ville_norm = unidecode.unidecode(ville)
        if ville_norm in city_coords:
            lat = city_coords[ville_norm]['lat']
            lon = city_coords[ville_norm]['lon']
            if not pd.isna(lat) and not pd.isna(lon):
                coords.append([lat, lon])
    coords = np.array(coords)

    clusters_coords = []
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
            radius_m = min(count * 1000, 30000)  # Taille dépend du nombre de malades
            clusters_coords.append((center_lat, center_lon, count, radius_m))

    # Partie 2 : clustering par ville avec prise en compte des populations
    pop_df = pd.read_csv("data-csv/2861-3.csv", encoding="latin1", sep=";")
    pop_df["Ville"] = pop_df["Municipios"].apply(lambda x: unidecode.unidecode(x.split(" ", 1)[1].split(",")[0].strip().lower()))
    pop_df["Population"] = pop_df["Total"].astype(str).str.replace(".", "", regex=False).astype(int)
    population_dict = dict(zip(pop_df["Ville"], pop_df["Population"]))

    ville_counts = {}
    coords_by_city = {}
    for _, row in patients.iterrows():
        ville = str(row[city_col]).strip().lower()
        ville_norm = unidecode.unidecode(ville)
        if ville_norm in city_coords:
            lat = city_coords[ville_norm]['lat']
            lon = city_coords[ville_norm]['lon']
            if not pd.isna(lat) and not pd.isna(lon):
                ville_counts[ville_norm] = ville_counts.get(ville_norm, 0) + 1
                coords_by_city[ville_norm] = [lat, lon]

    cities_data = []
    for ville, count in ville_counts.items():
        if ville in coords_by_city and ville in population_dict:
            lat, lon = coords_by_city[ville]
            pop = population_dict[ville]
            cities_data.append({
                "ville": ville,
                "lat": lat,
                "lon": lon,
                "count": count,
                "pop": pop
            })

    coords_villes = np.array([[c["lat"], c["lon"]] for c in cities_data])
    if len(coords_villes) > 0:
        clustering = DBSCAN(eps=0.12, min_samples=1).fit(coords_villes)
        labels = clustering.labels_
    else:
        labels = []

    clusters_villes = []
    for label in set(labels):
        cluster_points = [cities_data[i] for i in range(len(cities_data)) if labels[i] == label]
        total_malades = sum(p["count"] for p in cluster_points)
        total_habitants = sum(p["pop"] for p in cluster_points)
        if total_malades < 5 or total_habitants == 0:
            continue

        center_lat = np.mean([p["lat"] for p in cluster_points])
        center_lon = np.mean([p["lon"] for p in cluster_points])
        ratio = total_malades / total_habitants
        scale = 3_000_000
        radius = min(ratio * scale, 30000)
        clusters_villes.append((center_lat, center_lon, total_malades, radius, total_habitants, [p["ville"] for p in cluster_points]))

    # Création de la carte
    lat0, lon0 = coords[0] if len(coords) > 0 else (41.39, 2.15)
    m = folium.Map(location=[lat0, lon0], zoom_start=8, tiles="CartoDB positron")

    HeatMap(heatmap, min_opacity=0.3, max_opacity=0.8, radius=40, blur=25,
            gradient={'0.2': 'green', '0.5': 'yellow', '0.8': 'red'}).add_to(m)

    for center_lat, center_lon, count, radius_m in clusters_coords:
        folium.Circle(
            location=[center_lat, center_lon],
            radius=radius_m,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.25,
            popup=f"[Coord] Cluster: {count} patients"
        ).add_to(m)

    for lat, lon, count, radius, pop, villes in clusters_villes:
        folium.Circle(
            location=[lat, lon],
            radius=radius,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.25,
            popup=f"[Ville] Cluster: {count} malades / {pop} hab.\n({round(100*count/pop, 2)}%)\n{', '.join(villes)}"
        ).add_to(m)

    m.add_child(MiniMap(toggle_display=True))

    map_file = os.path.join(MAPS_DIR, f"map3_{disease}_{contaminant}.html")
    m.save(map_file)
    return map_file