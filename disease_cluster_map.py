import pandas as pd
import os
import folium
from folium.plugins import HeatMap, MiniMap
from sklearn.cluster import DBSCAN
import numpy as np
import webbrowser
import unidecode

# FICHIERS ET DOSSIERS
GENERATED_DIR = "generated_data"
CONTAMINANTS_DIR = os.path.join(GENERATED_DIR, "contaminants")
MAPS_DIR = os.path.join(GENERATED_DIR, "maps_cluster")
METADATA_FILE = "metadata.csv"
CONTAM_PATH = "contaminants data"
pattern = "tsaf-thgu-"
CACHE_FILE = "cache_geo.csv"
os.makedirs(CONTAMINANTS_DIR, exist_ok=True)
os.makedirs(MAPS_DIR, exist_ok=True)

# CHARGEMENT DES DONNÉES
metadata = pd.read_csv(METADATA_FILE)
metadata['city_norm'] = metadata['city'].astype(str).str.lower().apply(unidecode.unidecode)

if os.path.exists(CACHE_FILE):
    villes = pd.read_csv(CACHE_FILE)
else:
    villes = pd.DataFrame(columns=["ville", "lat", "lon"])
villes['ville_norm'] = villes['ville'].astype(str).str.lower().apply(unidecode.unidecode)
city_coords = villes.set_index('ville_norm')[['lat', 'lon']].to_dict(orient='index')

# PRÉPARATION MALADIES
disease_col = 'cod'
city_col = next((c for c in metadata.columns if "city" in c.lower()), None)
disease_counts = metadata[disease_col].value_counts()
diseases = disease_counts.index.tolist()

# LISTER TOUS LES CONTAMINANTS
all_contaminants = set()
for fn in os.listdir(CONTAM_PATH):
    if fn.startswith(pattern) and fn.endswith('.csv'):
        df = pd.read_csv(os.path.join(CONTAM_PATH, fn))
        if 'contaminant' in df.columns:
            all_contaminants.update(df['contaminant'].dropna().unique())
all_contaminants = sorted(all_contaminants)

# PRÉCOMPUTER HEATMAPS
heatmaps = {}
for contaminant in all_contaminants:
    heat_data = []
    for fn in os.listdir(CONTAM_PATH):
        if fn.startswith(pattern) and fn.endswith('.csv'):
            df = pd.read_csv(os.path.join(CONTAM_PATH, fn))
            if 'nom_estacio' in df.columns and 'contaminant' in df.columns:
                df_c = df[df['contaminant'] == contaminant]
                hour_cols = [f"h{i}" for i in range(1, 25) if f"h{i}" in df_c.columns]
                for _, row_c in df_c.iterrows():
                    ville_c = str(row_c['nom_estacio']).strip().lower()
                    ville_c_norm = unidecode.unidecode(ville_c)
                    if ville_c_norm in city_coords and hour_cols:
                        lat_c = city_coords[ville_c_norm]['lat']
                        lon_c = city_coords[ville_c_norm]['lon']
                        concs = [row_c[h] for h in hour_cols if not pd.isna(row_c[h])]
                        if concs:
                            avg_conc = sum(concs) / len(concs)
                            if all(isinstance(v, (int, float)) and not pd.isna(v) for v in [lat_c, lon_c, avg_conc]):
                                heat_data.append([float(lat_c), float(lon_c), float(avg_conc)])
    heatmaps[contaminant] = heat_data

# INDEX HTML
index_path = os.path.join(GENERATED_DIR, "index_diseases_cluster.html")
with open(index_path, "w", encoding="utf-8") as f:
    f.write("<html><head><title>Disease list</title></head><body>\n")
    f.write("<h2>List of diseases (codes), sorted by number of patients</h2>\n<ul>\n")
    for disease in diseases:
        f.write(f'<li><a href="contaminants/contaminants_{disease}.html">{disease} ({disease_counts[disease]} patients)</a></li>\n')
    f.write("</ul>\n</body></html>")
print(f"✅ Disease index created: {index_path}")

# CARTES POUR CHAQUE MALADIE
for disease in diseases:
    patients = metadata[metadata[disease_col] == disease]
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

    # DBSCAN clustering
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
            # Rayon de base : distance max entre centre et patient
            dists = np.sqrt((cluster_points[:, 0] - center_lat)**2 + (cluster_points[:, 1] - center_lon)**2)
            max_dist = dists.max()
            base_radius_m = max_dist * 111_000
            # Facteur lié au nombre de patients (exemple : +200m par patient)
            radius_m = min(base_radius_m + 200 * count, 25_000)  # Limite à 25 km
            clusters.append((center_lat, center_lon, count, radius_m))

    # Définir le seuil "zone rouge" (80% du max de la heatmap)
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

    # Pour chaque contaminant, compte le nombre de clusters "rouges"
    contaminant_scores = []
    for contaminant in all_contaminants:
        heatmap_points = heatmaps.get(contaminant, [])
        score = 0
        # Lors du calcul des scores
        for center_lat, center_lon, count, radius_m in clusters:
            if count >= 8:
                if is_red_zone(center_lat, center_lon, heatmap_points):
                    score += 1
        contaminant_scores.append((contaminant, score))
    contaminant_scores.sort(key=lambda x: x[1], reverse=True)

    # Page contaminants
    contaminant_file = os.path.join(CONTAMINANTS_DIR, f"contaminants_{disease}.html")
    with open(contaminant_file, "w", encoding="utf-8") as f:
        f.write(f"<html><head><title>Contaminants for {disease}</title></head><body>\n")
        f.write(f"<h2>Contaminants for disease {disease}</h2>\n<ul>\n")
        for contaminant, score in contaminant_scores:
            map_file = f"../maps_cluster/map_{disease}_{contaminant}.html"
            f.write(f'<li><a href="{map_file}">{contaminant} (clusters in red zones: {score})</a></li>\n')
        f.write("</ul>\n</body></html>")
    print(f"✅ Contaminant list for {disease} created: {contaminant_file}")

    # Maps
    for contaminant in all_contaminants:
        lat0, lon0 = coords[0] if len(coords) > 0 else (41.39, 2.15)
        m = folium.Map(location=[lat0, lon0], zoom_start=8, tiles="CartoDB positron")
        if contaminant in heatmaps and heatmaps[contaminant]:
            HeatMap(heatmaps[contaminant], min_opacity=0.3, max_opacity=0.8, radius=40, blur=25,
                    gradient={'0.2': 'green', '0.5': 'yellow', '0.8': 'red'}).add_to(m)
        # Ajoute un petit point pour chaque patient
        for lat_pt, lon_pt in coords:
            folium.CircleMarker(
                location=[lat_pt, lon_pt],
                radius=3,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.7,
                popup="Patient"
            ).add_to(m)
        # Ajoute les clusters
        for center_lat, center_lon, count, radius_m in clusters:
            if count >= 8:  # Affiche tous les clusters de 8 personnes ou plus
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
        print(f"✅ Map generated: {map_file}")

webbrowser.open('file://' + os.path.realpath(index_path))
