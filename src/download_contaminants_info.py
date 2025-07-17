import requests
import os
import shutil
import gzip

def download_contaminants_data(start_date, end_date, dest_path="data-csv/contaminants_data.csv"):
    """
    Télécharge les données de qualité de l'air pour tous les contaminants entre deux dates.
    Les dates doivent être au format 'YYYY-MM-DD'.
    """
    from urllib.parse import quote
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Construction du filtre SQL uniquement sur la date
        where_clause = f"data between '{start_date}T00:00:00' and '{end_date}T00:00:00'"
        encoded_where = quote(where_clause)

        url = f"https://analisi.transparenciacatalunya.cat/resource/tasf-thgu.csv?$limit=1000000&$offset=0&$where={encoded_where}"

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Test si c'est compressé (gzip)
            content_encoding = response.headers.get('Content-Encoding', '')
            if 'gzip' in content_encoding:
                # décompresser à la volée
                with gzip.GzipFile(fileobj=response.raw) as gz:
                    with open(dest_path, 'wb') as out_file:
                        shutil.copyfileobj(gz, out_file)
            else:
                with open(dest_path, "wb") as f:
                    shutil.copyfileobj(response.raw, f)
        else:
            raise RuntimeError(f"Erreur {response.status_code} : {response.text}")

    except Exception as e:
        print(f"❌ Erreur : {e}")
