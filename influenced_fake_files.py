import csv
import random
import os
import string

taulas = ['diagnosis','socioeconomiques','visites','paciente','UP']
camps = ["idps","cod","dat","dbaixa","tup","age_dia","age_dia_agr","pagr_dia","apo_far","medea_centre","vis_cod","motius","idup","idprofs","tipus_agr","sexe","situacio","age_sit","CODIABS","ambit","city","postal_code"]
contaminants = ['C6H6','CI2','CO','H2S','HCl','HCNM','HCT','Hg','NO','NO2','NOX','O3','PM1','PM10','PM2.5','PS','SO2']
tups = ['1011','1012','2021','2022','2024','2025','3038','5051','6061','6063','7071','7075','7076','9010','9505']
apo_fars = ['001','002','003','004','005','006']
tipus_agrs = ['Pre','Ect','Dom','Oth']
sexes = ['H', 'D']
situacios = ['A','D','T']
ambits = ['A','H','U','S','M','C','O']

def random_varchar(n):
    # Génère une chaîne alphanumérique de longueur n
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def random_integer(mini=0, maxi=1000):
    return random.randint(mini, maxi)

def random_date(start_year=2020, end_year=2023):
    # Génère une date aléatoire entre start_year et end_year
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # Pour simplifier, on limite à 28 jours
    return f"{year}-{month:02d}-{day:02d}"

cities_list = []
with open('data-csv/Codis_postals.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["Nom municipi"].zfill(6)  # S'assurer que l'INE a 6 chiffres
        postal_code = row["Codi postal"]
        cities_list.append((name, postal_code))

vis_cods = [random.randint(1_000_000, 9_999_999) for _ in range(50)]

# Liste pondérée de codes (ex : certains plus fréquents)
cods_frequents = [
    "A01.0", "B02.1", "C03.2", "D04.3", "E05.4"
]
cods_pondes = cods_frequents * 30  # chaque code fréquent apparaît 30 fois
cods_pondes += [
    random.choice(string.ascii_uppercase)+f"{random.randint(0, 99):02d}"+f".{random.randint(0, 9)}"
    for _ in range(100)
]

# Exemple : chaque ville a 1 ou 2 codes fréquents principaux
city_cod_map = {}
for city, postal_code in cities_list:
    city_cod_map[city] = random.sample(cods_frequents, k=random.randint(1, 2))

filename = f'metadata.csv'
with open(filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(camps)  # écrit l'en-tête
    for i in range(1000):
        idps = i
        city, postal_code = random.choice(cities_list)
        # 70% de chances de prendre un code fréquent de la ville, sinon un code aléatoire
        if random.random() < 0.7:
            cod = random.choice(city_cod_map[city])
        else:
            cod = random.choice(cods_pondes)
        dat = random_date()
        dbaixa = dat
        tup = random.choice(tups)
        age_dia = random_integer(0, 100)
        age_dia_agr = age_dia
        pagr_dia = random_integer(1, 50)
        apo_far = random.choice(apo_fars)
        medea_centre = f"U{random.randint(1, 5)}"
        vis_cod = random.choice(vis_cods)
        motius = random_varchar(10)
        idup = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        idprofs = f"UP{random.randint(0,999999):06d}"  # toujours 6 chiffres
        tipus_agr = random.choice(tipus_agrs)
        sexe = random.choice(sexes)
        situacio = random.choice(situacios)
        age_sit = age_dia
        CODIABS = str(random.randint(1,9))
        ambit = random.choice(ambits)

        row = [
            idps, cod, dat, dbaixa, tup, age_dia, age_dia_agr, pagr_dia,
            apo_far, medea_centre, vis_cod, motius, idup, idprofs, tipus_agr,
            sexe, situacio, age_sit, CODIABS, ambit, city, postal_code
        ]
        writer.writerow(row)
