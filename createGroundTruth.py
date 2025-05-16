import json

# Leggi il file di input
with open('../databases_load/ground_truth.csv', 'r') as file:
    lines = file.readlines()

# Crea una lista di dizionari con la struttura richiesta
data = []
for line in lines:
    # Dividi ogni riga in base alla virgola
    parts = line.strip().split(',')
    
    # Aggiungi il dizionario alla lista
    if len(parts) == 3:  # Assicurati che ogni riga abbia 3 colonne
        data.append({
            'answer': parts[0],
            'sr_why': parts[1]
        })

# Scrivi il risultato nel file JSON
with open('output.json', 'w') as json_file:
    json.dump(data, json_file, indent=2)
