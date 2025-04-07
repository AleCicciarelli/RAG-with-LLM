import json

# Funzione per caricare il file delle risposte
def load_responses(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Funzione per trasformare il campo 'answer' in un oggetto JSON
def transform_answers(responses):
    for response in responses:
        # Deserializzare la stringa 'answer' in un oggetto JSON
        try:
            response['answer'] = json.loads(response['answer'])
        except json.JSONDecodeError as e:
            print(f"Errore nel parsing del campo 'answer' per la domanda: {response['question']}")
            print(f"Contenuto errato: {response['answer']}")
    return responses

# Funzione per salvare il risultato trasformato in un nuovo file JSON
def save_transformed_responses(responses, output_filename="transformed_responses.json"):
    with open(output_filename, 'w') as file:
        json.dump(responses, file, indent=4)

# Caricare le risposte originali
responses = load_responses('results.json')

# Trasformare i campi 'answer' in oggetti JSON
transformed_responses = transform_answers(responses)

# Salvare il risultato trasformato
save_transformed_responses(transformed_responses)

# Stampa per vedere il risultato
for response in transformed_responses:
    print(json.dumps(response, indent=4))
