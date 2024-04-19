import pandas as pd

# Caricare il file mot_labels.csv con gestione migliore delle eccezioni
try:
    mot_labels = pd.read_csv('C:\\Users\\matte\\OneDrive\\Desktop\\Multimedia\\progetto\\mot_labels.csv')
except FileNotFoundError:
    print("File non trovato. Verifica il percorso del file.")
    exit(1)
except pd.errors.EmptyDataError:
    print("File vuoto. Fornisci un file CSV valido.")
    exit(1)

# Rimuovere la colonna indice inutile
mot_labels.drop(columns=['Unnamed: 0'], inplace=True)

# Convertire le colonne degli attributi in booleani
for col in ['attributes.crowd', 'attributes.occluded', 'attributes.truncated']:
    mot_labels[col] = mot_labels[col].astype(bool)

# Rimuovere righe con valori mancanti
mot_labels.dropna(inplace=True)

# Verificare il tipo di dati aggiornato e le righe rimanenti
print(mot_labels.info())
print(mot_labels.head())


# Pulire i dati rimuovendo le righe con valori mancanti e filtrando per 'car'
cleaned_mot_labels = mot_labels.dropna()
df_real_labels = cleaned_mot_labels[
    (cleaned_mot_labels['videoName'] == '00c4c672-26d36ad8') &
    (cleaned_mot_labels['category'] == 'car')
]

# Selezione delle colonne pertinenti
df_real_labels = df_real_labels[['frameIndex', 'box2d.x1', 'box2d.x2', 'box2d.y1', 'box2d.y2']]

# Salvataggio in un nuovo CSV
output_path = 'labels_reali_1.csv'
df_real_labels.to_csv(output_path, index=False)
print("Le etichette reali sono state salvate con successo in:", output_path)

print(df_real_labels)
