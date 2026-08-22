# ==============================================================================
# PROGETTO DL-QUANTUM — Fase 1: Pipeline Dati
# Notebook di destinazione: notebooks/01_pipeline_dati.ipynb
# Blocco 1/N: Setup, riproducibilità, caricamento e verifica struttura dati
# ==============================================================================
#
# Questo blocco NON usa ancora architetture di rete neurale (verranno nei
# blocchi successivi, basate su Lab5/Lab7): qui prepariamo solo i tensori
# di input che tutti i modelli (RNN e Transformer) useranno allo stesso modo.
#
# NOTA SU STRUTTURA CARTELLE (Drive):
#   Dataset:     /content/drive/MyDrive/DL Quantum/data/trajectories.csv
#   Checkpoint:  /content/drive/MyDrive/DL Quantum/checkpoints/<nome_modello>/
# Il codice di questo file va diviso in celle Colab separate (vedi indicazioni
# fornite a parte); qui è riportato come unico file solo per revisione/versioning
# su GitHub (src/data_pipeline.py).

# ------------------------------------------------------------------------
# 0. MOUNT GOOGLE DRIVE (prima cella Colab, eseguire una sola volta a sessione)
# ------------------------------------------------------------------------
# from google.colab import drive
# drive.mount('/content/drive')

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os

# ------------------------------------------------------------------------
# 1. CONFIGURAZIONE PARAMETRICA
# ------------------------------------------------------------------------
# Tutti i parametri della pipeline sono centralizzati qui: questo è il punto
# che modificheremo nelle fasi successive (es. finestre temporali diverse
# per la Challenge 1 "minimum time window") senza toccare il resto del codice.

SEED = 42                      # Seed globale per riproducibilità
N_TRAJ = 400                   # Numero totale di traiettorie nel dataset
N_TIMESTEPS = 1001             # Punti temporali per traiettoria (T_fin=20, dt=0.02)
N_FEATURES = 55                # 10 magnetizzazioni + 45 correlazioni (colonna tempo esclusa)
DT = 0.02                      # Passo temporale di campionamento
T_FIN = 20.0                   # Tempo finale della simulazione

# Split per traiettoria (non per riga!): 280 train / 60 val / 60 test
N_TRAIN = 280
N_VAL = 60
N_TEST = 60

# Path coerente con la struttura Drive concordata: cartella "DL Quantum/data/"
DATA_PATH = "/content/drive/MyDrive/DL Quantum/data/trajectories.csv"

# Indici delle feature all'interno delle 55 colonne (0-indexed), secondo la
# documentazione ufficiale del dataset:
#   - colonne 0..9   -> magnetizzazioni m_1, ..., m_10
#   - colonne 10..54 -> correlazioni c_12, c_13, ..., c_1N, c_23, ..., c_{N-1,N}
#                       (ordine lessicografico delle coppie i<j)
# PERCHÉ servono già ora, anche se non usati in questo blocco: il progetto
# richiede di valutare magnetizzazioni e correlazioni SEPARATAMENTE nei
# risultati finali (non è un dettaglio dei lab, ma un vincolo esplicito della
# consegna). Definirli qui, in modo centralizzato, evita di doverli ricavare
# a mano più avanti nel notebook, con rischio di errori di indicizzazione.
N_QUBITS = 10
MAGNETIZATION_COLS = slice(0, N_QUBITS)              # colonne 0..9
CORRELATION_COLS = slice(N_QUBITS, N_FEATURES)       # colonne 10..54


# ------------------------------------------------------------------------
# 2. FISSAGGIO DEI SEED (riproducibilità obbligatoria)
# ------------------------------------------------------------------------
# PERCHÉ: il docente richiede esplicitamente che ogni notebook sia
# riproducibile. Fissiamo TUTTE le fonti di casualità che TensorFlow/Keras
# e le librerie sottostanti possono usare: Python random, numpy, tensorflow.
# Se una sola di queste non venisse fissata, due run identiche potrebbero
# comunque dare pesi iniziali o shuffle diversi, rendendo impossibile
# confrontare in modo affidabile le configurazioni di HP tuning richieste
# più avanti (vincolo 4 del progetto).

def set_seeds(seed: int = SEED):
    """Fissa tutti i seed random per garantire risultati riproducibili."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seeds(SEED)


# ------------------------------------------------------------------------
# 3. CARICAMENTO DEL DATASET GREZZO
# ------------------------------------------------------------------------
# Adattato dal notebook ufficiale del docente (read_data.ipynb).
#
# PERCHÉ header=None: il file trajectories.csv NON ha una riga di intestazione
# con i nomi delle colonne (a differenza di molti csv "standard"). Se non lo
# specificassimo esplicitamente, pandas assumerebbe di default che la prima
# riga contenga i nomi delle colonne (header=0) e la scarterebbe dai dati:
# perderemmo silenziosamente il primo timestep (tempo=0) della prima
# traiettoria, corrompendo la struttura fin dall'inizio senza alcun errore
# visibile. index_col=False evita che pandas provi a interpretare la prima
# colonna come un indice invece che come dato (colonna tempo).
df = pd.read_csv(DATA_PATH, header=None, index_col=False)

print(f"Shape del dataframe grezzo: {df.shape}")
print(f"Numero di colonne attese: 1 (tempo) + {N_FEATURES} (feature) = {1 + N_FEATURES}")
print(f"Colonne trovate: {df.shape[1]}")
assert df.shape[1] == 1 + N_FEATURES, (
    f"Attese {1 + N_FEATURES} colonne, trovate {df.shape[1]}. "
    "Controllare il file prima di proseguire."
)

# Senza header, pandas assegna alle colonne indici interi 0, 1, 2, ...
# La colonna 0 è il tempo (per specifica del dataset); le colonne 1..55
# sono le feature (magnetizzazioni + correlazioni).
time_col = 0
feature_cols = list(range(1, 1 + N_FEATURES))

print(f"Colonna tempo identificata: indice {time_col}")
print(f"Numero colonne feature: {len(feature_cols)}")


# ------------------------------------------------------------------------
# 4. VERIFICA DELLA STRUTTURA A TRAIETTORIE
# ------------------------------------------------------------------------
# PERCHÉ questo controllo è cruciale: se procedessimo al reshape senza
# verificare, un'assunzione sbagliata sul numero di timestep per traiettoria
# mescolerebbe silenziosamente dati di traiettorie diverse nella stessa
# sequenza. Il codice non darebbe errore, ma i dati sarebbero fisicamente
# insensati (es. una sequenza che "salta" da una traiettoria all'altra),
# e ce ne accorgeremmo solo molto più tardi, magari dopo ore di training.

# 4a. Il numero totale di righe deve essere esattamente N_TRAJ * N_TIMESTEPS
righe_attese = N_TRAJ * N_TIMESTEPS
print(f"\nRighe attese (N_TRAJ * N_TIMESTEPS): {righe_attese}")
print(f"Righe effettive nel file: {len(df)}")
assert len(df) == righe_attese, (
    f"Il numero di righe ({len(df)}) non corrisponde a "
    f"N_TRAJ * N_TIMESTEPS ({righe_attese}). "
    "Verificare N_TRAJ e N_TIMESTEPS prima di continuare."
)

# 4b. Il tempo deve resettarsi a ~0 esattamente ogni N_TIMESTEPS righe.
# Troviamo tutti gli indici dove il tempo è (quasi) zero, e verifichiamo
# che siano equispaziati di N_TIMESTEPS e che siano esattamente N_TRAJ.
tempo = df[time_col].to_numpy()
indici_reset = np.where(np.isclose(tempo, 0.0, atol=1e-8))[0]

print(f"\nNumero di reset del tempo trovati: {len(indici_reset)} (attesi: {N_TRAJ})")
assert len(indici_reset) == N_TRAJ, (
    f"Trovati {len(indici_reset)} reset del tempo, attesi {N_TRAJ}. "
    "La struttura del file non corrisponde a quella ipotizzata."
)

# Verifichiamo che la distanza tra un reset e il successivo sia sempre N_TIMESTEPS
distanze = np.diff(indici_reset)
assert np.all(distanze == N_TIMESTEPS), (
    "Le traiettorie non hanno tutte la stessa lunghezza: "
    f"distanze trovate = {np.unique(distanze)}, attesa unica = {N_TIMESTEPS}."
)

# Verifichiamo anche che l'ultimo istante di ogni traiettoria sia vicino a T_FIN - dt
ultimo_istante_atteso = T_FIN - DT
ultimi_istanti = tempo[indici_reset + N_TIMESTEPS - 1]
assert np.allclose(ultimi_istanti, ultimo_istante_atteso, atol=1e-6), (
    f"L'ultimo istante di alcune traiettorie non è {ultimo_istante_atteso:.4f} "
    f"come atteso da T_fin={T_FIN} e dt={DT}."
)

print("\n✅ Struttura a traiettorie verificata correttamente: "
      f"{N_TRAJ} traiettorie da {N_TIMESTEPS} timestep ciascuna.")


# ------------------------------------------------------------------------
# 5. RESHAPE A TENSORE (N_TRAJ, N_TIMESTEPS, N_FEATURES)
# ------------------------------------------------------------------------
# PERCHÉ questa forma: RNN e Transformer (Lab5, Lab7) si aspettano input
# nella forma standard (batch, timestep, feature) — esattamente lo schema
# usato nei lab per dati sequenziali. La colonna tempo viene tenuta da parte
# come riferimento (utile più avanti per la super-resolution, Challenge 2),
# ma NON entra tra le feature: è solo l'indice che ordina i timestep, non
# una grandezza fisica che il modello deve imparare a prevedere.

features_matrix = df[feature_cols].to_numpy(dtype=np.float32)   # (400400, 55)
tempo_matrix = tempo.astype(np.float32)                          # (400400,)

# Il reshape è sicuro SOLO perché abbiamo verificato al punto 4 che le righe
# sono ordinate per traiettoria consecutiva (traiettoria 0 nelle prime 1001
# righe, traiettoria 1 nelle 1001 successive, ecc.)
#
# PERCHÉ -1 invece di N_TRAJ: come nel notebook ufficiale del docente,
# usiamo -1 per lasciare che numpy calcoli automaticamente il numero di
# traiettorie dal numero totale di righe. È più robusto di scrivere N_TRAJ
# a mano: se in futuro il dataset cambiasse (più o meno traiettorie), il
# reshape continuerebbe a funzionare senza bisogno di modificare il codice,
# perché tutto ciò che serve è che il numero di righe sia divisibile per
# N_TIMESTEPS (già garantito dagli assert della sezione 4).
data_tensor = features_matrix.reshape(-1, N_TIMESTEPS, N_FEATURES)
time_tensor = tempo_matrix.reshape(-1, N_TIMESTEPS)

print(f"\nShape tensore dati (traiettorie, timestep, feature): {data_tensor.shape}")
print(f"Shape tensore tempo (traiettorie, timestep): {time_tensor.shape}")


# ------------------------------------------------------------------------
# 6. SANITY CHECK FISICO
# ------------------------------------------------------------------------
# PERCHÉ: prima di costruire qualunque modello, conviene controllare a mano
# che i valori abbiano senso fisico. Le magnetizzazioni per un sistema di
# spin sono attese in un range piccolo attorno a 0 (tipicamente [-1, 1]);
# la presenza di NaN o Inf indicherebbe un problema nel dataset originale
# o nel parsing, da risolvere ORA e non dopo ore di training ("partire in
# piccolo", vincolo esplicito del docente).

print(f"\nStatistiche sul tensore dati (pre-normalizzazione):")
print(f"  Min:  {data_tensor.min():.4f}")
print(f"  Max:  {data_tensor.max():.4f}")
print(f"  Mean: {data_tensor.mean():.4f}")
print(f"  Std:  {data_tensor.std():.4f}")
print(f"  NaN presenti:  {np.isnan(data_tensor).any()}")
print(f"  Inf presenti:  {np.isinf(data_tensor).any()}")

assert not np.isnan(data_tensor).any(), "Trovati valori NaN nel dataset!"
assert not np.isinf(data_tensor).any(), "Trovati valori Inf nel dataset!"

print("\n✅ Blocco 1 completato: dati caricati, verificati e strutturati "
      f"in tensore {data_tensor.shape}.")

# Verifica rapida che gli indici magnetizzazioni/correlazioni siano coerenti
# con N_FEATURES = 55 (10 + 45): utile ora per non scoprire un errore di
# indicizzazione solo in fase di valutazione finale dei risultati.
n_mag = data_tensor[:, :, MAGNETIZATION_COLS].shape[-1]
n_corr = data_tensor[:, :, CORRELATION_COLS].shape[-1]
print(f"  Colonne magnetizzazioni: {n_mag} (attese {N_QUBITS})")
print(f"  Colonne correlazioni:   {n_corr} (attese {N_QUBITS * (N_QUBITS - 1) // 2})")
assert n_mag == N_QUBITS
assert n_corr == N_QUBITS * (N_QUBITS - 1) // 2

print("Prossimo blocco: split train/val/test per traiettoria + normalizzazione z-score.")