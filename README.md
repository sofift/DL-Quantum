# DL-Quantum – Predict Dynamics for better chips

Progetto per il corso di **Deep Learning** (Laurea Magistrale AI & CS, Unical, A.Y. 2025/2026 –
Gianluigi Greco, Carlo Adornetto), traccia **Quantum**, piano da 9 CFU.

## Obiettivo

Usare Deep Learning per prevedere la dinamica quantistica (magnetizzazioni e correlazioni) di un
sistema di 10 qubit oltre l'orizzonte temporale raggiungibile con simulazioni tradizionali.

Sfide affrontate:
1. Minimizzare la time window di input mantenendo previsioni accurate.
2. Abilitare la super-resolution (predire a risoluzione temporale più fine del dt campionato).
3. Estrarre mappe di attenzione per interpretare le previsioni.

## Dataset

`trajectories.csv` — dinamica del modello PXP, N=10 qubit, 400 traiettorie, dt=0.02, T_fin=20,
N_points=1001. 56 colonne (1 tempo + 10 magnetizzazioni + 45 correlazioni). 400400 righe totali.

**Il dataset NON è versionato in questo repository** (vedi `.gitignore`): risiede su Google Drive
in `deep learning/Quantum Dynamics/data/trajectories.csv` e viene montato/letto da lì nei notebook
Colab.

## Modelli

- **RNN** (Lab5 – RNN and NLP)
- **Transformer** (Lab7 – Attention and Transformers)

Entrambi addestrati nello stesso train loop con fasi combinate e parametriche (teacher forcing +
masked modeling + scheduled sampling). Solo architetture/classi viste nei Lab del corso, nessun
modello pre-addestrato.

## Struttura del repository

```
DL-Quantum/
├── notebooks/     # notebook Colab (esplorazione, preprocessing, modelli, hp tuning, valutazione)
├── src/           # moduli riutilizzabili (models/, utils/) importati dai notebook
├── results/       # figure (.jpeg, >=300dpi) e tabelle dei risultati
├── report/        # report finale (PDF, max 10 pagine) e materiale per la presentazione
└── README.md
```

## Come lavorare su questo progetto (Colab + GitHub)

All'inizio di ogni sessione Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/sofift/DL-Quantum.git
%cd DL-Quantum
```

Se il repo è già clonato nella sessione, `!git pull` invece di `clone`.

Import dei moduli in `src/` dentro un notebook:

```python
import sys
sys.path.append('/content/DL-Quantum/src')
from models.rnn import RNNForecaster
```

Lettura del dataset da Drive:

```python
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/deep learning/Quantum Dynamics/data/trajectories.csv',
                  header=None, index_col=False)
```

Prima di chiudere la sessione, salvare il lavoro:

```python
!git add .
!git commit -m "descrizione della modifica"
!git push
```

## Regole del progetto (vincolanti)

- Nessun modello pre-addestrato (salvo confronto aggiuntivo esplicito).
- Solo classi/architetture viste nei Lab1-8 del corso.
- Seed fissati ovunque per riproducibilità.
- Almeno 3 configurazioni di iperparametri e 2 time window per ciascun modello, tutti i risultati
  riportati (best in grassetto, media + std o CI 95%).
- Grafici in `.jpeg`, almeno 300 dpi.
