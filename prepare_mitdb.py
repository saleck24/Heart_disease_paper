import os
import wfdb
import numpy as np
from tqdm import tqdm

def download_and_prepare_mitdb():
    processed_dir = 'data/mitdb/processed'
    os.makedirs(processed_dir, exist_ok=True)

    # Liste standard des enregistrements MIT-BIH (48 patients)
    records = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]

    # Mapping AAMI (5 classes standardisées)
    aami_mapping = {
        'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
        'A': 1, 'a': 1, 'J': 1, 'S': 1,
        'V': 2, 'E': 2,
        'F': 3,
        '/': 4, 'f': 4, 'Q': 4
    }

    window_before = 90
    window_after  = 96  # 90 + 1 + 96 = 187 points (comme Kaggle)

    all_beats    = []
    all_labels   = []
    all_patients = []

    print("Téléchargement + extraction des battements depuis PhysioNet (patient par patient)...")
    for record in tqdm(records, desc="Patients"):
        try:
            # Lecture du signal directement depuis PhysioNet (streaming, pas de téléchargement bulk)
            rec = wfdb.rdrecord(record, pn_dir='mitdb')
            ann = wfdb.rdann(record, 'atr', pn_dir='mitdb')

            # Choisir le canal MLII si disponible, sinon le premier
            lead_idx = 0
            if 'MLII' in rec.sig_name:
                lead_idx = rec.sig_name.index('MLII')

            sig = rec.p_signal[:, lead_idx]

            for sym, pos in zip(ann.symbol, ann.sample):
                if sym in aami_mapping:
                    start = pos - window_before
                    end   = pos + window_after + 1
                    if start >= 0 and end <= len(sig):
                        beat = sig[start:end].copy()
                        # Normalisation Min-Max
                        bmin, bmax = beat.min(), beat.max()
                        if bmax - bmin > 0:
                            beat = (beat - bmin) / (bmax - bmin)
                        all_beats.append(beat)
                        all_labels.append(aami_mapping[sym])
                        all_patients.append(record)

        except Exception as e:
            tqdm.write(f"  [ERREUR] Record {record} : {e}")

    X      = np.array(all_beats)
    y      = np.array(all_labels)
    groups = np.array(all_patients)

    print(f"\nExtraction terminée : {len(X)} battements depuis {len(set(all_patients))} patients.")

    np.save(os.path.join(processed_dir, 'X.npy'),      X)
    np.save(os.path.join(processed_dir, 'y.npy'),      y)
    np.save(os.path.join(processed_dir, 'groups.npy'), groups)
    print(f"Données sauvegardées dans '{processed_dir}/'.")

if __name__ == "__main__":
    download_and_prepare_mitdb()
