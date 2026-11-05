"""Digital twins of ovarian-cancer patients (TCGA-OV) and an RL agent that learns to treat them.

    data.py    raw clinical export -> one treatment history per patient
    model.py   outcome model: response, progression and survival after a treatment line
    twin.py    one digital twin per patient, built from the 5 most similar training patients
    rl.py      the twins as an RL environment, and a DQN agent trained inside them
    record.py  the patient file uploaded on the demo website

Research prototype on public, de-identified data. Not medical advice.
"""
