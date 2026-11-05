// Typed client for the Python backend (demo/server.py): one call, upload a record and get the analysis.

// What the twin predicts for one medication (see DigitalTwin.predict in digital_twin_rl/twin.py).
// "_std" values are the spread between the twin's 10 networks (uncertainty).
export interface Medication {
  key: string; // drug classes, e.g. "platinum+taxanes"
  name: string; // display name, e.g. "Carboplatin + Paclitaxel"
  group: string;
  guideline_caution: boolean; // true = guidelines advise against it (platinum when platinum-resistant)
  response: number; // chance the tumour disappears
  response_std: number;
  progression_free_months: number; // expected months until it returns, within 5 years
  progression_free_months_std: number;
  survival_months: number; // expected months alive within 5 years
  survival_months_std: number;
  progression_free_12m: number;
  survival_12m: number; // chance of being alive after 1 year
  survival_24m: number;
}

// One of the 5 similar training patients the twin was built from (profile only, no outcomes).
export interface Neighbour {
  similarity: number;
  age: number | null;
  stage: number | null;
  treatment_line: number;
  situation: string;
  past_drugs: string[];
}

// The full answer for one uploaded patient.
export interface Analysis {
  label: string;
  today_month: number;
  patient: {
    age: number | null;
    stage: number | null;
    grade: number | null;
    residual: string;
    treatment_line: number;
    situation: string;
    months_since_platinum: number | null;
    platinum_resistant: boolean;
    past_treatments: { start_month: number; drugs: string[] }[];
  };
  neighbours: Neighbour[];
  risk_multipliers: { progression: number; death: number }; // from the 5 similar patients; 1 = no change
  medications: Medication[]; // most effective first
  agent_choice: string; // key of the medication the RL agent picks
}

// Send the chosen file to the backend and return its analysis. If the backend refuses the file,
// throw an error carrying the backend's explanation (e.g. "'figo_stage' must be one of ...").
export async function analyse(file: File): Promise<Analysis> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/analyse", { method: "POST", body: form });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* keep the HTTP status text */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<Analysis>;
}
