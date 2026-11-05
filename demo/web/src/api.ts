// Typed client for the Python backend (demo/server.py): one call, upload a record and get the analysis.

export interface Medication {
  key: string;
  name: string;
  group: string;
  guideline_caution: boolean;
  response: number;
  response_std: number;
  progression_free_months: number;
  progression_free_months_std: number;
  survival_months: number;
  survival_months_std: number;
  progression_free_12m: number;
  survival_12m: number;
  survival_24m: number;
}

export interface Neighbour {
  similarity: number;
  age: number | null;
  stage: number | null;
  treatment_line: number;
  situation: string;
  past_drugs: string[];
}

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
  risk_multipliers: { progression: number; death: number };
  medications: Medication[];
  agent_choice: string;
}

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
