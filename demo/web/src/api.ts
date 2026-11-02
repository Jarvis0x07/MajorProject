// Typed client for the Python backend (demo/server.py).

export interface Medication {
  key: string;
  name: string;
  group: string;
}

export interface ProjectInfo {
  cohort: { patients: number; patients_with_lines: number; treatment_lines: number; stored_twins: number; neighbors_per_twin: number };
  twin_accuracy: {
    response_auroc: number;
    survival_c_index: number;
    progression_c_index: number;
    mortality_12m_predicted: number;
    mortality_12m_observed: number;
  };
  neighbor_twins: { validation_progression_brier_shared: number; validation_progression_brier_neighbor: number };
  simulation_fidelity: { months: number[]; simulated: number[]; observed: number[]; patients: number };
  rl: {
    learning_curve: { episode: number; mean_return: number }[];
    first_line_survival_months: Record<"random" | "clinician" | "guideline" | "dqn", number>;
    test_patients: number;
  };
  medications: Medication[];
  twin_map: [number, number][];
}

export interface Neighbor {
  rank: number;
  similarity: number;
  map_index: number;
  age: number | null;
  stage: number | null;
  grade: number | null;
  residual: string | null;
  treatment_line: number;
  situation: string;
  prior_classes: string[];
}

export interface TwinSummary {
  id: string;
  label: string;
  today_month: number;
  map_position: [number, number];
  patient: {
    age_now: number | null;
    stage: number | null;
    grade: number | null;
    residual: string | null;
    treatment_line: number;
    situation: string;
    last_tumor_status: string;
    months_since_last_platinum: number | null;
    platinum_status: string;
    past_treatments: { line: number; start_month: number; classes: string[] }[];
  };
  build: { database_size: number; neighbors: Neighbor[]; progression_risk_multiplier: number };
}

export interface Prediction {
  medication: Medication;
  response_probability: number;
  progression_free: Record<"6m" | "12m" | "24m", number>;
  expected_progression_free_months: number;
  survival: Record<"6m" | "12m" | "24m", number>;
  expected_survival_months: number;
  side_effects: { any_severe: number; by_organ: { organ: string; probability: number; level: string }[] };
  uncertainty: { survival_months_std: number; progression_free_months_std: number; response_std: number };
  allowed_by_safety_rules: boolean;
  guideline_flags: string[];
}

export interface ComparisonRow {
  key: string;
  name: string;
  group: string;
  response_probability: number;
  expected_progression_free_months: number;
  expected_survival_months: number;
  any_severe_side_effect: number;
  survival_months_std: number;
  guideline_flags: string[];
}

export interface JourneyStep {
  medication: string;
  months: number;
  response: boolean;
  severe_side_effects: string[];
  ended_with: string;
}

export interface Simulation {
  medication: string;
  runs: number;
  months: number[];
  alive: number[];
  progression_free_first_line: number[];
  median_survival_months: number | null;
  alive_at_5_years: number;
  mean_treatment_lines: number;
  mean_severe_side_effects: number;
  example_journeys: JourneyStep[][];
  later_lines_policy: string;
}

export interface AiDoctor {
  agent: string;
  choice: string;
  choice_key: string;
  ranking: { key: string; name: string; value: number }[];
  excluded_by_constraints: string[];
  note: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
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
  return response.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  info: () => request<ProjectInfo>("/api/info"),
  createTwin: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<TwinSummary>("/api/twins", { method: "POST", body: form });
  },
  predict: (id: string, medication: string) => request<Prediction>(`/api/twins/${id}/predict`, json({ medication })),
  compare: (id: string) => request<ComparisonRow[]>(`/api/twins/${id}/compare`),
  simulate: (id: string, medication: string, runs = 300) =>
    request<Simulation>(`/api/twins/${id}/simulate`, json({ medication, runs })),
  aiDoctor: (id: string) => request<AiDoctor>(`/api/twins/${id}/ai-doctor`),
};
