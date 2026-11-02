export const pct = (p: number, digits = 0) => `${(p * 100).toFixed(digits)}%`;
export const months = (m: number, digits = 1) => `${m.toFixed(digits)} mo`;
export const stageName = (stage: number | null) => (stage ? `Stage ${["", "I", "II", "III", "IV"][stage]}` : "Stage not recorded");
export const gradeName = (grade: number | null) => (grade ? `Grade ${grade}` : "Grade not recorded");
export const organName = (organ: string) =>
  ({
    hematologic: "Blood counts",
    neuropathy: "Nerves (neuropathy)",
    cardiac: "Heart",
    renal: "Kidneys",
    gastrointestinal: "Stomach & gut",
    dermatologic: "Skin",
    vascular: "Blood vessels",
    hypersensitivity: "Allergic reaction",
  })[organ] ?? organ;
