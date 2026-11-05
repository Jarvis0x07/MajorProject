export const pct = (p: number) => `${Math.round(p * 100)}%`;
export const stageName = (stage: number | null) => (stage ? `Stage ${["", "I", "II", "III", "IV"][stage]}` : "Stage not recorded");
export const gradeName = (grade: number | null) => (grade ? `Grade ${grade}` : "Grade not recorded");
