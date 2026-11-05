// Small helpers that turn numbers from the backend into text for the page.

// 0.564 -> "56%"
export const pct = (p: number) => `${Math.round(p * 100)}%`;
// 3 -> "Stage III"
export const stageName = (stage: number | null) => (stage ? `Stage ${["", "I", "II", "III", "IV"][stage]}` : "Stage not recorded");
// 3 -> "Grade 3"
export const gradeName = (grade: number | null) => (grade ? `Grade ${grade}` : "Grade not recorded");
