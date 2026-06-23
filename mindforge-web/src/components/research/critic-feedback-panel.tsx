import type { CriticScore } from "@/types/research";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
} from "recharts";

interface Props {
  score: CriticScore | null;
}

export function CriticFeedbackPanel({ score }: Props) {
  if (!score) return null;

  const data = [
    { dimension: "完整性", value: score.completeness ?? 0 },
    { dimension: "准确性", value: score.accuracy ?? 0 },
    { dimension: "深度", value: score.depth ?? 0 },
    { dimension: "清晰度", value: score.clarity ?? 0 },
    { dimension: "引用质量", value: score.citations ?? 0 },
  ];

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <h4 className="text-sm font-semibold">评论家评分</h4>
        <span
          className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
            score.overall >= 7
              ? "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300"
              : "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300"
          }`}
        >
          {score.overall.toFixed(1)} / 10
        </span>
      </div>

      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data}>
            <PolarGrid stroke="#e9ecef" />
            <PolarAngleAxis
              dataKey="dimension"
              tick={{ fontSize: 11, fill: "#636e72" }}
            />
            <PolarRadiusAxis
              domain={[0, 10]}
              tick={{ fontSize: 9, fill: "#636e72" }}
            />
            <Radar
              name="评分"
              dataKey="value"
              stroke="#6c5ce7"
              fill="#6c5ce7"
              fillOpacity={0.2}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {score.issues && score.issues.length > 0 && (
        <div className="mt-3 text-xs text-text-muted leading-relaxed border-t border-border pt-3">
          {score.issues.slice(0, 3).map((issue, i) => (
            <p key={i} className="mb-1">• {issue}</p>
          ))}
        </div>
      )}
    </div>
  );
}
