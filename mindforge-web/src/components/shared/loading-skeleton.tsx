import { cn } from "@/lib/utils";

type SkeletonVariant = "card" | "table-row" | "chart" | "text";

interface Props {
  variant?: SkeletonVariant;
  className?: string;
  count?: number;
}

const base = "animate-pulse rounded bg-border/60";

export function LoadingSkeleton({
  variant = "text",
  className,
  count = 1,
}: Props) {
  const items = Array.from({ length: count }, (_, i) => i);

  switch (variant) {
    case "card":
      return (
        <div className={cn("flex flex-col gap-4", className)}>
          {items.map((i) => (
            <div key={i} className={cn(base, "h-32 rounded-lg")} />
          ))}
        </div>
      );
    case "table-row":
      return (
        <div className={cn("flex flex-col gap-3", className)}>
          {items.map((i) => (
            <div key={i} className={cn(base, "h-12 rounded-md")} />
          ))}
        </div>
      );
    case "chart":
      return (
        <div className={cn(base, "h-64 rounded-lg", className)} />
      );
    default:
      return (
        <div className={cn("flex flex-col gap-2", className)}>
          {items.map((i) => (
            <div
              key={i}
              className={cn(base, "h-5 rounded", i === items.length - 1 && "w-3/4")}
            />
          ))}
        </div>
      );
  }
}
