import {
  cloneElement,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { CircleHelp } from "lucide-react";
import { cn } from "@/lib/utils";

type TooltipSide = "top" | "right" | "bottom" | "left";
type DescribedElement = ReactElement<{
  "aria-describedby"?: string;
}>;

interface TooltipProps {
  content: ReactNode;
  children: DescribedElement;
  side?: TooltipSide;
  delay?: number;
  className?: string;
}

interface TooltipPosition {
  top: number;
  left: number;
}

const VIEWPORT_MARGIN = 8;
const TOOLTIP_GAP = 8;
const TOOLTIP_OPEN_EVENT = "mindforge:tooltip-open";

function oppositeSide(side: TooltipSide): TooltipSide {
  if (side === "top") return "bottom";
  if (side === "bottom") return "top";
  if (side === "left") return "right";
  return "left";
}

export function Tooltip({
  content,
  children,
  side = "top",
  delay = 250,
  className,
}: TooltipProps) {
  const tooltipId = useId();
  const triggerRef = useRef<HTMLSpanElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const openTimerRef = useRef<number | null>(null);
  const [open, setOpen] = useState(false);
  const [resolvedSide, setResolvedSide] = useState(side);
  const [position, setPosition] = useState<TooltipPosition | null>(null);

  const clearOpenTimer = useCallback(() => {
    if (openTimerRef.current !== null) {
      window.clearTimeout(openTimerRef.current);
      openTimerRef.current = null;
    }
  }, []);

  const show = useCallback(() => {
    clearOpenTimer();
    window.dispatchEvent(
      new CustomEvent<string>(TOOLTIP_OPEN_EVENT, {
        detail: tooltipId,
      }),
    );
    if (delay <= 0) {
      setOpen(true);
      return;
    }
    openTimerRef.current = window.setTimeout(() => {
      openTimerRef.current = null;
      setOpen(true);
    }, delay);
  }, [clearOpenTimer, delay, tooltipId]);

  const hide = useCallback(() => {
    clearOpenTimer();
    setOpen(false);
    setPosition(null);
  }, [clearOpenTimer]);

  useEffect(() => clearOpenTimer, [clearOpenTimer]);

  useEffect(() => {
    const closeOtherTooltip = (event: Event) => {
      const openEvent = event as CustomEvent<string>;
      if (openEvent.detail !== tooltipId) hide();
    };
    window.addEventListener(TOOLTIP_OPEN_EVENT, closeOtherTooltip);
    return () => {
      window.removeEventListener(TOOLTIP_OPEN_EVENT, closeOtherTooltip);
    };
  }, [hide, tooltipId]);

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current;
    const tooltip = tooltipRef.current;
    if (!trigger || !tooltip) return;

    const triggerRect = trigger.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    const available: Record<TooltipSide, number> = {
      top: triggerRect.top,
      right: window.innerWidth - triggerRect.right,
      bottom: window.innerHeight - triggerRect.bottom,
      left: triggerRect.left,
    };
    const required =
      side === "top" || side === "bottom"
        ? tooltipRect.height + TOOLTIP_GAP
        : tooltipRect.width + TOOLTIP_GAP;
    const fallback = oppositeSide(side);
    const nextSide =
      available[side] < required && available[fallback] > available[side]
        ? fallback
        : side;

    let top: number;
    let left: number;
    if (nextSide === "top" || nextSide === "bottom") {
      top =
        nextSide === "top"
          ? triggerRect.top - tooltipRect.height - TOOLTIP_GAP
          : triggerRect.bottom + TOOLTIP_GAP;
      left = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
    } else {
      top = triggerRect.top + triggerRect.height / 2 - tooltipRect.height / 2;
      left =
        nextSide === "left"
          ? triggerRect.left - tooltipRect.width - TOOLTIP_GAP
          : triggerRect.right + TOOLTIP_GAP;
    }

    setResolvedSide(nextSide);
    setPosition({
      top: Math.min(
        Math.max(VIEWPORT_MARGIN, top),
        window.innerHeight - tooltipRect.height - VIEWPORT_MARGIN,
      ),
      left: Math.min(
        Math.max(VIEWPORT_MARGIN, left),
        window.innerWidth - tooltipRect.width - VIEWPORT_MARGIN,
      ),
    });
  }, [side]);

  useLayoutEffect(() => {
    if (!open) return;
    const frame = window.requestAnimationFrame(updatePosition);
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [open, updatePosition]);

  const describedBy = [
    children.props["aria-describedby"],
    open ? tooltipId : undefined,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <>
      <span
        ref={triggerRef}
        className={cn("inline-flex max-w-full", className)}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocusCapture={show}
        onBlurCapture={(event) => {
          if (
            !event.currentTarget.contains(event.relatedTarget as Node | null)
          ) {
            hide();
          }
        }}
        onKeyDown={(event) => {
          if (event.key === "Escape") hide();
        }}
      >
        {cloneElement(children, {
          "aria-describedby": describedBy || undefined,
        })}
      </span>
      {open &&
        typeof document !== "undefined" &&
        createPortal(
          <div
            ref={tooltipRef}
            id={tooltipId}
            role="tooltip"
            data-side={resolvedSide}
            className="mf-tooltip pointer-events-none fixed z-[100] max-w-72 rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-xs leading-5 text-slate-50 shadow-xl"
            style={{
              top: position?.top ?? 0,
              left: position?.left ?? 0,
              visibility: position ? "visible" : "hidden",
            }}
          >
            {content}
          </div>,
          document.body,
        )}
    </>
  );
}

export function HelpTooltip({
  content,
  label = "查看说明",
  side = "top",
}: {
  content: ReactNode;
  label?: string;
  side?: TooltipSide;
}) {
  return (
    <Tooltip content={content} side={side}>
      <span
        tabIndex={0}
        aria-label={
          typeof content === "string" ? `${label}：${content}` : label
        }
        className="inline-flex h-5 w-5 shrink-0 cursor-help items-center justify-center rounded text-text-muted outline-none transition-colors hover:bg-surface-alt hover:text-text focus-visible:ring-2 focus-visible:ring-primary/40"
      >
        <CircleHelp className="h-3.5 w-3.5" aria-hidden="true" />
      </span>
    </Tooltip>
  );
}
