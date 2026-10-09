import type { ComponentChildren } from "preact";
import { useEffect, useRef } from "preact/hooks";
import { errorText } from "../lib/api";

export function ErrorBanner({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  if (!error) return null;
  const e = errorText(error);
  return (
    <div class="banner banner-error" role="alert">
      <p class="banner-title">{e.message}</p>
      {e.action && <p class="banner-action">{e.action}</p>}
      <p class="banner-code">Code: {e.code}</p>
      {onRetry && (
        <button type="button" class="btn btn-secondary" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function InfoBanner({ children, tone = "info" }: { children: ComponentChildren; tone?: "info" | "warn" }) {
  return <div class={`banner banner-${tone}`}>{children}</div>;
}

export function ProgressBar({ value, label }: { value: number; label: string }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div class="progress">
      <div class="progress-label" id={`pl-${label.replace(/\W/g, "")}`}>
        {label}
      </div>
      <div class="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} aria-labelledby={`pl-${label.replace(/\W/g, "")}`}>
        <div class="progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function Spinner({ label }: { label: string }) {
  return (
    <div class="spinner-wrap" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ComponentChildren }) {
  return (
    <div class="empty">
      <p class="empty-title">{title}</p>
      {children}
    </div>
  );
}

export function Chip({ children, tone = "neutral", title }: { children: ComponentChildren; tone?: "neutral" | "ok" | "warn" | "bad" | "exp"; title?: string }) {
  return (
    <span class={`chip chip-${tone}`} title={title}>
      {children}
    </span>
  );
}

/** Accessible confirmation dialog using <dialog>. */
export function ConfirmDialog(props: { open: boolean; title: string; body: ComponentChildren; confirmLabel: string; danger?: boolean;
  onConfirm: () => void; onCancel: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (props.open && !d.open) {
      if (typeof d.showModal === "function") d.showModal();
      else d.setAttribute("open", "");
    } else if (!props.open && d.open) {
      d.close();
    }
  }, [props.open]);
  return (
    <dialog ref={ref} class="dialog" aria-labelledby="dlg-title" onCancel={(e) => { e.preventDefault(); props.onCancel(); }}>
      <h2 id="dlg-title">{props.title}</h2>
      <div class="dialog-body">{props.body}</div>
      <div class="dialog-actions">
        <button type="button" class="btn btn-secondary" onClick={props.onCancel}>
          Cancel
        </button>
        <button type="button" class={`btn ${props.danger ? "btn-danger" : "btn-primary"}`} onClick={props.onConfirm}>
          {props.confirmLabel}
        </button>
      </div>
    </dialog>
  );
}

export function Field(props: { label: string; id: string; hint?: string; error?: string | null; children: ComponentChildren }) {
  return (
    <div class="field">
      <label for={props.id}>{props.label}</label>
      {props.children}
      {props.hint && <p class="hint" id={`${props.id}-hint`}>{props.hint}</p>}
      {props.error && (
        <p class="field-error" role="alert">
          {props.error}
        </p>
      )}
    </div>
  );
}
