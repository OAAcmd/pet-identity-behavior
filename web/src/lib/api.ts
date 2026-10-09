// Thin typed client for the backend API. Same origin; session cookie + CSRF header.

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public action: string | null = null,
    public details: Record<string, unknown> | null = null,
  ) {
    super(message);
  }
  get isNetwork(): boolean {
    return this.status === 0;
  }
}

let csrfToken: string | null = null;
export function setCsrf(token: string | null): void {
  csrfToken = token;
}

const BASE = "/api/v1";

async function parseError(resp: Response): Promise<ApiError> {
  try {
    const body = await resp.json();
    const e = body?.error;
    if (e && typeof e.code === "string") {
      return new ApiError(resp.status, e.code, e.message ?? "Request failed", e.action ?? null, e.details ?? null);
    }
  } catch {
    /* non-JSON error */
  }
  return new ApiError(resp.status, `http_${resp.status}`, `Request failed (${resp.status}).`, "Try again.");
}

export async function request<T>(method: string, path: string, body?: unknown, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json", ...(init.headers as Record<string, string> | undefined) };
  let payload: BodyInit | undefined;
  if (body instanceof Blob || body instanceof ArrayBuffer) {
    payload = body as BodyInit;
    headers["Content-Type"] = "application/octet-stream";
  } else if (body !== undefined) {
    payload = JSON.stringify(body);
    headers["Content-Type"] = "application/json";
  }
  if (method !== "GET" && csrfToken) headers["X-CSRF-Token"] = csrfToken;
  let resp: Response;
  try {
    resp = await fetch(BASE + path, { method, headers, body: payload, credentials: "same-origin", ...init });
  } catch {
    throw new ApiError(0, "network_error", "Could not reach the server.", "Check your connection; the app will retry.");
  }
  if (resp.status === 204) return undefined as T;
  if (!resp.ok) throw await parseError(resp);
  return (await resp.json()) as T;
}

export const api = {
  get: <T>(p: string) => request<T>("GET", p),
  post: <T>(p: string, b?: unknown) => request<T>("POST", p, b),
  patch: <T>(p: string, b?: unknown) => request<T>("PATCH", p, b),
  put: <T>(p: string, b?: unknown) => request<T>("PUT", p, b),
  del: <T = void>(p: string) => request<T>("DELETE", p),
};

export function errorText(e: unknown): { message: string; action: string | null; code: string } {
  if (e instanceof ApiError) return { message: e.message, action: e.action, code: e.code };
  const o = e as { message?: unknown; action?: unknown; code?: unknown } | null;
  if (o && typeof o.message === "string") {
    return { message: o.message, action: typeof o.action === "string" ? o.action : null, code: typeof o.code === "string" ? o.code : "client_error" };
  }
  return { message: "Something went wrong.", action: "Reload the page and try again.", code: "client_error" };
}
