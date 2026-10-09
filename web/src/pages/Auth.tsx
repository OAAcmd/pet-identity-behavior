import { useEffect, useState } from "preact/hooks";
import { ErrorBanner, Field } from "../components/ui";
import { api } from "../lib/api";
import type { Me } from "../lib/types";
import { useApp } from "../state";

export function AuthPage() {
  const { setMe } = useApp();
  const [canRegister, setCanRegister] = useState(false);
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    api.get<{ registration_open: boolean }>("/auth/status").then((s) => {
      setCanRegister(s.registration_open);
      if (s.registration_open) setMode("register");
    }).catch(() => undefined);
  }, []);

  const submit = async (e: Event) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const me = await api.post<Me>(mode === "login" ? "/auth/login" : "/auth/register", { username, password });
      setMe(me);
      location.hash = "#/";
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <main id="main" class="main auth">
      <h1 tabIndex={-1}>Pet Observer</h1>
      <p class="lede">
        Record or upload short clips of your cat or dog. You get timestamped observations of what was seen and heard, plus clearly marked,
        uncertain interpretations. It does not translate animal language or diagnose anything.
      </p>
      <form onSubmit={submit} class="card" aria-describedby="auth-help">
        <h2>{mode === "login" ? "Sign in" : "Create the first account"}</h2>
        {mode === "register" && <p id="auth-help" class="hint">The first account becomes the administrator of this installation.</p>}
        <Field label="Username" id="username">
          <input id="username" name="username" autoComplete="username" required minLength={3} maxLength={32} value={username}
            onInput={(e) => setUsername((e.target as HTMLInputElement).value)} />
        </Field>
        <Field label="Password" id="password" hint={mode === "register" ? "At least 10 characters." : undefined}>
          <input id="password" name="password" type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} required
            minLength={mode === "register" ? 10 : 1} value={password} onInput={(e) => setPassword((e.target as HTMLInputElement).value)} />
        </Field>
        <ErrorBanner error={error} />
        <button class="btn btn-primary btn-block" type="submit" disabled={busy}>
          {busy ? "Please wait..." : mode === "login" ? "Sign in" : "Create account"}
        </button>
        {canRegister && mode === "login" && (
          <button type="button" class="btn btn-link" onClick={() => setMode("register")}>Create the first account instead</button>
        )}
        {mode === "register" && (
          <button type="button" class="btn btn-link" onClick={() => setMode("login")}>I already have an account</button>
        )}
      </form>
    </main>
  );
}
