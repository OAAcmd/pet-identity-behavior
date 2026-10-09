import { render } from "preact";
import { useEffect, useState } from "preact/hooks";
import { Spinner } from "./components/ui";
import { ApiError, api } from "./lib/api";
import { OnDeviceProvider } from "./lib/ondevice";
import { navigate, useRoute } from "./lib/router";
import type { Me } from "./lib/types";
import { AboutPage } from "./pages/About";
import { AnalysisPage } from "./pages/Analysis";
import { AuthPage } from "./pages/Auth";
import { CapturePage } from "./pages/Capture";
import { DeviceCheckPage } from "./pages/DeviceCheck";
import { HistoryPage } from "./pages/History";
import { HomePage } from "./pages/Home";
import { LocalAnalysisPage } from "./pages/LocalAnalysis";
import { PetsPage } from "./pages/Pets";
import { PlacesPage } from "./pages/Places";
import { SettingsPage } from "./pages/Settings";
import { AppProvider, offlineCache, useApp } from "./state";
import "./styles.css";

const NAV = [
  { name: "home", path: "/", label: "Home", icon: "⌂" },
  { name: "capture", path: "/capture", label: "Record", icon: "●" },
  { name: "history", path: "/history", label: "History", icon: "☰" },
  { name: "pets", path: "/pets", label: "Pets", icon: "♥" },
  { name: "settings", path: "/settings", label: "Settings", icon: "⚙" },
];

function Shell() {
  const { me, setMe, toasts, online } = useApp();
  const route = useRoute();
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    api.get<Me>("/auth/me")
      .then((m) => { setMe(m); offlineCache.save("me", { ...m, csrf_token: null }); })
      .catch((e) => {
        // Offline: keep using the last signed-in account on this device (on-device analysis works without the server).
        const cached = e instanceof ApiError && e.isNetwork ? offlineCache.load<Me>("me") : null;
        if (!cached) offlineCache.clear();
        setMe(cached);
      })
      .finally(() => setChecked(true));
  }, [setMe]);

  useEffect(() => {
    // Move focus to the main heading on navigation for screen-reader users.
    const h = document.querySelector<HTMLElement>("main h1");
    h?.focus();
  }, [route.name, route.params.id]);

  if (!checked) return <Spinner label="Loading..." />;
  if (!me) return <AuthPage />;

  let page;
  switch (route.name) {
    case "home": page = <HomePage />; break;
    case "capture": page = <CapturePage />; break;
    case "analysis": page = <AnalysisPage id={route.params.id} />; break;
    case "local": page = <LocalAnalysisPage id={route.params.id} />; break;
    case "device": page = <DeviceCheckPage />; break;
    case "history": page = <HistoryPage />; break;
    case "pets": page = <PetsPage />; break;
    case "places": page = <PlacesPage />; break;
    case "settings": page = <SettingsPage />; break;
    case "about": page = <AboutPage />; break;
    default:
      page = (
        <section>
          <h1 tabIndex={-1}>Page not found</h1>
          <button type="button" class="btn btn-primary" onClick={() => navigate("/")}>Go home</button>
        </section>
      );
  }
  return (
    <div class="shell">
      <a class="skip-link" href="#main">Skip to content</a>
      <header class="topbar">
        <a href="#/" class="brand">Pet Observer</a>
        <span class="badge-exp" title="Interpretations are experimental and unvalidated">Research preview</span>
      </header>
      {!online && <div class="offline" role="status">You are offline. On-device analysis still works once the models are installed; anything waiting to sync stays on this device.</div>}
      <main id="main" class="main">{page}</main>
      <nav class="bottomnav" aria-label="Main">
        {NAV.map((n) => (
          <a key={n.name} href={`#${n.path}`} class={`navitem ${route.name === n.name ? "active" : ""}`} aria-current={route.name === n.name ? "page" : undefined}>
            <span aria-hidden="true" class="navicon">{n.icon}</span>
            <span>{n.label}</span>
          </a>
        ))}
      </nav>
      <div class="toasts" aria-live="polite" role="status">
        {toasts.map((t) => (
          <div key={t.id} class={`toast toast-${t.tone}`}>{t.text}</div>
        ))}
      </div>
    </div>
  );
}

render(
  <AppProvider>
    <OnDeviceProvider>
      <Shell />
    </OnDeviceProvider>
  </AppProvider>,
  document.getElementById("app")!,
);

if ("serviceWorker" in navigator && import.meta.env.PROD) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(() => undefined);
  });
}
