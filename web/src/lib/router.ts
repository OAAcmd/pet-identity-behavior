// Tiny hash router: #/path/with/:params

import { useEffect, useState } from "preact/hooks";

export interface Route {
  name: string;
  params: Record<string, string>;
}

const PATTERNS: [string, RegExp, string[]][] = [
  ["home", /^\/?$/, []],
  ["capture", /^\/capture$/, []],
  ["analysis", /^\/analyses\/([A-Za-z0-9]+)$/, ["id"]],
  ["history", /^\/history$/, []],
  ["pets", /^\/pets$/, []],
  ["places", /^\/places$/, []],
  ["settings", /^\/settings$/, []],
  ["about", /^\/about$/, []],
];

export function parseHash(hash: string): Route {
  const path = hash.replace(/^#/, "").split("?")[0] || "/";
  for (const [name, re, keys] of PATTERNS) {
    const m = re.exec(path);
    if (m) return { name, params: Object.fromEntries(keys.map((k, i) => [k, decodeURIComponent(m[i + 1])])) };
  }
  return { name: "not_found", params: {} };
}

export function navigate(path: string): void {
  location.hash = `#${path}`;
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(location.hash));
  useEffect(() => {
    const on = () => setRoute(parseHash(location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}
