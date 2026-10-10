import { defineConfig } from "@playwright/test";

// Phone test build (var/device_test/site, from `petperception export-device-test`) served as plain
// static files without cross-origin isolation, like the Artifact host. Desktop Chromium: this checks
// the harness logic, not phone behaviour.
const port = Number(process.env.PT_PORT ?? 8790);
export default defineConfig({
  testDir: "tests/devicetest",
  timeout: 600_000,
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: `http://127.0.0.1:${port}/`, launchOptions: { executablePath: process.env.PW_CHROMIUM ?? "/opt/pw-browsers/chromium" } },
  webServer: { command: `python3 -m http.server ${port} --bind 127.0.0.1 --directory ../var/device_test/site`, url: `http://127.0.0.1:${port}/index.html`,
    reuseExistingServer: false, timeout: 30_000 },
});
