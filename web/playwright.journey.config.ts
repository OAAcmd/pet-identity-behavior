import { defineConfig } from "@playwright/test";

// The user journey against a running server with the real analysis pipeline (see docs/UX_FLOW.md):
//   petperception serve --port 8765 --workers 1   (fresh database), then
//   JOURNEY_URL=http://127.0.0.1:8765 JOURNEY_VIDEO=<clip> npx playwright test -c playwright.journey.config.ts
export default defineConfig({
  testDir: "tests/journey",
  timeout: 900_000,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: process.env.JOURNEY_URL ?? "http://127.0.0.1:8765",
    viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true,
    userAgent: "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Mobile Safari/537.36",
    launchOptions: { executablePath: process.env.PW_CHROMIUM ?? "/opt/pw-browsers/chromium" },
  },
});
