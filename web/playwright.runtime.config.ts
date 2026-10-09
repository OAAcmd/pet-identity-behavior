import { defineConfig } from "@playwright/test";

// Model runtime parity in desktop headless Chromium (not a phone).
export default defineConfig({
  testDir: "tests/runtime",
  timeout: 300_000,
  workers: 1,
  reporter: [["list"]],
  use: { launchOptions: { executablePath: process.env.PW_CHROMIUM ?? "/opt/pw-browsers/chromium" } },
});
