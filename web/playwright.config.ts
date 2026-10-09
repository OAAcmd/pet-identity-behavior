import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Automated browser tests (Chromium only, emulated mobile viewports).
// They are NOT real-device tests: no iOS Safari/WebKit or physical Android
// hardware is exercised here. See docs/TEST_REPORT.md.
const here = path.dirname(fileURLToPath(import.meta.url));
const fixtures = path.resolve(here, "../var/e2e");
const port = Number(process.env.E2E_PORT ?? 8765);
const chromium = {
  executablePath: process.env.PW_CHROMIUM ?? "/opt/pw-browsers/chromium",
  args: [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    `--use-file-for-fake-video-capture=${fixtures}/cat_camera.y4m`,
    `--use-file-for-fake-audio-capture=${fixtures}/meow_mic.wav`,
  ],
};

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 180_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["json", { outputFile: "test-results/e2e-report.json" }]],
  use: { baseURL: `http://127.0.0.1:${port}`, trace: "retain-on-failure", video: "off", screenshot: "only-on-failure" },
  projects: [
    { name: "android-pixel7-chromium", use: { ...devices["Pixel 7"], launchOptions: chromium } },
    { name: "iphone-viewport-chromium", use: { ...devices["iPhone 13"], browserName: "chromium", defaultBrowserType: "chromium", launchOptions: chromium },
      grep: /@layout/ },
    { name: "desktop-chromium", use: { ...devices["Desktop Chrome"], launchOptions: chromium }, grep: /@layout/ },
  ],
  webServer: {
    command: "./tests/e2e/server.sh",
    url: `http://127.0.0.1:${port}/api/v1/health`,
    reuseExistingServer: false,
    timeout: 120_000,
    env: { E2E_PORT: String(port) },
  },
});
