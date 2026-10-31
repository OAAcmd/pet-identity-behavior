// The Android shell's start screen (android-app/www) in mobile Chromium: connects to a running server and opens the web app.
// The real WebView, permissions and system camera need an emulator or a phone (android-app/README.md).
import { expect, test } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const launcher = process.env.LAUNCHER_URL ?? ""; // e.g. http://127.0.0.1:8790/ serving android-app/www

test("launcher reports an unreachable server, then connects", async ({ page, baseURL }) => {
  test.skip(!launcher, "set LAUNCHER_URL");
  await page.goto(launcher);
  await page.getByLabel("Server address").fill("http://127.0.0.1:9");
  await page.getByRole("button", { name: "Connect" }).click();
  await expect(page.getByRole("alert")).toContainText("Could not connect", { timeout: 15_000 });
  await page.screenshot({ path: path.resolve(here, "../../../docs/ux/10_android_launcher_unreachable.png") });
  await page.getByLabel("Server address").fill(baseURL ?? "");
  await page.getByRole("button", { name: /Try again|Connect/ }).click();
  await expect(page).toHaveURL(new RegExp(`^${baseURL}`), { timeout: 15_000 });
  await expect(page.getByText("Pet Observer").first()).toBeVisible();
});
