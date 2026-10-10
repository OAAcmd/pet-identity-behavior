import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));

// The site's index.html is the page body only (the Artifact host adds the document skeleton);
// browsers wrap it the same way.
const CLIP = path.resolve(here, "../../../var/e2e/big_cat_clip.webm");

async function step(page: Page, title: string, button: string, expected = /passed|done/) {
  const li = page.locator("li.step", { has: page.getByRole("heading", { name: title }) });
  await li.getByRole("button", { name: button }).click();
  await expect(li.locator(".chip")).toHaveText(expected, { timeout: 300_000 });
  return li;
}

test("phone test build runs every step without a server and reports it", async ({ page, context }) => {
  await page.goto("index.html");
  await page.getByLabel("Phone model").fill("desktop test (not a phone)");
  await step(page, "Download and verify the models", "Download");
  await step(page, "Start the engine", "Start");
  await step(page, "Check the models", "Check");
  await step(page, "Analyse the reference clip three times", "Analyse");

  await context.setOffline(true);
  const off = await step(page, "Analyse with the network off", "Analyse offline");
  await expect(off.locator(".result")).toContainText("0 network requests");
  await context.setOffline(false);

  await page.getByLabel("What will you record?").selectOption("cat");
  await page.locator("#pt-file").setInputFiles(CLIP);
  const rec = page.locator("li.step", { has: page.getByRole("heading", { name: "Record a clip and analyse it" }) });
  await expect(rec.locator(".chip")).toHaveText("done", { timeout: 300_000 });
  await expect(rec.locator(".result")).toContainText("cat");

  await page.reload();
  await step(page, "Reload and start from saved models", "Start from saved models");

  await page.getByText("Show report").click();
  const report = JSON.parse((await page.locator("pre.report").textContent()) ?? "{}");
  expect(report.schema).toBe("petperception.device_check/3");
  expect(report.functional_pass).toBe(true);
  expect(report.tester.device_model).toBe("desktop test (not a phone)");
  expect(report.offline.network_requests_during_analysis.requests).toBe(0);
  expect(report.recordings).toHaveLength(1);
  expect(report.recordings[0].tester_says_recorded).toBe("cat");
  expect(report.engine.reload_from_cache_ms).toBeGreaterThan(0);
  expect(report.runtime_backend.variant).toBeTruthy();
  expect(report.model_files.length).toBeGreaterThanOrEqual(3);
  expect(report.models.detector_float.argmax_equal).toBe(true);
  fs.mkdirSync(path.resolve(here, "../../../var/device_test"), { recursive: true });
  fs.writeFileSync(path.resolve(here, "../../../var/device_test/desktop_chromium_report.json"), JSON.stringify(report, null, 1));

  await page.getByRole("button", { name: "Send report" }).click();
  await expect(page.getByRole("status").last()).toContainText("claude.ai");
});
