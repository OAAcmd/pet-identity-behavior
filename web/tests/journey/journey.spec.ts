// Primary journey with the real server pipeline: pet & concern -> recording -> progress -> answer -> correction -> history.
// Screenshots go to docs/ux/. Needs a running server with a fresh database (first account becomes the administrator).
import { expect, test } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const shots = path.resolve(here, "../../../docs/ux");
const video = process.env.JOURNEY_VIDEO ?? "";
const shortClip = process.env.JOURNEY_SHORT_VIDEO ?? "";
const shot = (page: import("@playwright/test").Page, name: string) => page.screenshot({ path: path.join(shots, `${name}.png`), fullPage: true });

test("ask about a behaviour, read the answer, correct it", async ({ page }) => {
  test.skip(!video, "set JOURNEY_VIDEO");
  await page.goto("/");
  await page.getByLabel("Username").fill("owner");
  await page.getByLabel("Password").fill("correct horse battery staple");
  await page.getByRole("button", { name: /create account|register/i }).click();
  await expect(page.getByRole("link", { name: /Ask about my pet's behaviour/ })).toBeVisible();
  await shot(page, "01_home");

  await page.getByRole("link", { name: /Ask about my pet's behaviour/ }).click();
  await page.getByRole("button", { name: "A cat" }).click();
  await page.getByLabel("What did you notice?").fill("My cat keeps turning in circles while chasing or biting its tail. Why does it do this?");
  await page.getByLabel(/Where or when/).fill("open living room with a carpet");
  await shot(page, "02_pet_and_concern");
  await page.getByRole("button", { name: /Next: add a recording/ }).click();

  if (shortClip) {
    await page.locator('input[type=file]:not([capture])').setInputFiles(shortClip);
    await expect(page.getByText(/shorter than 3 seconds/)).toBeVisible();
    await shot(page, "03b_too_short");
  }
  await page.locator('input[type=file]:not([capture])').setInputFiles(video);
  await expect(page.getByRole("button", { name: "Send and get an answer" })).toBeEnabled({ timeout: 20_000 });
  await shot(page, "03_recording");
  await page.getByRole("button", { name: "Send and get an answer" }).click();

  await expect(page.getByText(/Looking at the recording|Answering your question/).first()).toBeVisible({ timeout: 120_000 });
  await shot(page, "04_progress");
  await expect(page.getByText("What we saw")).toBeVisible({ timeout: 840_000 });
  await shot(page, "05_answer");
  const visible = await page.locator(".answer").innerText();
  expect(visible).not.toMatch(/model|threshold|zero-shot|score|E3[0-9]/i); // no implementation details in the journey

  await page.getByRole("button", { name: "Right" }).first().click();
  await page.getByRole("button", { name: "Tell us what your pet actually did" }).click();
  await page.getByLabel("What did your pet actually do?").selectOption("tail_chasing");
  await page.getByLabel(/just before or after/).fill("It was lying on the sofa first, then spun three times and went back to sleep.");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByText(/Saved/).first()).toBeVisible();
  await shot(page, "06_correction_saved");

  await page.goto("/#/history");
  await expect(page.getByText(/keeps turning in circles/)).toBeVisible();
  await shot(page, "07_history");
});
