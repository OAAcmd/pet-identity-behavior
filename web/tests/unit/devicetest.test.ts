import { describe, expect, it } from "vitest";
import { fromBase64, parseUserAgent, sha256Hex } from "../../src/devicetest/harness";

describe("phone test harness", () => {
  it("reads OS and browser from phone user agents and flags in-app web views", () => {
    const iphone = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1";
    expect(parseUserAgent(iphone)).toEqual({ os_guess: "iOS 17.5.1", browser_guess: "Safari 17.5", in_app_webview: false });
    const pixel = "Mozilla/5.0 (Linux; Android 14; Pixel 7a) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.6478.122 Mobile Safari/537.36";
    expect(parseUserAgent(pixel)).toEqual({ os_guess: "Android 14", browser_guess: "Chrome 126.0.6478.122", in_app_webview: false });
    const webview = "Mozilla/5.0 (Linux; Android 13; SM-A546B; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/125.0.0.0 Mobile Safari/537.36";
    expect(parseUserAgent(webview).in_app_webview).toBe(true);
    const iosApp = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148";
    expect(parseUserAgent(iosApp)).toMatchObject({ os_guess: "iOS 17.4", in_app_webview: true });
    const chromeIos = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/126.0.6478.54 Mobile/15E148 Safari/604.1";
    expect(parseUserAgent(chromeIos).browser_guess).toBe("Chrome (iOS) 126.0.6478.54");
  });

  it("decodes Base64 model parts to the original bytes", async () => {
    const bytes = new Uint8Array(1000).map((_, i) => (i * 37) % 256);
    const text = Buffer.from(bytes).toString("base64").replace(/(.{76})/g, "$1\n");
    const back = fromBase64(text);
    expect(Array.from(back)).toEqual(Array.from(bytes));
    expect(await sha256Hex(back)).toBe(await sha256Hex(bytes));
  });
});
