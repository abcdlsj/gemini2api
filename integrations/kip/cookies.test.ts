/**
 * The Firefox cookie reader Kip vendors. Run with `bun test integrations/kip`.
 * Kip has its own copy of the module under the same assertions; these tests are
 * the source of truth for the behaviour both rely on.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { Database } from "bun:sqlite";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { firefoxProfiles, firefoxRoots, readFirefoxCookies } from "./cookies";

let dir = "";

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "gemini-cookies-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

function profile(
  root: string,
  name: string,
  rows: Array<[string, string, string]>,
): string {
  const path = join(root, "Profiles", name);
  mkdirSync(path, { recursive: true });
  const database = new Database(join(path, "cookies.sqlite"), { create: true });
  database.exec("create table moz_cookies (host text, name text, value text)");
  const insert = database.prepare("insert into moz_cookies values (?, ?, ?)");
  for (const [host, cookie, value] of rows) insert.run(host, cookie, value);
  database.close();
  return path;
}

const SESSION: Array<[string, string, string]> = [
  [".google.com", "__Secure-1PSID", "psid"],
  [".google.com", "__Secure-1PSIDTS", "psidts"],
  [".google.com", "SID", "sid"],
];

describe("Firefox cookies", () => {
  test("reads Google cookies and drops analytics", () => {
    const root = join(dir, "Firefox");
    profile(root, "kip.default", [
      ...SESSION,
      [".google.com", "_ga", "analytics"],
      [".google.com", "_ga_CUSTOM123", "analytics-ga4"],
      ["example.com", "unrelated", "x"],
    ]);

    const jar = readFirefoxCookies([root]);

    expect(Object.keys(jar.cookies)).toHaveLength(3);
    expect(jar.cookies["__Secure-1PSID"]).toBe("psid");
    expect(jar.cookies["_ga"]).toBeUndefined();
    expect(jar.cookies["_ga_CUSTOM123"]).toBeUndefined();
    expect(jar.profile.endsWith("kip.default")).toBe(true);
  });

  test("reads cookies from a live WAL-mode database", () => {
    const root = join(dir, "Firefox-WAL");
    const path = profile(root, "live.default", SESSION);
    const live = new Database(join(path, "cookies.sqlite"));
    try {
      live.exec("PRAGMA journal_mode = WAL;");
      live.prepare("insert into moz_cookies values (?, ?, ?)").run(
        ".google.com",
        "SIDCC",
        "fresh",
      );
      expect(readFirefoxCookies([root]).cookies["SIDCC"]).toBe("fresh");
    } finally {
      live.close();
    }
  });

  test("a profile without a Gemini session is refused", () => {
    const root = join(dir, "empty");
    profile(root, "x.default", [["example.com", "unrelated", "x"]]);
    expect(() => readFirefoxCookies([root])).toThrow(/No Firefox profile/);
  });

  test("picks the signed-in profile, not the one with more cookies", () => {
    const root = join(dir, "multi");
    profile(
      root,
      "logged-out.default",
      Array.from({ length: 9 }, (_, index) => [".google.com", `NID${index}`, "x"] as const),
    );
    profile(root, "signed-in.default", SESSION);

    const jar = readFirefoxCookies([root]);
    expect(jar.profile.endsWith("signed-in.default")).toBe(true);
    expect(jar.cookies["__Secure-1PSID"]).toBe("psid");
  });

  test("an unreadable profile does not fail the read", () => {
    const root = join(dir, "broken");
    const path = profile(root, "broken.default", []);
    writeFileSync(join(path, "cookies.sqlite"), "not a database");
    profile(root, "good.default", SESSION);

    expect(readFirefoxCookies([root]).cookies["__Secure-1PSID"]).toBe("psid");
  });

  test("roots are platform-specific and non-empty", () => {
    expect(firefoxRoots().length).toBeGreaterThan(0);
    expect(firefoxProfiles([join(dir, "missing")])).toEqual([]);
  });
});
