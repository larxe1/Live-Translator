import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, "..");
const EXT_DIR = path.join(ROOT_DIR, "extension");

const background = require("../extension/background.js");
const content = require("../extension/content.js");
const overlay = require("../extension/overlay.js");

function createMockStorage(initial = {}) {
  const data = { ...initial };
  return {
    _data: data,
    async get(keys) {
      if (keys == null) return { ...data };
      const list = Array.isArray(keys) ? keys : [keys];
      const out = {};
      for (const k of list) {
        if (k in data) out[k] = data[k];
      }
      return out;
    },
    async set(values) {
      Object.assign(data, values);
    },
    async remove(keys) {
      const list = Array.isArray(keys) ? keys : [keys];
      for (const k of list) {
        delete data[k];
      }
    },
  };
}

function createJsonResponse(payload, status = 200) {
  const text = JSON.stringify(payload);
  return {
    ok: status >= 200 && status < 300,
    status,
    async text() {
      return text;
    },
  };
}

function loadBackgroundSandbox({ storageInitial = {}, fetchImpl, sendNativeImpl } = {}) {
  const storage = createMockStorage(storageInitial);
  const listeners = { onMessage: [] };
  const nativeCalls = [];

  const chromeMock = {
    storage: { local: storage },
    alarms: {
      async create() {},
      onAlarm: { addListener() {} },
    },
    runtime: {
      id: "test-extension-id",
      lastError: null,
      getManifest() {
        return JSON.parse(fs.readFileSync(path.join(EXT_DIR, "manifest.json"), "utf8"));
      },
      sendNativeMessage(hostName, payload, callback) {
        nativeCalls.push({ hostName, payload });
        if (typeof sendNativeImpl === "function") {
          return sendNativeImpl(hostName, payload, callback, chromeMock.runtime);
        }
        return Promise.resolve({ ok: true, started: true, already_running: false, pid: 1234 });
      },
      onMessage: {
        addListener(fn) {
          listeners.onMessage.push(fn);
        },
      },
      onInstalled: { addListener() {} },
      onStartup: { addListener() {} },
    },
    contextMenus: {
      removeAll(cb) {
        if (typeof cb === "function") cb();
      },
      create() {},
      onClicked: { addListener() {} },
    },
    tabs: { async sendMessage() {} },
    declarativeNetRequest: { async updateSessionRules() {} },
  };

  const sandbox = {
    chrome: chromeMock,
    fetch: fetchImpl || (async () => createJsonResponse({})),
    setTimeout,
    clearTimeout,
    AbortController,
    URL,
    Date,
    Math,
    JSON,
    String,
    Number,
    Boolean,
    Array,
    Object,
    Map,
    Set,
    Promise,
    Error,
    Uint8Array,
    Buffer,
    console,
  };
  sandbox.globalThis = sandbox;

  const src = fs.readFileSync(path.join(EXT_DIR, "background.js"), "utf8");
  vm.runInNewContext(src, sandbox, { filename: "extension/background.js" });

  return {
    storage,
    nativeCalls,
    chromeMock,
    dispatchMessage(message, sender = {}) {
      return new Promise((resolve) => {
        const handler = listeners.onMessage[0];
        const keepOpen = handler(message, sender, (resp) => {
          resolve({ keepOpen, resp });
        });
        assert.equal(keepOpen, true, "onMessage listener must return true");
      });
    },
  };
}

function createMinimalDom(href = "https://mangadex.org/chapter/101") {
  function makeEl(tagName) {
    const el = {
      tagName: String(tagName).toUpperCase(),
      children: [],
      parentNode: null,
      style: {},
      dataset: {},
      attributes: new Map(),
      listeners: new Map(),
      className: "",
      textContent: "",
      title: "",
      complete: true,
      naturalWidth: 800,
      naturalHeight: 1200,
      width: 800,
      height: 1200,
      src: "",
      classList: {
        toggle() {},
        contains() {
          return false;
        },
      },
      appendChild(child) {
        child.parentNode = el;
        el.children.push(child);
        return child;
      },
      removeChild(child) {
        el.children = el.children.filter((c) => c !== child);
        child.parentNode = null;
        return child;
      },
      addEventListener(type, fn) {
        if (!el.listeners.has(type)) el.listeners.set(type, []);
        el.listeners.get(type).push(fn);
      },
      dispatchEvent(evt) {
        for (const fn of el.listeners.get(evt.type) || []) {
          fn(evt);
        }
      },
      setAttribute(k, v) {
        el.attributes.set(k, String(v));
      },
      getAttribute(k) {
        return el.attributes.get(k) || "";
      },
      querySelectorAll(selector) {
        const out = [];
        for (const child of el.children) {
          if (selector.split(",").some((s) => s.trim().toUpperCase() === child.tagName)) {
            out.push(child);
          }
          out.push(...child.querySelectorAll(selector));
        }
        return out;
      },
      getBoundingClientRect() {
        return { left: 0, top: 0, width: el.width, height: el.height };
      },
    };
    return el;
  }

  const doc = {
    documentElement: makeEl("html"),
    body: makeEl("body"),
    createElement: makeEl,
    querySelectorAll(sel) {
      return doc.body.querySelectorAll(sel);
    },
    addEventListener() {},
    removeEventListener() {},
  };
  doc.documentElement.appendChild(doc.body);

  const u = new URL(href);
  const win = {
    location: { href, hostname: u.hostname, pathname: u.pathname, reload() {} },
    addEventListener() {},
    removeEventListener() {},
    getComputedStyle() {
      return { position: "static" };
    },
  };
  return { doc, win };
}

test("1. manifest.json has version 1.3.3 and nativeMessaging permission", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(EXT_DIR, "manifest.json"), "utf8"));
  assert.equal(manifest.version, "1.3.3");
  assert.ok(manifest.permissions.includes("nativeMessaging"));
});

test("2. START_SERVER sends { action: 'start_server' } to com.mangalivetranslator.host and stores serverStartingAt", async () => {
  const sb = loadBackgroundSandbox({
    sendNativeImpl: async () => ({
      ok: true,
      started: true,
      already_running: false,
      pid: 4321,
    }),
  });

  const { resp } = await sb.dispatchMessage({ type: "START_SERVER" });
  assert.equal(resp.ok, true);
  assert.equal(resp.started, true);
  assert.equal(resp.already_running, false);
  assert.equal(resp.pid, 4321);
  assert.equal(sb.nativeCalls.length, 1);
  assert.equal(sb.nativeCalls[0].hostName, "com.mangalivetranslator.host");
  assert.deepEqual(JSON.parse(JSON.stringify(sb.nativeCalls[0].payload)), {
    action: "start_server",
  });
  assert.ok(Number(sb.storage._data.serverStartingAt) > 0);
});

test("3. Concurrent TRANSLATE_IMAGE failures while offline trigger only ONE native start_server call", async () => {
  let nativeStartedCount = 0;
  const sb = loadBackgroundSandbox({
    fetchImpl: async (url) => {
      if (url.endsWith("/translate")) {
        const err = new TypeError("Failed to fetch");
        throw err;
      }
      return createJsonResponse({}, 404);
    },
    sendNativeImpl: async () => {
      nativeStartedCount += 1;
      await new Promise((r) => setTimeout(r, 25));
      return { ok: true, started: true, already_running: false, pid: 9999 };
    },
  });

  const [r1, r2, r3, r4] = await Promise.all([
    sb.dispatchMessage({ type: "TRANSLATE_IMAGE", imageBase64: "AAAA" }),
    sb.dispatchMessage({ type: "TRANSLATE_IMAGE", imageBase64: "BBBB" }),
    sb.dispatchMessage({ type: "TRANSLATE_IMAGE", imageBase64: "CCCC" }),
    sb.dispatchMessage({ type: "TRANSLATE_IMAGE", imageBase64: "DDDD" }),
  ]);

  assert.equal(nativeStartedCount, 1);
  for (const { resp } of [r1, r2, r3, r4]) {
    assert.equal(resp.ok, false);
    assert.equal(resp.starting, true);
  }

  // Subsequent auto-start within 12s cooldown must also be throttled without spawning again.
  const r5 = await sb.dispatchMessage({ type: "TRANSLATE_IMAGE", imageBase64: "EEEE" });
  assert.equal(r5.resp.ok, false);
  assert.equal(r5.resp.starting, true);
  assert.equal(nativeStartedCount, 1);
});

test("4. GET_HEALTH reports starting: true while booting and clears serverStartingAt once ready: true", async () => {
  let nowMs = 100000;
  const storage = createMockStorage({ serverStartingAt: nowMs - 5000 });

  const offlineRes = await background.handleMessage(
    { type: "GET_HEALTH" },
    {},
    {
      storage,
      now: () => nowMs,
      fetch: async () => {
        throw new TypeError("Failed to fetch");
      },
    }
  );
  assert.equal(offlineRes.ok, false);
  assert.equal(offlineRes.offline, true);
  assert.equal(offlineRes.starting, true);
  assert.equal(storage._data.serverStartingAt, nowMs - 5000);

  // Loading (ready: false) keeps starting state until ready: true
  const loadingRes = await background.handleMessage(
    { type: "GET_HEALTH" },
    {},
    {
      storage,
      now: () => nowMs,
      fetch: async () => createJsonResponse({ ready: false, loading_stage: "starting_llm" }),
    }
  );
  assert.equal(loadingRes.ok, true);
  assert.equal(loadingRes.data.ready, false);
  assert.ok(storage._data.serverStartingAt);

  // Once ready: true, serverStartingAt is cleared
  const readyRes = await background.handleMessage(
    { type: "GET_HEALTH" },
    {},
    {
      storage,
      now: () => nowMs,
      fetch: async () => createJsonResponse({ ready: true, loading_stage: "ready" }),
    }
  );
  assert.equal(readyRes.ok, true);
  assert.equal(readyRes.data.ready, true);
  assert.equal(storage._data.serverStartingAt, undefined);

  // Subsequent offline GET_HEALTH without serverStartingAt reports starting: false
  const postStopRes = await background.handleMessage(
    { type: "GET_HEALTH" },
    {},
    {
      storage,
      now: () => nowMs,
      fetch: async () => {
        throw new TypeError("Failed to fetch");
      },
    }
  );
  assert.equal(postStopRes.ok, false);
  assert.equal(postStopRes.starting, false);
});

test("5. SHUTDOWN_SERVER falls back to native { action: 'stop_server' } when /shutdown is unreachable", async () => {
  const sb = loadBackgroundSandbox({
    storageInitial: { serverStartingAt: Date.now() },
    fetchImpl: async () => {
      throw new TypeError("Failed to fetch");
    },
    sendNativeImpl: async (_host, payload) => {
      assert.equal(payload?.action, "stop_server");
      return { ok: true, stopped: true };
    },
  });

  const { resp } = await sb.dispatchMessage({ type: "SHUTDOWN_SERVER" });
  assert.equal(resp.ok, true);
  assert.equal(resp.via, "native");
  assert.equal(sb.nativeCalls.length, 1);
  assert.equal(sb.nativeCalls[0].payload?.action, "stop_server");
  assert.equal(sb.storage._data.serverStartingAt, undefined);
});

test("6. Native host missing error returns { ok: false, hostMissing: true, error }", async () => {
  // Callback + runtime.lastError style
  const sb = loadBackgroundSandbox({
    sendNativeImpl: (_host, _payload, callback, runtime) => {
      runtime.lastError = { message: "Specified native messaging host not found." };
      callback(undefined);
      runtime.lastError = null;
    },
  });

  const { resp } = await sb.dispatchMessage({ type: "START_SERVER" });
  assert.equal(resp.ok, false);
  assert.equal(resp.hostMissing, true);
  assert.match(resp.error, /not installed|not found/i);
});

test("7. content.js transitions to 'Starting Engine...', polls GET_HEALTH, and auto-translates once ready", async () => {
  const { doc, win } = createMinimalDom();
  const img = doc.createElement("img");
  img.src = "https://cdn.example.org/page1.jpg";
  doc.body.appendChild(img);

  let serverReady = false;
  let healthPolls = 0;
  let translateCalls = 0;

  const controller = content.createPageController({
    document: doc,
    window: win,
    hostname: "mangadex.org",
    healthPollIntervalMs: 10,
    healthPollMaxTries: 10,
    storage: {
      async get() {
        return { enabledDomains: { "mangadex.org": true }, langMode: "auto", glossary: {} };
      },
    },
    sendMessage: async (msg) => {
      if (msg.type === "TRANSLATE_IMAGE") {
        translateCalls += 1;
        if (!serverReady) {
          return { ok: false, offline: true, starting: true, error: "Starting..." };
        }
        return { ok: true, data: { lang: "ja", width: 800, height: 1200, bubbles: [] } };
      }
      if (msg.type === "GET_HEALTH") {
        healthPolls += 1;
        if (healthPolls >= 2) {
          serverReady = true;
          return { ok: true, data: { ready: true, loading_stage: "ready" } };
        }
        return { ok: true, data: { ready: false, loading_stage: "starting_llm" } };
      }
      return { ok: true };
    },
    overlayApi: overlay,
    MutationObserver: null,
    IntersectionObserver: null,
  });

  await controller.init();
  controller.setVisibility(img, true);
  await controller.refreshVisibleAndLookahead();

  const pill = doc.body.children.find((c) => c.className === "mlt-pill");
  const statusText = pill.children[0].children[0].children[1];
  assert.equal(statusText.textContent, "Starting Engine...");

  // Wait for bounded health poll to observe ready: true and re-trigger translation
  await new Promise((r) => setTimeout(r, 80));

  assert.ok(healthPolls >= 2);
  assert.equal(translateCalls, 2);
  assert.equal(statusText.textContent, "Translated 1/1");
});

test("8. content.js notifyEngineReady recovers from start timeout and auto-translates", async () => {
  const { doc, win } = createMinimalDom();
  const img = doc.createElement("img");
  img.src = "https://cdn.example.org/page1.jpg";
  doc.body.appendChild(img);

  let serverReady = false;
  let translateCalls = 0;

  const controller = content.createPageController({
    document: doc,
    window: win,
    hostname: "mangadex.org",
    healthPollIntervalMs: 5,
    healthPollMaxTries: 2,
    storage: {
      async get() {
        return { enabledDomains: { "mangadex.org": true }, langMode: "auto", glossary: {} };
      },
    },
    sendMessage: async (msg) => {
      if (msg.type === "TRANSLATE_IMAGE") {
        translateCalls += 1;
        if (!serverReady) {
          return { ok: false, offline: true, starting: true, error: "Starting..." };
        }
        return { ok: true, data: { lang: "ja", width: 800, height: 1200, bubbles: [] } };
      }
      if (msg.type === "GET_HEALTH") {
        return { ok: true, data: { ready: false, loading_stage: "starting_llm" } };
      }
      return { ok: true };
    },
    overlayApi: overlay,
    MutationObserver: null,
    IntersectionObserver: null,
  });

  await controller.init();
  controller.setVisibility(img, true);
  await controller.refreshVisibleAndLookahead();
  await new Promise((r) => setTimeout(r, 40));

  const pill = doc.body.children.find((c) => c.className === "mlt-pill");
  const statusText = pill.children[0].children[0].children[1];
  assert.equal(statusText.textContent, "Start timed out — Click to retry");

  serverReady = true;
  controller.notifyEngineReady();
  await new Promise((r) => setTimeout(r, 30));

  assert.equal(translateCalls, 2);
  assert.equal(statusText.textContent, "Translated 1/1");
});
