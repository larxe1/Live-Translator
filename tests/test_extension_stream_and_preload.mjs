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
      for (const k of list) delete data[k];
    },
  };
}

function createMockPortPair(name = "mlt-translate-stream") {
  const bgMessageListeners = [];
  const bgDisconnectListeners = [];
  const clientMessageListeners = [];
  const clientDisconnectListeners = [];
  let disconnected = false;

  const bgPort = {
    name,
    sender: { tab: { url: "https://mangadex.org/chapter/1" } },
    postMessage(msg) {
      if (disconnected) throw new Error("Attempting to use a disconnected port object");
      for (const fn of clientMessageListeners) fn(msg);
    },
    disconnect() {
      if (disconnected) return;
      disconnected = true;
      for (const fn of clientDisconnectListeners) fn();
    },
    onMessage: {
      addListener(fn) {
        bgMessageListeners.push(fn);
      },
    },
    onDisconnect: {
      addListener(fn) {
        bgDisconnectListeners.push(fn);
      },
    },
  };

  const clientPort = {
    name,
    postMessage(msg) {
      if (disconnected) throw new Error("Attempting to use a disconnected port object");
      for (const fn of bgMessageListeners) fn(msg);
    },
    disconnect() {
      if (disconnected) return;
      disconnected = true;
      for (const fn of bgDisconnectListeners) fn();
    },
    onMessage: {
      addListener(fn) {
        clientMessageListeners.push(fn);
      },
    },
    onDisconnect: {
      addListener(fn) {
        clientDisconnectListeners.push(fn);
      },
    },
  };

  return { bgPort, clientPort, isDisconnected: () => disconnected };
}

function loadBackgroundSandbox({ storageInitial = {}, fetchImpl, sendNativeImpl } = {}) {
  const storage = createMockStorage(storageInitial);
  const connectListeners = [];
  const nativeCalls = [];

  const chromeMock = {
    storage: { local: storage },
    alarms: { async create() {}, onAlarm: { addListener() {} } },
    runtime: {
      id: "test-extension-id",
      lastError: null,
      getManifest() {
        return JSON.parse(fs.readFileSync(path.join(EXT_DIR, "manifest.json"), "utf8"));
      },
      sendNativeMessage(hostName, payload) {
        nativeCalls.push({ hostName, payload });
        if (typeof sendNativeImpl === "function") {
          return sendNativeImpl(hostName, payload);
        }
        return Promise.resolve({ ok: true, started: true, already_running: false, pid: 1234 });
      },
      onMessage: { addListener() {} },
      onConnect: {
        addListener(fn) {
          connectListeners.push(fn);
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
    fetch: fetchImpl || (async () => ({ ok: true, status: 200, async text() { return "{}"; } })),
    setTimeout,
    clearTimeout,
    AbortController,
    TextDecoder,
    TextEncoder,
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
    connectPort(name = "mlt-translate-stream") {
      const pair = createMockPortPair(name);
      for (const fn of connectListeners) fn(pair.bgPort);
      return pair;
    },
  };
}

function createMinimalDom(href = "https://mangadex.org/chapter/10") {
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

  const winListeners = new Map();
  const u = new URL(href);
  const win = {
    location: { href, hostname: u.hostname, pathname: u.pathname, reload() {} },
    addEventListener(type, fn) {
      if (!winListeners.has(type)) winListeners.set(type, []);
      winListeners.get(type).push(fn);
    },
    removeEventListener() {},
    dispatchEvent(evt) {
      for (const fn of winListeners.get(evt.type) || []) fn(evt);
    },
    getComputedStyle() {
      return { position: "static" };
    },
  };
  return { doc, win };
}

test("1. mlt-translate-stream Port forwards NDJSON meta -> bubble -> bubble -> done incrementally", async () => {
  const encoder = new TextEncoder();
  const chunks = [
    JSON.stringify({ type: "meta", lang: "ja", width: 800, height: 1200, bubbles: [{ id: 0 }, { id: 1 }] }) + "\n",
    JSON.stringify({ type: "bubble", id: 0, translated: "First", final: true }) + "\n{\"type\":\"bub",
    "ble\",\"id\":1,\"translated\":\"Second\",\"final\":true}\n",
    JSON.stringify({ type: "done", response: { lang: "ja", width: 800, height: 1200, bubbles: [{ id: 0, translated: "First" }, { id: 1, translated: "Second" }] } }) + "\n",
  ];

  const sb = loadBackgroundSandbox({
    fetchImpl: async (url) => {
      if (url.endsWith("/translate/stream")) {
        let idx = 0;
        return {
          ok: true,
          status: 200,
          body: {
            getReader() {
              return {
                async read() {
                  if (idx >= chunks.length) return { done: true, value: undefined };
                  const value = encoder.encode(chunks[idx++]);
                  await new Promise((r) => setTimeout(r, 5));
                  return { done: false, value };
                },
              };
            },
          },
        };
      }
      if (url.endsWith("/health")) {
        return { ok: true, status: 200, async text() { return JSON.stringify({ ready: true }); } };
      }
      return { ok: false, status: 404, async text() { return "{}"; } };
    },
  });

  const { clientPort } = sb.connectPort("mlt-translate-stream");
  const received = [];
  await new Promise((resolve) => {
    clientPort.onMessage.addListener((evt) => {
      received.push(evt);
      if (evt.type === "done") resolve();
    });
    clientPort.postMessage({
      type: "START",
      imageBase64: "AAAA",
      lang: "ja",
      chapterId: "ch1",
      priority: 0,
      glossary: {},
      requestId: "req-stream-1",
    });
  });

  assert.deepEqual(
    received.map((e) => e.type),
    ["meta", "bubble", "bubble", "done"]
  );
  assert.equal(received[1].translated, "First");
  assert.equal(received[2].translated, "Second");
});

test("2. port.onDisconnect aborts in-flight fetch and posts /translate/cancel", async () => {
  let aborted = false;
  let cancelledRequestId = null;

  const sb = loadBackgroundSandbox({
    fetchImpl: async (url, opts = {}) => {
      if (url.endsWith("/translate/stream")) {
        return new Promise((_resolve, reject) => {
          opts.signal.addEventListener("abort", () => {
            aborted = true;
            const err = new Error("Aborted");
            err.name = "AbortError";
            reject(err);
          });
        });
      }
      if (url.endsWith("/translate/cancel")) {
        cancelledRequestId = JSON.parse(opts.body).request_id;
        return { ok: true, status: 200, async text() { return "{}"; } };
      }
      return { ok: false, status: 404, async text() { return "{}"; } };
    },
  });

  const { clientPort } = sb.connectPort("mlt-translate-stream");
  clientPort.postMessage({
    type: "START",
    imageBase64: "AAAA",
    requestId: "req-abort-1",
  });

  await new Promise((r) => setTimeout(r, 15));
  clientPort.disconnect();
  await new Promise((r) => setTimeout(r, 20));

  assert.equal(aborted, true);
  assert.equal(cancelledRequestId, "req-abort-1");
});

test("3. Offline stream start triggers single-flight native auto-start and posts { type: 'error', starting: true }", async () => {
  let nativeCalls = 0;
  const sb = loadBackgroundSandbox({
    fetchImpl: async (url) => {
      if (url.endsWith("/translate/stream")) {
        throw new TypeError("Failed to fetch");
      }
      return { ok: false, status: 404, async text() { return "{}"; } };
    },
    sendNativeImpl: async () => {
      nativeCalls += 1;
      await new Promise((r) => setTimeout(r, 15));
      return { ok: true, started: true, already_running: false, pid: 5555 };
    },
  });

  const p1 = sb.connectPort("mlt-translate-stream");
  const p2 = sb.connectPort("mlt-translate-stream");

  const events1 = [];
  const events2 = [];

  await Promise.all([
    new Promise((resolve) => {
      p1.clientPort.onMessage.addListener((e) => {
        events1.push(e);
        resolve();
      });
      p1.clientPort.postMessage({ type: "START", imageBase64: "AAAA", requestId: "s1" });
    }),
    new Promise((resolve) => {
      p2.clientPort.onMessage.addListener((e) => {
        events2.push(e);
        resolve();
      });
      p2.clientPort.postMessage({ type: "START", imageBase64: "BBBB", requestId: "s2" });
    }),
  ]);

  assert.equal(nativeCalls, 1);
  assert.equal(events1[0].type, "error");
  assert.equal(events1[0].starting, true);
  assert.equal(events2[0].type, "error");
  assert.equal(events2[0].starting, true);
});

test("4. Whole-chapter pre-translation queues offscreen loaded images at priority 1 behind visible priority 0, skips small/unloaded, and cancels on pagehide", async () => {
  const { doc, win } = createMinimalDom();

  // Visible page 0
  const p0 = doc.createElement("img");
  p0.src = "https://cdn.example.org/p0.jpg";
  doc.body.appendChild(p0);

  // Small icon placeholder (should be skipped)
  const icon = doc.createElement("img");
  icon.src = "https://cdn.example.org/icon.png";
  icon.width = 48;
  icon.height = 48;
  icon.naturalWidth = 48;
  icon.naturalHeight = 48;
  doc.body.appendChild(icon);

  // Unloaded image (complete=false, should be skipped until loaded)
  const unloaded = doc.createElement("img");
  unloaded.src = "https://cdn.example.org/unloaded.jpg";
  unloaded.complete = false;
  doc.body.appendChild(unloaded);

  // Offscreen loaded pages 1..5
  const offscreen = [];
  for (let i = 1; i <= 5; i += 1) {
    const img = doc.createElement("img");
    img.src = `https://cdn.example.org/p${i}.jpg`;
    doc.body.appendChild(img);
    offscreen.push(img);
  }

  const started = [];
  const disconnectedIds = [];
  const cancelledIds = [];
  let holdP5Resolve = null;

  const controller = content.createPageController({
    document: doc,
    window: win,
    hostname: "mangadex.org",
    storage: {
      async get() {
        return { enabledDomains: { "mangadex.org": true }, langMode: "ja", glossary: {} };
      },
    },
    sendMessage: async (msg) => {
      if (msg.type === "CANCEL_TRANSLATE") {
        cancelledIds.push(msg.requestId);
      }
      return { ok: true };
    },
    connectPort: () => {
      const listeners = [];
      let reqId = null;
      return {
        onMessage: { addListener(fn) { listeners.push(fn); } },
        onDisconnect: { addListener() {} },
        disconnect() {
          if (reqId) disconnectedIds.push(reqId);
        },
        postMessage(msg) {
          if (msg.type !== "START") return;
          reqId = msg.requestId;
          const file = msg.imageUrl.split("/").pop();
          started.push({ file, priority: msg.priority, requestId: reqId });
          if (file === "p5.jpg") {
            holdP5Resolve = () => {
              for (const fn of listeners) {
                fn({ type: "done", response: { lang: "ja", width: 800, height: 1200, bubbles: [] } });
              }
            };
            return;
          }
          setTimeout(() => {
            for (const fn of listeners) {
              fn({ type: "meta", lang: "ja", width: 800, height: 1200, bubbles: [] });
              fn({ type: "done", response: { lang: "ja", width: 800, height: 1200, bubbles: [] } });
            }
          }, 2);
        },
      };
    },
    overlayApi: overlay,
    MutationObserver: null,
    IntersectionObserver: null,
  });

  await controller.init();
  controller.setVisibility(p0, true);
  await controller.refreshVisibleAndLookahead();

  // Wait for background sequential preload to walk p4 -> p5 (where p5 stays in-flight)
  await new Promise((r) => setTimeout(r, 60));

  assert.equal(started[0].file, "p0.jpg");
  assert.equal(started[0].priority, 0);
  const files = started.map((s) => s.file);
  assert.ok(files.includes("p4.jpg"));
  assert.ok(files.includes("p5.jpg"));
  assert.equal(files.includes("icon.png"), false);
  assert.equal(files.includes("unloaded.jpg"), false);
  assert.ok(started.find((s) => s.file === "p4.jpg").priority === 1);
  assert.ok(started.find((s) => s.file === "p5.jpg").priority === 1);

  // Leaving page triggers pagehide -> disconnects active p5 port and sends CANCEL_TRANSLATE
  const p5ReqId = started.find((s) => s.file === "p5.jpg").requestId;
  win.dispatchEvent({ type: "pagehide" });
  assert.ok(disconnectedIds.includes(p5ReqId));
  assert.ok(cancelledIds.includes(p5ReqId));
  if (holdP5Resolve) holdP5Resolve();
});

test("5. HTTP 503 on /translate/stream emits starting:true and HTTP 404 falls back to POST /translate", async () => {
  // Part A: 503 while models are loading
  const sb503 = loadBackgroundSandbox({
    fetchImpl: async (url) => {
      if (url.endsWith("/translate/stream")) {
        return {
          ok: false,
          status: 503,
          async text() {
            return JSON.stringify({ detail: "Models are not ready yet (loading_detector)" });
          },
        };
      }
      return { ok: false, status: 404, async text() { return "{}"; } };
    },
  });
  const p503 = sb503.connectPort("mlt-translate-stream");
  const errEvt = await new Promise((resolve) => {
    p503.clientPort.onMessage.addListener((evt) => resolve(evt));
    p503.clientPort.postMessage({ type: "START", imageBase64: "AAAA", requestId: "req-503" });
  });
  assert.equal(errEvt.type, "error");
  assert.equal(errEvt.starting, true);
  assert.equal(errEvt.status, 503);

  // Part B: 404 falls back to POST /translate and emits meta -> bubble -> done
  const sb404 = loadBackgroundSandbox({
    fetchImpl: async (url) => {
      if (url.endsWith("/translate/stream")) {
        return { ok: false, status: 404, async text() { return "Not Found"; } };
      }
      if (url.endsWith("/translate")) {
        return {
          ok: true,
          status: 200,
          async text() {
            return JSON.stringify({
              lang: "ja",
              width: 800,
              height: 1200,
              bubbles: [{ id: 0, bbox: [0, 0, 100, 100], safe_box: [5, 5, 95, 95], translated: "Fallback OK", original: "テスト" }],
            });
          },
        };
      }
      if (url.endsWith("/health")) {
        return { ok: true, status: 200, async text() { return JSON.stringify({ ready: true }); } };
      }
      return { ok: false, status: 404, async text() { return "{}"; } };
    },
  });
  const p404 = sb404.connectPort("mlt-translate-stream");
  const fallbackEvents = [];
  await new Promise((resolve) => {
    p404.clientPort.onMessage.addListener((evt) => {
      fallbackEvents.push(evt);
      if (evt.type === "done") resolve();
    });
    p404.clientPort.postMessage({ type: "START", imageBase64: "AAAA", requestId: "req-404" });
  });
  assert.deepEqual(fallbackEvents.map((e) => e.type), ["meta", "bubble", "done"]);
  assert.equal(fallbackEvents[1].translated, "Fallback OK");
});

test("6. Lazy-loaded placeholder img swapping src + load event triggers translation and shows/clears on-page loading badge", async () => {
  const { doc, win } = createMinimalDom();
  win.innerWidth = 1280;
  win.innerHeight = 900;

  // Placeholder <img> starts complete=true with 1x1 dimensions (common lazy-load pattern)
  const lazyImg = doc.createElement("img");
  lazyImg.src = "https://cdn.example.org/placeholder.gif";
  lazyImg.width = 1;
  lazyImg.height = 1;
  lazyImg.naturalWidth = 1;
  lazyImg.naturalHeight = 1;
  lazyImg.complete = true;
  lazyImg.getBoundingClientRect = () => ({
    left: 100,
    top: 20,
    width: lazyImg.width,
    height: lazyImg.height,
    right: 100 + lazyImg.width,
    bottom: 20 + lazyImg.height,
  });
  const loadHandlers = [];
  lazyImg.addEventListener = (type, fn) => {
    if (type === "load") loadHandlers.push(fn);
  };
  doc.body.appendChild(lazyImg);

  let observedIoOpts = null;
  class FakeIO {
    constructor(_cb, opts) {
      observedIoOpts = opts;
    }
    observe() {}
    unobserve() {}
    disconnect() {}
  }

  let badgeSeenWhileTranslating = "";
  let badgeSeenAfterBubble = "";
  const startedUrls = [];
  let finishStream = null;

  const controller = content.createPageController({
    document: doc,
    window: win,
    hostname: "mangadex.org",
    storage: {
      async get() {
        return { enabledDomains: { "mangadex.org": true }, langMode: "ja", glossary: {} };
      },
    },
    sendMessage: async () => ({ ok: true }),
    connectPort: () => {
      const listeners = [];
      return {
        onMessage: { addListener(fn) { listeners.push(fn); } },
        onDisconnect: { addListener() {} },
        disconnect() {},
        postMessage(msg) {
          if (msg.type !== "START") return;
          startedUrls.push(msg.imageUrl);
          const findBadges = () =>
            doc.body.children.filter((c) => c.className === "mlt-page-loading");
          const badges = findBadges();
          badgeSeenWhileTranslating = badges.length > 0 ? badges[0].children[1].textContent : "";
          finishStream = () => {
            for (const fn of listeners) {
              fn({ type: "meta", lang: "ja", width: 800, height: 15000, bubbles: [{ id: 0, bbox: [0, 0, 100, 100] }] });
              fn({ type: "bubble", id: 0, translated: "Hello!", final: true });
            }
            const badgesMid = findBadges();
            badgeSeenAfterBubble = badgesMid.length > 0 ? badgesMid[0].children[1].textContent : "";
            for (const fn of listeners) {
              fn({
                type: "done",
                response: {
                  lang: "ja",
                  width: 800,
                  height: 15000,
                  bubbles: [{ id: 0, bbox: [0, 0, 100, 100], translated: "Hello!" }],
                },
              });
            }
          };
        },
      };
    },
    overlayApi: overlay,
    MutationObserver: null,
    IntersectionObserver: FakeIO,
  });

  await controller.init();
  assert.equal(observedIoOpts.threshold, 0);
  assert.equal(startedUrls.length, 0);

  // Lazy-loader swaps src onto placeholder -> complete becomes false while downloading
  lazyImg.src = "https://cdn.example.org/long-strip-page1.jpg";
  lazyImg.complete = false;
  controller.handleSourceMutation(lazyImg);
  assert.equal(startedUrls.length, 0);

  // Real tall webtoon image finishes loading -> fires load event
  lazyImg.complete = true;
  lazyImg.width = 800;
  lazyImg.height = 15000;
  lazyImg.naturalWidth = 800;
  lazyImg.naturalHeight = 15000;
  for (const fn of loadHandlers) fn();

  assert.equal(startedUrls.length, 1);
  assert.equal(startedUrls[0], "https://cdn.example.org/long-strip-page1.jpg");
  assert.equal(badgeSeenWhileTranslating, "Translating...");

  finishStream();
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(badgeSeenAfterBubble, "Translating 1/1...");
  assert.equal(doc.body.children.filter((c) => c.className === "mlt-page-loading").length, 0);
});

test("7. Failed image in preload records failure once and advances to next page instead of spinning", async () => {
  const { doc, win } = createMinimalDom();
  const badImg = doc.createElement("img");
  badImg.src = "https://cdn.example.org/bad.jpg";
  doc.body.appendChild(badImg);

  const goodImg = doc.createElement("img");
  goodImg.src = "https://cdn.example.org/good.jpg";
  doc.body.appendChild(goodImg);

  const calls = [];
  const controller = content.createPageController({
    document: doc,
    window: win,
    hostname: "mangadex.org",
    storage: {
      async get() {
        return { enabledDomains: { "mangadex.org": true }, langMode: "ja", glossary: {} };
      },
    },
    sendMessage: async () => ({ ok: true }),
    connectPort: () => {
      const listeners = [];
      return {
        onMessage: { addListener(fn) { listeners.push(fn); } },
        onDisconnect: { addListener() {} },
        disconnect() {},
        postMessage(msg) {
          if (msg.type !== "START") return;
          const file = msg.imageUrl.split("/").pop();
          calls.push(file);
          setTimeout(() => {
            if (file === "bad.jpg") {
              for (const fn of listeners) fn({ type: "error", detail: "HTTP 413 Image too large" });
            } else {
              for (const fn of listeners) {
                fn({ type: "meta", lang: "ja", width: 800, height: 1200, bubbles: [] });
                fn({ type: "done", response: { lang: "ja", width: 800, height: 1200, bubbles: [] } });
              }
            }
          }, 2);
        },
      };
    },
    overlayApi: overlay,
    MutationObserver: null,
    IntersectionObserver: null,
  });

  await controller.init();
  await new Promise((r) => setTimeout(r, 50));

  assert.equal(calls.filter((c) => c === "bad.jpg").length, 1);
  assert.equal(calls.filter((c) => c === "good.jpg").length, 1);
});
