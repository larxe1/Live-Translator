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

function loadBackgroundInSandbox({
  storageInitial = {},
  fetchImpl,
  alarmCreateShouldFailHalf = false,
} = {}) {
  const storage = createMockStorage(storageInitial);
  const listeners = {
    onMessage: [],
    onInstalled: [],
    onStartup: [],
    onAlarm: [],
  };
  const createdAlarms = [];
  let reloadCalls = 0;

  const chromeMock = {
    storage: {
      local: storage,
    },
    alarms: {
      async create(name, info) {
        if (alarmCreateShouldFailHalf && info?.periodInMinutes < 1) {
          throw new Error("periodInMinutes must be >= 1");
        }
        createdAlarms.push({ name, info });
      },
      onAlarm: {
        addListener(fn) {
          listeners.onAlarm.push(fn);
        },
      },
    },
    runtime: {
      id: "test-extension-id",
      getManifest() {
        return JSON.parse(fs.readFileSync(path.join(EXT_DIR, "manifest.json"), "utf8"));
      },
      reload() {
        reloadCalls += 1;
      },
      onMessage: {
        addListener(fn) {
          listeners.onMessage.push(fn);
        },
      },
      onInstalled: {
        addListener(fn) {
          listeners.onInstalled.push(fn);
        },
      },
      onStartup: {
        addListener(fn) {
          listeners.onStartup.push(fn);
        },
      },
    },
    contextMenus: {
      removeAll(cb) {
        if (typeof cb === "function") cb();
      },
      create() {},
      onClicked: {
        addListener() {},
      },
    },
    tabs: {
      async sendMessage() {},
    },
    declarativeNetRequest: {
      async updateSessionRules() {},
    },
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
    listeners,
    createdAlarms,
    getReloadCalls: () => reloadCalls,
    dispatchMessage(message, sender = {}) {
      return new Promise((resolve) => {
        let settled = false;
        const handler = listeners.onMessage[0];
        const keepOpen = handler(message, sender, (resp) => {
          if (!settled) {
            settled = true;
            resolve({ keepOpen, resp });
          }
        });
        assert.equal(keepOpen, true, "onMessage listener must return true");
      });
    },
  };
}

function createMinimalDom(href = "https://mangadex.org/chapter/99") {
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
        const targetTag = selector.toUpperCase();
        for (const child of el.children) {
          if (
            selector.split(",").some((s) => s.trim().toUpperCase() === child.tagName) ||
            child.tagName === targetTag
          ) {
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

  let reloaded = false;
  const u = new URL(href);
  const win = {
    location: {
      href,
      hostname: u.hostname,
      pathname: u.pathname,
      reload() {
        reloaded = true;
      },
    },
    addEventListener() {},
    removeEventListener() {},
    getComputedStyle() {
      return { position: "static" };
    },
  };

  return { doc, win, wasReloaded: () => reloaded };
}

test("1. manifest.json is valid MV3 v1.3.2 with alarms + nativeMessaging permissions and all referenced files exist", () => {
  const manifestPath = path.join(EXT_DIR, "manifest.json");
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));

  assert.equal(manifest.manifest_version, 3);
  assert.equal(manifest.version, "1.3.2");
  assert.ok(Array.isArray(manifest.permissions));
  assert.ok(manifest.permissions.includes("alarms"));
  assert.ok(manifest.permissions.includes("nativeMessaging"));
  assert.ok(manifest.permissions.includes("storage"));
  assert.ok(manifest.permissions.includes("activeTab"));
  assert.ok(manifest.permissions.includes("contextMenus"));
  assert.ok(manifest.permissions.includes("declarativeNetRequest"));

  const referencedFiles = [
    manifest.background?.service_worker,
    manifest.action?.default_popup,
    ...Object.values(manifest.action?.default_icon || {}),
    ...Object.values(manifest.icons || {}),
  ];

  for (const cs of manifest.content_scripts || []) {
    referencedFiles.push(...(cs.js || []), ...(cs.css || []));
  }
  for (const war of manifest.web_accessible_resources || []) {
    referencedFiles.push(...(war.resources || []));
  }

  for (const rel of referencedFiles) {
    assert.ok(rel, "Referenced file path must be non-empty");
    const full = path.join(EXT_DIR, rel);
    assert.equal(fs.existsSync(full), true, `Missing referenced file: ${rel}`);
  }

  const popupHtml = fs.readFileSync(path.join(EXT_DIR, "popup.html"), "utf8");
  const inlineScriptMatches = popupHtml.match(/<script(?![^>]*\bsrc=)[^>]*>/gi);
  assert.equal(inlineScriptMatches, null, "popup.html must not contain inline scripts");
  assert.equal(/\son\w+\s*=/i.test(popupHtml), false, "popup.html must not contain inline event handlers");
});

test("2. first health response sets baseline knownExtensionBuildId without reloading", async () => {
  const storage = createMockStorage();
  let reloads = 0;
  const runtime = {
    reload() {
      reloads += 1;
    },
  };

  const res = await background.handleMessage(
    { type: "GET_HEALTH" },
    {},
    {
      storage,
      runtime,
      reloadDelayMs: 0,
      fetch: async () =>
        createJsonResponse({
          ready: true,
          extension_version: "1.1.0",
          extension_build_id: "build-aaa111",
        }),
    }
  );

  assert.equal(res.ok, true);
  assert.equal(storage._data.knownExtensionBuildId, "build-aaa111");
  assert.equal(storage._data.lastAutoReloadBuildId, undefined);
  assert.equal(reloads, 0);
});

test("3. changed extension_build_id triggers reload, updates storage, and never loops on same build", async () => {
  const storage = createMockStorage({
    autoUpdateEnabled: true,
    knownExtensionBuildId: "build-aaa111",
  });
  let reloads = 0;
  const runtime = {
    reload() {
      reloads += 1;
    },
  };

  const reloaded = await background.maybeAutoReloadForBuild(
    "build-bbb222",
    {},
    {
      storage,
      runtime,
      reloadDelayMs: 0,
      now: () => 50000,
    }
  );
  await new Promise((r) => setTimeout(r, 10));

  assert.equal(reloaded, true);
  assert.equal(reloads, 1);
  assert.equal(storage._data.knownExtensionBuildId, "build-bbb222");
  assert.equal(storage._data.lastAutoReloadBuildId, "build-bbb222");
  assert.equal(storage._data.lastAutoReloadAt, 50000);

  // Same build must NEVER reload again.
  const second = await background.maybeAutoReloadForBuild(
    "build-bbb222",
    {},
    {
      storage,
      runtime,
      reloadDelayMs: 0,
      now: () => 100000,
    }
  );
  assert.equal(second, false);
  assert.equal(reloads, 1);

  // Even if knownExtensionBuildId somehow reverted, lastAutoReloadBuildId prevents reload loop.
  storage._data.knownExtensionBuildId = "build-old";
  const loopAttempt = await background.maybeAutoReloadForBuild(
    "build-bbb222",
    {},
    {
      storage,
      runtime,
      reloadDelayMs: 0,
      now: () => 100000,
    }
  );
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(loopAttempt, false);
  assert.equal(reloads, 1);
  assert.equal(storage._data.knownExtensionBuildId, "build-bbb222");
});

test("4. cooldown (<15s) and autoUpdateEnabled: false suppress automatic reload", async () => {
  const storage = createMockStorage({
    autoUpdateEnabled: true,
    knownExtensionBuildId: "build-1",
    lastAutoReloadBuildId: "build-1",
    lastAutoReloadAt: 10000,
  });
  let reloads = 0;
  const runtime = {
    reload() {
      reloads += 1;
    },
  };

  // Within 15s cooldown -> blocked
  const withinCooldown = await background.maybeAutoReloadForBuild(
    "build-2",
    {},
    {
      storage,
      runtime,
      reloadDelayMs: 0,
      now: () => 20000,
    }
  );
  assert.equal(withinCooldown, false);
  assert.equal(reloads, 0);

  // Disabled autoUpdateEnabled -> suppressed even after cooldown
  storage._data.autoUpdateEnabled = false;
  const disabledAttempt = await background.maybeAutoReloadForBuild(
    "build-2",
    {},
    {
      storage,
      runtime,
      reloadDelayMs: 0,
      now: () => 30000,
    }
  );
  assert.equal(disabledAttempt, false);
  assert.equal(reloads, 0);

  // Re-enabled after cooldown -> reloads
  storage._data.autoUpdateEnabled = true;
  const allowedAttempt = await background.maybeAutoReloadForBuild(
    "build-2",
    {},
    {
      storage,
      runtime,
      reloadDelayMs: 0,
      now: () => 30000,
    }
  );
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(allowedAttempt, true);
  assert.equal(reloads, 1);
});

test("5. active translation guard defers reload while translations are in flight", async () => {
  const storage = createMockStorage({
    autoUpdateEnabled: true,
    knownExtensionBuildId: "build-1",
  });
  let reloads = 0;
  const runtime = {
    reload() {
      reloads += 1;
    },
  };

  const deferred = await background.maybeAutoReloadForBuild(
    "build-2",
    {},
    {
      storage,
      runtime,
      reloadDelayMs: 0,
      now: () => 50000,
      getInflightCount: () => 2,
    }
  );
  assert.equal(deferred, false);
  assert.equal(reloads, 0);
  assert.equal(storage._data.knownExtensionBuildId, "build-1");

  // Once active translations drop to 0, reload proceeds.
  const ready = await background.maybeAutoReloadForBuild(
    "build-2",
    {},
    {
      storage,
      runtime,
      reloadDelayMs: 0,
      now: () => 50000,
      getInflightCount: () => 0,
    }
  );
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(ready, true);
  assert.equal(reloads, 1);
});

test("6. sandboxed MV3 background.js schedules alarm with fallback and auto-applies remote update on alarm", async () => {
  const fetchCalls = [];
  const sb = loadBackgroundInSandbox({
    storageInitial: {
      autoUpdateEnabled: true,
      knownExtensionBuildId: "build-initial",
    },
    alarmCreateShouldFailHalf: true,
    fetchImpl: async (url, opts = {}) => {
      fetchCalls.push({ url, method: opts.method || "GET" });
      if (url.endsWith("/health")) {
        return createJsonResponse({
          ready: true,
          extension_version: "1.1.0",
          extension_build_id: "build-initial",
          has_remote: true,
          update_available: true,
        });
      }
      if (url.endsWith("/update/apply")) {
        return createJsonResponse({
          ok: true,
          updated: true,
          message: "Updated",
          extension_changed: true,
          server_changed: false,
          restart_scheduled: false,
          status: {
            extension_version: "1.1.0",
            extension_build_id: "build-remote-new",
            has_remote: true,
            update_available: false,
          },
        });
      }
      return createJsonResponse({}, 404);
    },
  });

  // Boot should have created the alarm with fallback periodInMinutes: 1.
  await new Promise((r) => setTimeout(r, 10));
  assert.ok(
    sb.createdAlarms.some(
      (a) => a.name === "mlt-update-check" && a.info?.periodInMinutes === 1
    )
  );

  // Fire alarm
  assert.equal(sb.listeners.onAlarm.length, 1);
  sb.listeners.onAlarm[0]({ name: "mlt-update-check" });

  // Wait for async alarm work + 150ms reload timer
  await new Promise((r) => setTimeout(r, 220));

  assert.equal(fetchCalls.some((c) => c.url.endsWith("/health") && c.method === "GET"), true);
  assert.equal(
    fetchCalls.some((c) => c.url.endsWith("/update/apply") && c.method === "POST"),
    true
  );
  assert.equal(sb.getReloadCalls(), 1);
  assert.equal(sb.storage._data.knownExtensionBuildId, "build-remote-new");
});

test("7. GET_UPDATE_STATUS, CHECK_FOR_UPDATES, APPLY_UPDATE, SET_AUTO_UPDATE match contract", async () => {
  const sb = loadBackgroundInSandbox({
    storageInitial: {
      knownExtensionBuildId: "build-v1",
      autoUpdateEnabled: true,
    },
    fetchImpl: async (url, opts = {}) => {
      if (url.endsWith("/update/status")) {
        return createJsonResponse({
          extension_version: "1.1.0",
          extension_build_id: "build-v1",
          has_remote: true,
          remote_url: "https://github.com/example/repo.git",
          branch: "master",
          current_commit: "abc1234",
          remote_commit: "def5678",
          commits_behind: 1,
          update_available: true,
          auto_update_enabled: true,
          working_tree_clean: true,
          last_checked: 1700000000,
          last_error: null,
          updating: false,
          restart_required: false,
        });
      }
      if (url.endsWith("/update/check")) {
        const body = JSON.parse(opts.body || "{}");
        return createJsonResponse({
          extension_version: "1.1.0",
          extension_build_id: "build-v1",
          has_remote: true,
          commits_behind: 1,
          update_available: true,
          auto_apply_seen: body.auto_apply,
        });
      }
      if (url.endsWith("/update/apply")) {
        return createJsonResponse({
          ok: true,
          updated: true,
          message: "Applied update",
          extension_changed: true,
          server_changed: false,
          restart_scheduled: false,
          status: {
            extension_version: "1.1.0",
            extension_build_id: "build-v2",
            has_remote: true,
            update_available: false,
          },
        });
      }
      return createJsonResponse({}, 404);
    },
  });

  const statusRes = await sb.dispatchMessage({ type: "GET_UPDATE_STATUS" });
  assert.equal(statusRes.resp.ok, true);
  assert.equal(statusRes.resp.localVersion, "1.3.2");
  assert.equal(statusRes.resp.autoUpdateEnabled, true);
  assert.equal(statusRes.resp.data.update_available, true);

  const toggleOffRes = await sb.dispatchMessage({ type: "SET_AUTO_UPDATE", enabled: false });
  assert.equal(toggleOffRes.resp.ok, true);
  assert.equal(toggleOffRes.resp.autoUpdateEnabled, false);
  assert.equal(sb.storage._data.autoUpdateEnabled, false);

  const checkRes = await sb.dispatchMessage({ type: "CHECK_FOR_UPDATES" });
  assert.equal(checkRes.resp.ok, true);
  assert.equal(checkRes.resp.reloaded, false);
  assert.equal(checkRes.resp.data.auto_apply_seen, false);

  const applyRes = await sb.dispatchMessage({ type: "APPLY_UPDATE" });
  assert.equal(applyRes.resp.ok, true);
  assert.equal(applyRes.resp.reloaded, true);
  assert.equal(applyRes.resp.data.updated, true);

  await new Promise((r) => setTimeout(r, 200));
  assert.equal(sb.getReloadCalls(), 1);
  assert.equal(sb.storage._data.knownExtensionBuildId, "build-v2");
});

test("8. content.js handles Extension context invalidated gracefully and exposes clickable refresh pill", async () => {
  assert.equal(
    content.isContextInvalidatedError(new Error("Extension context invalidated.")),
    true
  );
  assert.equal(content.isContextInvalidatedError(new Error("Network error")), false);

  const { doc, win, wasReloaded } = createMinimalDom("https://mangadex.org/chapter/42");
  const img = doc.createElement("img");
  img.src = "https://cdn.example.org/p1.jpg";
  doc.body.appendChild(img);

  let sendCount = 0;
  const controller = content.createPageController({
    document: doc,
    window: win,
    hostname: "mangadex.org",
    storage: {
      async get() {
        return { enabledDomains: { "mangadex.org": true }, langMode: "auto", glossary: {} };
      },
    },
    sendMessage: async () => {
      sendCount += 1;
      throw new Error("Extension context invalidated.");
    },
    overlayApi: overlay,
    MutationObserver: null,
    IntersectionObserver: null,
  });

  await controller.init();
  controller.setVisibility(img, true);
  await controller.refreshVisibleAndLookahead();

  assert.equal(controller.isContextInvalidated(), true);
  assert.equal(sendCount, 1);

  // Subsequent calls must NOT retry sendMessage against the dead service worker.
  await controller.refreshVisibleAndLookahead();
  assert.equal(sendCount, 1);

  const pill = doc.body.children.find((c) => c.className === "mlt-pill");
  assert.ok(pill, "Floating pill should be mounted");
  const header = pill.children[0];
  const statusWrap = header.children[0];
  const statusSpan = statusWrap.children[1];
  assert.equal(statusSpan.textContent, "Updated — Click to refresh");

  statusWrap.dispatchEvent({ type: "click" });
  assert.equal(wasReloaded(), true);
});

test("9. TRANSLATE_IMAGE runs throttled (10s) post-translate health check and reloads on new build", async () => {
  const storage = createMockStorage({
    autoUpdateEnabled: true,
    knownExtensionBuildId: "build-old",
    lastPostTranslateCheckAt: 0,
  });
  let reloads = 0;
  let healthChecks = 0;
  let nowMs = 20000;

  const deps = {
    storage,
    runtime: {
      reload() {
        reloads += 1;
      },
    },
    reloadDelayMs: 0,
    now: () => nowMs,
    fetch: async (url) => {
      if (url.endsWith("/translate")) {
        return createJsonResponse({ lang: "ja", width: 100, height: 100, bubbles: [] });
      }
      if (url.endsWith("/health")) {
        healthChecks += 1;
        return createJsonResponse({
          ready: true,
          extension_version: "1.1.0",
          extension_build_id: "build-old",
        });
      }
      return createJsonResponse({}, 404);
    },
  };

  // First translation triggers /health check
  const t1 = await background.handleMessage(
    { type: "TRANSLATE_IMAGE", imageBase64: "AAAA" },
    {},
    deps
  );
  assert.equal(t1.ok, true);
  assert.equal(healthChecks, 1);

  // Second translation within 10s is throttled (no /health call)
  nowMs = 25000;
  const t2 = await background.handleMessage(
    { type: "TRANSLATE_IMAGE", imageBase64: "AAAA" },
    {},
    deps
  );
  assert.equal(t2.ok, true);
  assert.equal(healthChecks, 1);

  // Third translation after 10s sees new build and reloads
  nowMs = 35000;
  deps.fetch = async (url) => {
    if (url.endsWith("/translate")) {
      return createJsonResponse({ lang: "ja", width: 100, height: 100, bubbles: [] });
    }
    if (url.endsWith("/health")) {
      healthChecks += 1;
      return createJsonResponse({
        ready: true,
        extension_version: "1.1.0",
        extension_build_id: "build-updated",
      });
    }
    return createJsonResponse({}, 404);
  };
  const t3 = await background.handleMessage(
    { type: "TRANSLATE_IMAGE", imageBase64: "AAAA" },
    {},
    deps
  );
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(t3.ok, true);
  assert.equal(healthChecks, 2);
  assert.equal(reloads, 1);
  assert.equal(storage._data.knownExtensionBuildId, "build-updated");
});

test("10. ENSURE_CONTENT_SCRIPT injects dead/missing tabs, skips live tabs, and GET_HEALTH ready notifies tabs", async () => {
  const storage = createMockStorage({
    serverStartingAt: Date.now() - 5000,
    knownExtensionBuildId: "b1",
  });
  const injectedScripts = [];
  const injectedCss = [];
  const sentMessages = [];
  let pingAlive = false;

  const tabsMock = {
    async query() {
      return [
        { id: 101, url: "https://rawlazy.io/chapter-5" },
        { id: 102, url: "https://example.org/page" },
      ];
    },
    async sendMessage(tabId, msg) {
      sentMessages.push({ tabId, msg });
      if (msg.type === "PING_CONTENT_SCRIPT") {
        if (pingAlive && tabId === 101) {
          return { ok: true, enabled: true };
        }
        throw new Error("Could not establish connection. Receiving end does not exist.");
      }
      return { ok: true };
    },
  };
  const scriptingMock = {
    async insertCSS(arg) {
      injectedCss.push(arg);
    },
    async executeScript(arg) {
      injectedScripts.push(arg);
    },
  };

  const deps = {
    storage,
    tabs: tabsMock,
    scripting: scriptingMock,
    fetch: async (url) => {
      if (url.endsWith("/health")) {
        return createJsonResponse({ ready: true, extension_build_id: "b1" });
      }
      return createJsonResponse({}, 404);
    },
  };

  // Tab 101 has no live content script -> injects styles.css + overlay.js + content.js
  const ensure1 = await background.handleMessage({ type: "ENSURE_CONTENT_SCRIPT", tabId: 101 }, {}, deps);
  assert.equal(ensure1.ok, true);
  assert.equal(ensure1.injected, true);
  assert.equal(injectedCss.length, 1);
  assert.deepEqual(injectedScripts[0].files, ["overlay.js", "content.js"]);

  // Once live, ENSURE_CONTENT_SCRIPT is a no-op
  pingAlive = true;
  const ensure2 = await background.handleMessage({ type: "ENSURE_CONTENT_SCRIPT", tabId: 101 }, {}, deps);
  assert.equal(ensure2.ok, true);
  assert.equal(ensure2.alreadyActive, true);
  assert.equal(injectedScripts.length, 1);

  // GET_HEALTH transitioning from starting -> ready notifies open tabs with ENGINE_READY
  const healthRes = await background.handleMessage({ type: "GET_HEALTH" }, {}, deps);
  assert.equal(healthRes.ok, true);
  await new Promise((r) => setTimeout(r, 10));
  const readyMsgs = sentMessages.filter((m) => m.msg?.type === "ENGINE_READY");
  assert.equal(readyMsgs.length, 2);
});
