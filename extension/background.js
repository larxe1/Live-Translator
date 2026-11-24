"use strict";

const SERVER_BASE_URL = "http://127.0.0.1:8765";
const UPDATE_CHECK_ALARM_NAME = "mlt-update-check";
const RELOAD_COOLDOWN_MS = 15000;
const POST_TRANSLATE_HEALTH_THROTTLE_MS = 10000;
const RELOAD_DELAY_MS = 150;
const NATIVE_HOST_NAME = "com.mangalivetranslator.host";
const AUTO_START_COOLDOWN_MS = 12000;
const SERVER_STARTING_WINDOW_MS = 90000;
const NATIVE_COMMAND_TIMEOUT_MS = 20000;
const STARTING_ENGINE_MESSAGE = "Starting local translator engine...";
const STREAM_PORT_NAME = "mlt-translate-stream";

const inflightControllers = new Map();
let nativeStartInFlight = null;
let ruleIdCounter = 10000;

function nextRuleId() {
  ruleIdCounter = (ruleIdCounter % 900000) + 10000;
  return ruleIdCounter + Math.floor(Math.random() * 100);
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  if (typeof btoa === "function") {
    const chunkSize = 0x4000;
    const chunks = [];
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const sub = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
      chunks.push(String.fromCharCode.apply(null, sub));
    }
    return btoa(chunks.join(""));
  }
  if (typeof Buffer !== "undefined") {
    return Buffer.from(bytes).toString("base64");
  }
  throw new Error("No base64 encoder available");
}

function extractBase64FromDataUrl(dataUrl) {
  const commaIdx = dataUrl.indexOf(",");
  if (commaIdx === -1) return dataUrl;
  const header = dataUrl.slice(0, commaIdx);
  const body = dataUrl.slice(commaIdx + 1);
  if (header.includes(";base64")) {
    return body;
  }
  if (typeof btoa === "function") {
    return btoa(decodeURIComponent(body));
  }
  return Buffer.from(decodeURIComponent(body), "utf8").toString("base64");
}

/**
 * Fetches a cross-origin manga CDN image as raw base64.
 * When `refererUrl` is supplied, temporarily installs a declarativeNetRequest
 * session rule so hotlink-protected CDNs receive the page Referer cleanly.
 */
async function fetchImageAsBase64(imageUrl, refererUrl, deps = {}) {
  if (!imageUrl || typeof imageUrl !== "string") {
    throw new Error("Missing imageUrl");
  }
  if (imageUrl.startsWith("data:")) {
    return extractBase64FromDataUrl(imageUrl);
  }

  const fetchImpl = deps.fetch || fetch;
  const dnr =
    deps.declarativeNetRequest !== undefined
      ? deps.declarativeNetRequest
      : typeof chrome !== "undefined"
        ? chrome.declarativeNetRequest
        : null;

  let addedRuleId = null;
  if (refererUrl && dnr && typeof dnr.updateSessionRules === "function") {
    addedRuleId = nextRuleId();
    const cleanUrl = imageUrl.split("#")[0];
    try {
      await dnr.updateSessionRules({
        addRules: [
          {
            id: addedRuleId,
            priority: 1,
            action: {
              type: "modifyHeaders",
              requestHeaders: [
                {
                  header: "Referer",
                  operation: "set",
                  value: refererUrl,
                },
              ],
            },
            condition: {
              urlFilter: cleanUrl,
              resourceTypes: ["xmlhttprequest", "other"],
            },
          },
        ],
      });
    } catch (_err) {
      addedRuleId = null;
    }
  }

  try {
    const response = await fetchImpl(imageUrl, {
      method: "GET",
      credentials: "include",
      signal: deps.signal,
    });
    if (!response.ok) {
      throw new Error(`Failed to fetch image (${response.status})`);
    }
    const buffer = await response.arrayBuffer();
    return arrayBufferToBase64(buffer);
  } finally {
    if (addedRuleId !== null && dnr && typeof dnr.updateSessionRules === "function") {
      try {
        await dnr.updateSessionRules({ removeRuleIds: [addedRuleId] });
      } catch (_err) {
        // Ignore rule cleanup errors.
      }
    }
  }
}

async function fetchJson(url, options = {}, fetchImpl = fetch) {
  const response = await fetchImpl(url, options);
  const text = await response.text();
  let parsed = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch (_err) {
      parsed = { raw: text };
    }
  }
  if (!response.ok) {
    const detail = parsed?.detail || parsed?.raw || `HTTP ${response.status}`;
    throw new Error(String(detail));
  }
  return parsed;
}

async function getServerBaseUrl(deps = {}) {
  if (deps.serverBaseUrl) {
    return String(deps.serverBaseUrl).replace(/\/+$/, "");
  }
  const storage = getStorageLocal(deps);
  if (storage && typeof storage.get === "function") {
    try {
      const stored = await storage.get(["serverUrl"]);
      const raw = String(stored?.serverUrl || "").trim();
      if (/^https?:\/\/[^/\s]+/i.test(raw)) {
        return raw.replace(/\/+$/, "");
      }
    } catch (_err) {
      // Fallback to default localhost URL.
    }
  }
  return SERVER_BASE_URL;
}

function getStorageLocal(deps = {}) {
  if (deps.storage !== undefined) return deps.storage;
  return typeof chrome !== "undefined" ? chrome.storage?.local ?? null : null;
}

function getRuntime(deps = {}) {
  if (deps.runtime !== undefined) return deps.runtime;
  return typeof chrome !== "undefined" ? chrome.runtime ?? null : null;
}

function getAlarms(deps = {}) {
  if (deps.alarms !== undefined) return deps.alarms;
  return typeof chrome !== "undefined" ? chrome.alarms ?? null : null;
}

function getTabs(deps = {}) {
  if (deps.tabs !== undefined) return deps.tabs;
  return typeof chrome !== "undefined" ? chrome.tabs ?? null : null;
}

function getScripting(deps = {}) {
  if (deps.scripting !== undefined) return deps.scripting;
  return typeof chrome !== "undefined" ? chrome.scripting ?? null : null;
}

async function ensureTabContentScript(tabId, deps = {}) {
  const tabs = getTabs(deps);
  const scripting = getScripting(deps);
  if (!tabId) return { ok: false, error: "No tabId" };

  if (tabs && typeof tabs.sendMessage === "function") {
    try {
      const pong = await tabs.sendMessage(tabId, { type: "PING_CONTENT_SCRIPT" });
      if (pong && pong.ok) {
        return { ok: true, injected: false, alreadyActive: true };
      }
    } catch (_err) {
      // Content script missing or invalidated; proceed to inject.
    }
  }

  if (!scripting || typeof scripting.executeScript !== "function") {
    return { ok: false, error: "Scripting API unavailable" };
  }

  try {
    if (typeof scripting.insertCSS === "function") {
      try {
        await scripting.insertCSS({ target: { tabId }, files: ["styles.css"] });
      } catch (_cssErr) {
        // Ignore CSS injection failures on restricted frames.
      }
    }
    await scripting.executeScript({
      target: { tabId },
      files: ["overlay.js", "content.js"],
    });
    return { ok: true, injected: true };
  } catch (err) {
    return { ok: false, error: err?.message || String(err) };
  }
}

async function reinjectContentScriptsInOpenTabs(deps = {}) {
  const tabs = getTabs(deps);
  if (!tabs || typeof tabs.query !== "function") return 0;
  let list = [];
  try {
    list = (await tabs.query({ url: ["http://*/*", "https://*/*"] })) || [];
  } catch (_err) {
    return 0;
  }
  let count = 0;
  for (const tab of list) {
    if (!tab?.id) continue;
    const res = await ensureTabContentScript(tab.id, deps);
    if (res.ok && res.injected) count += 1;
  }
  return count;
}

async function notifyOpenTabsEngineReady(deps = {}) {
  const tabs = getTabs(deps);
  if (!tabs || typeof tabs.query !== "function" || typeof tabs.sendMessage !== "function") {
    return;
  }
  try {
    const list = (await tabs.query({ url: ["http://*/*", "https://*/*"] })) || [];
    for (const tab of list) {
      if (!tab?.id) continue;
      try {
        await tabs.sendMessage(tab.id, { type: "ENGINE_READY" });
      } catch (_err) {
        // Ignore tabs without active content script.
      }
    }
  } catch (_err) {
    // Ignore tab query errors.
  }
}

function getLocalManifestVersion(deps = {}) {
  if (deps.localVersion) return String(deps.localVersion);
  const runtime = getRuntime(deps);
  try {
    if (runtime && typeof runtime.getManifest === "function") {
      const manifest = runtime.getManifest();
      if (manifest?.version) return String(manifest.version);
    }
  } catch (_err) {
    // Ignore manifest lookup failures.
  }
  return "1.3.0";
}

async function safeStorageGet(storage, keys) {
  if (!storage || typeof storage.get !== "function") return null;
  try {
    return (await storage.get(keys)) || {};
  } catch (_err) {
    return null;
  }
}

async function safeStorageSet(storage, values) {
  if (!storage || typeof storage.set !== "function") return false;
  try {
    await storage.set(values);
    return true;
  } catch (_err) {
    return false;
  }
}

async function getUpdateStorageState(deps = {}) {
  const storage = getStorageLocal(deps);
  const stored = await safeStorageGet(storage, [
    "autoUpdateEnabled",
    "knownExtensionBuildId",
    "lastAutoReloadBuildId",
    "lastAutoReloadAt",
    "lastPostTranslateCheckAt",
  ]);
  return {
    storage,
    autoUpdateEnabled: stored?.autoUpdateEnabled !== false,
    knownExtensionBuildId: String(stored?.knownExtensionBuildId || ""),
    lastAutoReloadBuildId: String(stored?.lastAutoReloadBuildId || ""),
    lastAutoReloadAt: Number(stored?.lastAutoReloadAt || 0),
    lastPostTranslateCheckAt: Number(stored?.lastPostTranslateCheckAt || 0),
  };
}

function scheduleExtensionReload(deps = {}) {
  const runtime = getRuntime(deps);
  if (!runtime || typeof runtime.reload !== "function") return false;
  const scheduleTimeout = deps.setTimeout || setTimeout;
  const delayMs = deps.reloadDelayMs ?? RELOAD_DELAY_MS;
  scheduleTimeout(() => {
    try {
      runtime.reload();
    } catch (_err) {
      // Ignore reload failure outside active extension context.
    }
  }, delayMs);
  return true;
}

function getInflightCount() {
  return inflightControllers.size;
}

/**
 * Evaluates a server-reported extension_build_id and triggers a loop-proof
 * chrome.runtime.reload() when local extension files have changed on disk.
 */
async function maybeAutoReloadForBuild(serverBuildId, options = {}, deps = {}) {
  const buildId = typeof serverBuildId === "string" ? serverBuildId.trim() : "";
  if (!buildId) return false;

  const { force = false } = options || {};
  const {
    storage,
    autoUpdateEnabled,
    knownExtensionBuildId,
    lastAutoReloadBuildId,
    lastAutoReloadAt,
  } = await getUpdateStorageState(deps);

  if (!storage) return false;

  if (!knownExtensionBuildId) {
    await safeStorageSet(storage, { knownExtensionBuildId: buildId });
    return false;
  }

  if (buildId === knownExtensionBuildId) {
    return false;
  }

  if (!force && autoUpdateEnabled === false) {
    return false;
  }

  if (buildId === lastAutoReloadBuildId) {
    await safeStorageSet(storage, { knownExtensionBuildId: buildId });
    return false;
  }

  const now = typeof deps.now === "function" ? deps.now() : Date.now();
  if (!force && now - lastAutoReloadAt < RELOAD_COOLDOWN_MS) {
    return false;
  }

  const activeCount =
    typeof deps.getInflightCount === "function"
      ? deps.getInflightCount()
      : inflightControllers.size;
  if (activeCount > 0) {
    return false;
  }

  const saved = await safeStorageSet(storage, {
    knownExtensionBuildId: buildId,
    lastAutoReloadBuildId: buildId,
    lastAutoReloadAt: now,
  });
  if (!saved) return false;

  scheduleExtensionReload(deps);
  return true;
}

async function ensureUpdateAlarm(deps = {}) {
  const alarms = getAlarms(deps);
  if (!alarms || typeof alarms.create !== "function") return false;
  try {
    await alarms.create(UPDATE_CHECK_ALARM_NAME, { periodInMinutes: 0.5 });
    return true;
  } catch (_err) {
    try {
      await alarms.create(UPDATE_CHECK_ALARM_NAME, { periodInMinutes: 1 });
      return true;
    } catch (_fallbackErr) {
      return false;
    }
  }
}

async function runScheduledUpdateCheck(deps = {}) {
  const fetchImpl = deps.fetch || fetch;
  try {
    const serverBase = await getServerBaseUrl(deps);
    const health = await fetchJson(`${serverBase}/health`, { method: "GET" }, fetchImpl);
    const reloadedFromHealth = await maybeAutoReloadForBuild(
      health?.extension_build_id,
      {},
      deps
    );
    if (reloadedFromHealth) {
      return { ok: true, reloaded: true, applied: false };
    }

    const { autoUpdateEnabled } = await getUpdateStorageState(deps);
    if (health?.has_remote && health?.update_available && autoUpdateEnabled !== false) {
      const applyRes = await fetchJson(
        `${serverBase}/update/apply`,
        { method: "POST" },
        fetchImpl
      );
      const nextBuildId = applyRes?.status?.extension_build_id || "";
      let reloaded = false;
      if (nextBuildId) {
        reloaded = await maybeAutoReloadForBuild(nextBuildId, {}, deps);
      } else if (applyRes?.extension_changed) {
        const activeCount =
          typeof deps.getInflightCount === "function"
            ? deps.getInflightCount()
            : inflightControllers.size;
        if (activeCount === 0) {
          reloaded = scheduleExtensionReload(deps);
        }
      }
      return {
        ok: true,
        reloaded,
        applied: Boolean(applyRes?.updated),
        data: applyRes,
      };
    }

    return { ok: true, reloaded: false, applied: false };
  } catch (err) {
    return { ok: false, reloaded: false, applied: false, error: err?.message || String(err) };
  }
}

async function maybeCheckHealthAfterTranslate(serverBase, fetchImpl, deps = {}) {
  try {
    const { storage, lastPostTranslateCheckAt } = await getUpdateStorageState(deps);
    const now = typeof deps.now === "function" ? deps.now() : Date.now();
    if (now - lastPostTranslateCheckAt < POST_TRANSLATE_HEALTH_THROTTLE_MS) {
      return false;
    }
    if (storage) {
      await safeStorageSet(storage, { lastPostTranslateCheckAt: now });
    }
    const health = await fetchJson(`${serverBase}/health`, { method: "GET" }, fetchImpl);
    return await maybeAutoReloadForBuild(health?.extension_build_id, {}, deps);
  } catch (_err) {
    return false;
  }
}

function getNowMs(deps = {}) {
  return typeof deps.now === "function" ? deps.now() : Date.now();
}

function isLoopbackServer(serverBase) {
  try {
    const host = new URL(serverBase).hostname.replace(/^\[|\]$/g, "").toLowerCase();
    return host === "127.0.0.1" || host === "localhost" || host === "::1";
  } catch (_err) {
    return false;
  }
}

/**
 * True when the error says the server could not be reached at all
 * (connection refused / DNS / offline), not an HTTP error status.
 */
function isServerUnreachableError(err) {
  if (!err) return false;
  if (err.name === "TypeError") return true;
  const msg = String(err.message || err);
  return /failed to fetch|fetch failed|networkerror|network error|load failed|econnrefused|err_connection/i.test(
    msg
  );
}

function isNativeHostMissingMessage(message) {
  const msg = String(message || "");
  return /native messaging host not found|host not found|forbidden|invalid native messaging host name|native messaging unavailable|nativemessaging/i.test(
    msg
  );
}

/**
 * Calls runtime.sendNativeMessage exactly once and supports both the
 * Promise form and the callback + runtime.lastError form.
 */
function invokeNativeMessaging(sendNative, runtime, payload, deps = {}) {
  const scheduleTimeout = deps.setTimeout || setTimeout;
  const cancelTimeout = deps.clearTimeout || clearTimeout;
  const timeoutMs = deps.nativeTimeoutMs ?? NATIVE_COMMAND_TIMEOUT_MS;

  return new Promise((resolve, reject) => {
    let settled = false;
    let timer = null;
    const settle = (fn, value) => {
      if (settled) return;
      settled = true;
      if (timer !== null) cancelTimeout(timer);
      fn(value);
    };

    timer = scheduleTimeout(() => {
      settle(reject, new Error("Native host did not respond in time"));
    }, timeoutMs);

    const callback = (response) => {
      const lastErr = runtime?.lastError;
      if (lastErr) {
        settle(reject, new Error(lastErr.message || String(lastErr)));
      } else {
        settle(resolve, response);
      }
    };

    let returned;
    try {
      returned = sendNative(NATIVE_HOST_NAME, payload, callback);
    } catch (err) {
      settle(reject, err);
      return;
    }

    if (returned && typeof returned.then === "function") {
      void (async () => {
        try {
          settle(resolve, await returned);
        } catch (err) {
          settle(reject, err);
        }
      })();
    }
  });
}

/**
 * Sends one command to the native launcher host.
 * Resolves to { ok: true, response } or { ok: false, error, hostMissing }.
 */
async function sendNativeCommand(payload, deps = {}) {
  const runtime = getRuntime(deps);
  const sendNative =
    typeof deps.sendNativeMessage === "function"
      ? deps.sendNativeMessage
      : runtime && typeof runtime.sendNativeMessage === "function"
        ? runtime.sendNativeMessage.bind(runtime)
        : null;

  if (!sendNative) {
    return {
      ok: false,
      hostMissing: true,
      error: "Native messaging unavailable. Reload the extension and re-run setup.ps1.",
    };
  }

  try {
    const response = await invokeNativeMessaging(sendNative, runtime, payload, deps);
    if (!response || typeof response !== "object") {
      return { ok: false, hostMissing: false, error: "Empty response from native host" };
    }
    if (response.ok === false) {
      return {
        ok: false,
        hostMissing: false,
        error: String(response.error || "Native host reported an error"),
        response,
      };
    }
    return { ok: true, response };
  } catch (err) {
    const raw = err?.message || String(err);
    const hostMissing = isNativeHostMissingMessage(raw);
    return {
      ok: false,
      hostMissing,
      error: hostMissing
        ? `Local engine launcher is not installed (${raw}). Re-run setup.ps1.`
        : raw,
    };
  }
}

async function getServerStartingAt(deps = {}) {
  const stored = await safeStorageGet(getStorageLocal(deps), ["serverStartingAt"]);
  return Number(stored?.serverStartingAt || 0);
}

async function clearServerStartingAt(deps = {}) {
  const storage = getStorageLocal(deps);
  if (!storage) return false;
  const current = await getServerStartingAt(deps);
  if (!current) return false;
  if (typeof storage.remove === "function") {
    try {
      await storage.remove("serverStartingAt");
      return true;
    } catch (_err) {
      // Fall back to overwrite below.
    }
  }
  return await safeStorageSet(storage, { serverStartingAt: 0 });
}

async function runNativeStart(auto, deps) {
  const storage = getStorageLocal(deps);
  const now = getNowMs(deps);
  const startingAt = await getServerStartingAt(deps);

  if (auto && startingAt && now - startingAt < AUTO_START_COOLDOWN_MS) {
    return { ok: true, started: false, already_running: false, throttled: true, pid: null };
  }

  const res = await sendNativeCommand({ action: "start_server" }, deps);
  if (!res.ok) {
    return { ok: false, error: res.error, hostMissing: Boolean(res.hostMissing) };
  }

  const data = res.response || {};
  const started = Boolean(data.started);
  const alreadyRunning = Boolean(data.already_running);
  const startingRecent = startingAt && now - startingAt < SERVER_STARTING_WINDOW_MS;
  if (started || !startingRecent) {
    await safeStorageSet(storage, { serverStartingAt: getNowMs(deps) });
  }
  return {
    ok: true,
    started,
    already_running: alreadyRunning,
    pid: Number.isFinite(Number(data.pid)) && data.pid !== null ? Number(data.pid) : null,
  };
}

/**
 * Single-flight launcher. Concurrent callers share one native request;
 * automatic callers are additionally throttled by AUTO_START_COOLDOWN_MS.
 */
async function startServerViaNativeHost(options = {}, deps = {}) {
  const auto = Boolean(options?.auto);
  if (nativeStartInFlight) {
    return await nativeStartInFlight;
  }
  const run = runNativeStart(auto, deps);
  nativeStartInFlight = run;
  try {
    return await run;
  } catch (err) {
    return { ok: false, error: err?.message || String(err), hostMissing: false };
  } finally {
    if (nativeStartInFlight === run) {
      nativeStartInFlight = null;
    }
  }
}

async function cancelServerTranslate(serverBase, requestId, fetchImpl = fetch) {
  if (!requestId) return;
  try {
    await fetchImpl(`${serverBase}/translate/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: requestId }),
    });
  } catch (_err) {
    // Best-effort server-side cancellation.
  }
}

async function resolveTranslateImageBase64(message, referer, signal, fetchImpl, deps) {
  let imageBase64 = message?.imageBase64 || "";
  if (!imageBase64 && message?.imageUrl) {
    imageBase64 = await fetchImageAsBase64(message.imageUrl, referer, {
      ...deps,
      fetch: fetchImpl,
      signal,
    });
  }
  return imageBase64;
}

function buildTranslatePayload(msg, imageBase64, requestId) {
  const payload = {
    image_base64: imageBase64,
    lang: msg?.lang || "auto",
    chapter_id: msg?.chapterId || "default",
    include_floating: msg?.includeFloating !== false,
    priority: msg?.priority ?? 0,
    glossary: msg?.glossary || {},
    request_id: requestId,
  };
  if (msg?.seriesId) {
    payload.series_id = msg.seriesId;
  }
  const translationMode = String(msg?.translation_mode || msg?.translationMode || "").trim();
  if (translationMode) {
    payload.translation_mode = translationMode;
  }
  return payload;
}

async function handleMessage(message, sender = {}, deps = {}) {
  const fetchImpl = deps.fetch || fetch;
  const serverBase = await getServerBaseUrl(deps);
  const type = message?.type;

  switch (type) {
    case "GET_HEALTH": {
      try {
        const data = await fetchJson(`${serverBase}/health`, { method: "GET" }, fetchImpl);
        if (data?.ready === true) {
          const wasStarting = await clearServerStartingAt(deps);
          if (wasStarting) {
            void notifyOpenTabsEngineReady(deps);
          }
        }
        await maybeAutoReloadForBuild(data?.extension_build_id, {}, deps);
        return { ok: true, data };
      } catch (err) {
        const startingAt = await getServerStartingAt(deps);
        const starting = Boolean(
          startingAt && getNowMs(deps) - startingAt < SERVER_STARTING_WINDOW_MS
        );
        return { ok: false, offline: true, starting, error: err.message };
      }
    }

    case "ENSURE_CONTENT_SCRIPT": {
      let targetTabId = message?.tabId || sender?.tab?.id || null;
      if (!targetTabId) {
        const tabs = getTabs(deps);
        if (tabs && typeof tabs.query === "function") {
          try {
            const [activeTab] = (await tabs.query({ active: true, currentWindow: true })) || [];
            targetTabId = activeTab?.id || null;
          } catch (_err) {
            targetTabId = null;
          }
        }
      }
      return await ensureTabContentScript(targetTabId, deps);
    }

    case "START_SERVER": {
      return await startServerViaNativeHost({ auto: false }, deps);
    }

    case "GET_UPDATE_STATUS": {
      const localVersion = getLocalManifestVersion(deps);
      const { autoUpdateEnabled } = await getUpdateStorageState(deps);
      try {
        const data = await fetchJson(`${serverBase}/update/status`, { method: "GET" }, fetchImpl);
        await maybeAutoReloadForBuild(data?.extension_build_id, {}, deps);
        return {
          ok: true,
          data,
          autoUpdateEnabled,
          localVersion,
        };
      } catch (err) {
        return {
          ok: true,
          data: null,
          autoUpdateEnabled,
          localVersion,
          error: err.message,
        };
      }
    }

    case "CHECK_FOR_UPDATES": {
      const { autoUpdateEnabled } = await getUpdateStorageState(deps);
      const autoApply =
        message.autoApply !== undefined ? Boolean(message.autoApply) : autoUpdateEnabled;
      try {
        const data = await fetchJson(
          `${serverBase}/update/check`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ auto_apply: autoApply }),
          },
          fetchImpl
        );
        const reloaded = await maybeAutoReloadForBuild(data?.extension_build_id, {}, deps);
        return { ok: true, data, reloaded };
      } catch (err) {
        return { ok: false, error: err.message };
      }
    }

    case "APPLY_UPDATE": {
      try {
        const data = await fetchJson(`${serverBase}/update/apply`, { method: "POST" }, fetchImpl);
        const nextBuildId = data?.status?.extension_build_id || "";
        let reloaded = false;
        if (nextBuildId) {
          reloaded = await maybeAutoReloadForBuild(nextBuildId, { force: true }, deps);
        } else if (data?.extension_changed) {
          const activeCount =
            typeof deps.getInflightCount === "function"
              ? deps.getInflightCount()
              : inflightControllers.size;
          if (activeCount === 0) {
            reloaded = scheduleExtensionReload(deps);
          }
        }
        return { ok: true, data, reloaded };
      } catch (err) {
        return { ok: false, error: err.message };
      }
    }

    case "SET_AUTO_UPDATE": {
      const enabled = message.enabled !== false;
      const storage = getStorageLocal(deps);
      await safeStorageSet(storage, { autoUpdateEnabled: enabled });
      return { ok: true, autoUpdateEnabled: enabled };
    }

    case "SAVE_SETTINGS": {
      try {
        const payload = {};
        const mode = message?.translation_mode ?? message?.translationMode;
        if (mode !== undefined && mode !== null && String(mode).trim() !== "") {
          payload.translation_mode = String(mode).trim();
        }
        const apiKey = message?.gemini_api_key ?? message?.geminiApiKey;
        if (apiKey !== undefined && apiKey !== null) {
          payload.gemini_api_key = String(apiKey);
        }
        const clearKey = message?.clear_gemini_api_key ?? message?.clearGeminiApiKey;
        if (clearKey !== undefined) {
          payload.clear_gemini_api_key = Boolean(clearKey);
        }
        const data = await fetchJson(
          `${serverBase}/settings`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          },
          fetchImpl
        );
        if (data && typeof data === "object" && !Array.isArray(data)) {
          return { ok: data.ok !== false, ...data };
        }
        return { ok: true, data };
      } catch (err) {
        return { ok: false, error: err?.message || String(err) };
      }
    }

    case "GET_GLOSSARY": {
      try {
        const chapterId = encodeURIComponent(message.chapterId || "default");
        const seriesParam = message.seriesId
          ? `&series_id=${encodeURIComponent(message.seriesId)}`
          : "";
        const data = await fetchJson(
          `${serverBase}/glossary?chapter_id=${chapterId}${seriesParam}`,
          { method: "GET" },
          fetchImpl
        );
        return { ok: true, data };
      } catch (err) {
        return { ok: false, error: err.message };
      }
    }

    case "SET_GLOSSARY": {
      try {
        const payload = {
          chapter_id: message.chapterId || "default",
          entries: message.entries || {},
        };
        if (message.seriesId) {
          payload.series_id = message.seriesId;
        }
        const data = await fetchJson(
          `${serverBase}/glossary`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          },
          fetchImpl
        );
        return { ok: true, data };
      } catch (err) {
        return { ok: false, error: err.message };
      }
    }

    case "LOOKUP_GLOSSARY": {
      try {
        const data = await fetchJson(
          `${serverBase}/glossary/lookup`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              query: message.query || "",
              source: message.source || "auto",
            }),
          },
          fetchImpl
        );
        return { ok: true, data };
      } catch (err) {
        return { ok: false, error: err.message };
      }
    }

    case "CLEAR_MEMORY": {
      try {
        const payload = { chapter_id: message.chapterId || "default" };
        if (message.seriesId) {
          payload.series_id = message.seriesId;
        }
        const data = await fetchJson(
          `${serverBase}/memory/clear`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          },
          fetchImpl
        );
        return { ok: true, data };
      } catch (err) {
        return { ok: false, error: err.message };
      }
    }

    case "SHUTDOWN_SERVER": {
      let httpError = "";
      try {
        const data = await fetchJson(`${serverBase}/shutdown`, { method: "POST" }, fetchImpl);
        await clearServerStartingAt(deps);
        return { ok: true, data };
      } catch (err) {
        httpError = err?.message || String(err);
      }
      await clearServerStartingAt(deps);
      if (!isLoopbackServer(serverBase)) {
        return { ok: false, error: httpError };
      }
      const nativeRes = await sendNativeCommand({ action: "stop_server" }, deps);
      if (nativeRes.ok) {
        return { ok: true, data: nativeRes.response, via: "native" };
      }
      return {
        ok: false,
        error: httpError || nativeRes.error,
        nativeError: nativeRes.error,
        hostMissing: Boolean(nativeRes.hostMissing),
      };
    }

    case "CANCEL_TRANSLATE": {
      const requestId = message.requestId;
      if (requestId && inflightControllers.has(requestId)) {
        inflightControllers.get(requestId).abort();
        inflightControllers.delete(requestId);
      }
      await cancelServerTranslate(serverBase, requestId, fetchImpl);
      return { ok: true };
    }

    case "TRANSLATE_IMAGE": {
      const requestId = message.requestId || `bg-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const controller = new AbortController();
      inflightControllers.set(requestId, controller);
      let completedOk = false;

      try {
        const referer = message.refererUrl || sender?.tab?.url || "";
        const imageBase64 = await resolveTranslateImageBase64(
          message,
          referer,
          controller.signal,
          fetchImpl,
          deps
        );
        if (controller.signal.aborted) {
          return { ok: false, aborted: true };
        }
        if (!imageBase64) {
          return { ok: false, error: "No image data provided" };
        }

        let data;
        try {
          data = await fetchJson(
            `${serverBase}/translate`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              signal: controller.signal,
              body: JSON.stringify(buildTranslatePayload(message, imageBase64, requestId)),
            },
            fetchImpl
          );
        } catch (serverErr) {
          if (serverErr?.name === "AbortError" || controller.signal.aborted) {
            throw serverErr;
          }
          if (isServerUnreachableError(serverErr) && isLoopbackServer(serverBase)) {
            const startRes = await startServerViaNativeHost({ auto: true }, deps);
            if (startRes.ok) {
              return {
                ok: false,
                offline: true,
                starting: true,
                error: STARTING_ENGINE_MESSAGE,
              };
            }
            return {
              ok: false,
              offline: true,
              starting: false,
              hostMissing: Boolean(startRes.hostMissing),
              error: serverErr?.message || String(serverErr),
            };
          }
          throw serverErr;
        }
        completedOk = true;
        return { ok: true, data };
      } catch (err) {
        if (err?.name === "AbortError" || controller.signal.aborted) {
          return { ok: false, aborted: true };
        }
        return { ok: false, error: err.message };
      } finally {
        inflightControllers.delete(requestId);
        if (completedOk) {
          await maybeCheckHealthAfterTranslate(serverBase, fetchImpl, deps);
        }
      }
    }

    default:
      return { ok: false, error: `Unknown message type: ${type}` };
  }
}

function handlePortConnect(port, deps = {}) {
  if (!port || port.name !== STREAM_PORT_NAME) return;

  let disconnected = false;
  let activeController = null;
  let activeRequestId = null;
  let activeServerBase = "";

  const safePost = (event) => {
    if (disconnected) return;
    try {
      port.postMessage(event);
    } catch (_err) {
      disconnected = true;
    }
  };

  if (port.onDisconnect?.addListener) {
    port.onDisconnect.addListener(() => {
      disconnected = true;
      if (activeController) {
        activeController.abort();
      }
      if (activeRequestId) {
        const reqId = activeRequestId;
        const base = activeServerBase;
        inflightControllers.delete(reqId);
        activeController = null;
        activeRequestId = null;
        const fetchImpl = deps.fetch || fetch;
        void (async () => {
          const resolvedBase = base || (await getServerBaseUrl(deps));
          await cancelServerTranslate(resolvedBase, reqId, fetchImpl);
        })();
      }
    });
  }

  if (port.onMessage?.addListener) {
    port.onMessage.addListener((msg) => {
      if (msg?.type !== "START" || disconnected) return;
      void runStreamRequest(msg);
    });
  }

  async function runStreamRequest(msg) {
    const fetchImpl = deps.fetch || fetch;
    const serverBase = await getServerBaseUrl(deps);
    const requestId =
      msg.requestId || `bg-stream-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const controller = new AbortController();
    activeController = controller;
    activeRequestId = requestId;
    activeServerBase = serverBase;
    inflightControllers.set(requestId, controller);

    let completedOk = false;
    try {
      const referer = msg.refererUrl || port?.sender?.tab?.url || "";
      const imageBase64 = await resolveTranslateImageBase64(
        msg,
        referer,
        controller.signal,
        fetchImpl,
        deps
      );
      if (disconnected || controller.signal.aborted) return;
      if (!imageBase64) {
        safePost({ type: "error", detail: "No image data provided" });
        return;
      }

      let response;
      try {
        response = await fetchImpl(`${serverBase}/translate/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify(buildTranslatePayload(msg, imageBase64, requestId)),
        });
      } catch (serverErr) {
        if (disconnected || serverErr?.name === "AbortError" || controller.signal.aborted) {
          return;
        }
        if (isServerUnreachableError(serverErr) && isLoopbackServer(serverBase)) {
          const startRes = await startServerViaNativeHost({ auto: true }, deps);
          if (startRes.ok) {
            safePost({
              type: "error",
              offline: true,
              starting: true,
              error: STARTING_ENGINE_MESSAGE,
            });
          } else {
            safePost({
              type: "error",
              offline: true,
              starting: false,
              hostMissing: Boolean(startRes.hostMissing),
              error: serverErr?.message || String(serverErr),
            });
          }
          return;
        }
        safePost({ type: "error", detail: serverErr?.message || String(serverErr) });
        return;
      }

      if (!response.ok) {
        if (response.status === 404) {
          // Graceful fallback when talking to an older server without /translate/stream.
          const fallbackData = await fetchJson(
            `${serverBase}/translate`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              signal: controller.signal,
              body: JSON.stringify(buildTranslatePayload(msg, imageBase64, requestId)),
            },
            fetchImpl
          );
          if (disconnected || controller.signal.aborted) return;
          safePost({
            type: "meta",
            lang: fallbackData.lang,
            width: fallbackData.width,
            height: fallbackData.height,
            bubbles: (fallbackData.bubbles || []).map((b) => ({
              id: b.id,
              bbox: b.bbox,
              safe_box: b.safe_box,
              bg_color: b.bg_color,
              text_color: b.text_color,
              is_dark: b.is_dark,
              original: b.original,
              region_type: b.region_type || "bubble",
            })),
          });
          for (const b of fallbackData.bubbles || []) {
            safePost({
              type: "bubble",
              id: b.id,
              translated: b.translated ?? "",
              region_type: b.region_type || "bubble",
              final: true,
            });
          }
          completedOk = true;
          safePost({ type: "done", response: fallbackData });
          return;
        }

        const errText = await response.text();
        let detail = `HTTP ${response.status}`;
        if (errText) {
          try {
            const parsed = JSON.parse(errText);
            detail = parsed?.detail || errText;
          } catch (_err) {
            detail = errText;
          }
        }
        if (response.status === 503) {
          safePost({
            type: "error",
            starting: true,
            status: 503,
            detail: String(detail),
            error: String(detail),
          });
          return;
        }
        safePost({
          type: "error",
          status: response.status,
          detail: String(detail),
          error: String(detail),
        });
        return;
      }

      const emitLine = (rawLine) => {
        const trimmed = rawLine.trim();
        if (!trimmed || disconnected) return;
        try {
          const evt = JSON.parse(trimmed);
          if (evt?.type === "done") completedOk = true;
          safePost(evt);
        } catch (_err) {
          // Ignore malformed NDJSON line.
        }
      };

      const reader = response.body?.getReader?.();
      if (!reader) {
        const text = await response.text();
        for (const line of text.split("\n")) {
          emitLine(line);
        }
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";
      while (!disconnected && !controller.signal.aborted) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          emitLine(line);
        }
      }
      buffer += decoder.decode();
      if (buffer.trim()) {
        emitLine(buffer);
      }
    } catch (err) {
      if (!disconnected && err?.name !== "AbortError" && !controller.signal.aborted) {
        safePost({ type: "error", detail: err?.message || String(err) });
      }
    } finally {
      inflightControllers.delete(requestId);
      if (activeRequestId === requestId) {
        activeController = null;
        activeRequestId = null;
      }
      if (completedOk && !disconnected) {
        await maybeCheckHealthAfterTranslate(serverBase, fetchImpl, deps);
      }
    }
  }
}

function registerExtensionListeners() {
  if (typeof chrome === "undefined" || !chrome.runtime) return;

  if (chrome.runtime.onConnect?.addListener) {
    chrome.runtime.onConnect.addListener((port) => {
      handlePortConnect(port);
    });
  }

  if (chrome.runtime.onMessage?.addListener) {
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      void (async () => {
        try {
          const res = await handleMessage(message, sender);
          sendResponse(res);
        } catch (err) {
          sendResponse({ ok: false, error: err?.message || String(err) });
        }
      })();
      return true;
    });
  }

  if (chrome.runtime.onInstalled?.addListener) {
    chrome.runtime.onInstalled.addListener(() => {
      if (chrome.contextMenus) {
        try {
          chrome.contextMenus.removeAll(() => {
            try {
              chrome.contextMenus.create({
                id: "mlt-translate-image",
                title: "Translate this manga page",
                contexts: ["image"],
              });
            } catch (_err) {
              // Ignore context menu creation errors.
            }
          });
        } catch (_err) {
          // Ignore context menu errors.
        }
      }
      void ensureUpdateAlarm();
      void runScheduledUpdateCheck();
      void reinjectContentScriptsInOpenTabs();
    });
  }

  if (chrome.runtime.onStartup?.addListener) {
    chrome.runtime.onStartup.addListener(() => {
      void ensureUpdateAlarm();
      void runScheduledUpdateCheck();
      void reinjectContentScriptsInOpenTabs();
    });
  }

  if (chrome.alarms?.onAlarm?.addListener) {
    chrome.alarms.onAlarm.addListener((alarm) => {
      if (alarm?.name === UPDATE_CHECK_ALARM_NAME) {
        void runScheduledUpdateCheck();
      }
    });
  }

  if (chrome.contextMenus?.onClicked?.addListener) {
    chrome.contextMenus.onClicked.addListener(async (info, tab) => {
      if (info.menuItemId === "mlt-translate-image" && tab?.id) {
        try {
          await chrome.tabs.sendMessage(tab.id, {
            type: "MANUAL_TRANSLATE_IMAGE",
            srcUrl: info.srcUrl || "",
          });
        } catch (_err) {
          // Tab may not have the content script active.
        }
      }
    });
  }

  void ensureUpdateAlarm();
}

registerExtensionListeners();

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    SERVER_BASE_URL,
    UPDATE_CHECK_ALARM_NAME,
    RELOAD_COOLDOWN_MS,
    POST_TRANSLATE_HEALTH_THROTTLE_MS,
    NATIVE_HOST_NAME,
    STREAM_PORT_NAME,
    AUTO_START_COOLDOWN_MS,
    SERVER_STARTING_WINDOW_MS,
    arrayBufferToBase64,
    fetchImageAsBase64,
    getInflightCount,
    getUpdateStorageState,
    maybeAutoReloadForBuild,
    ensureUpdateAlarm,
    runScheduledUpdateCheck,
    sendNativeCommand,
    startServerViaNativeHost,
    ensureTabContentScript,
    reinjectContentScriptsInOpenTabs,
    notifyOpenTabsEngineReady,
    handleMessage,
    handlePortConnect,
    registerExtensionListeners,
  };
}
