"use strict";

const SERVER_BASE_URL = "http://127.0.0.1:8765";
const UPDATE_CHECK_ALARM_NAME = "mlt-update-check";
const RELOAD_COOLDOWN_MS = 15000;
const POST_TRANSLATE_HEALTH_THROTTLE_MS = 10000;
const RELOAD_DELAY_MS = 150;

const inflightControllers = new Map();
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
  return "1.1.0";
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

async function handleMessage(message, sender = {}, deps = {}) {
  const fetchImpl = deps.fetch || fetch;
  const serverBase = await getServerBaseUrl(deps);
  const type = message?.type;

  switch (type) {
    case "GET_HEALTH": {
      try {
        const data = await fetchJson(`${serverBase}/health`, { method: "GET" }, fetchImpl);
        await maybeAutoReloadForBuild(data?.extension_build_id, {}, deps);
        return { ok: true, data };
      } catch (err) {
        return { ok: false, offline: true, error: err.message };
      }
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

    case "GET_GLOSSARY": {
      try {
        const chapterId = encodeURIComponent(message.chapterId || "default");
        const data = await fetchJson(
          `${serverBase}/glossary?chapter_id=${chapterId}`,
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
        const data = await fetchJson(
          `${serverBase}/glossary`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              chapter_id: message.chapterId || "default",
              entries: message.entries || {},
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
        const data = await fetchJson(
          `${serverBase}/memory/clear`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ chapter_id: message.chapterId || "default" }),
          },
          fetchImpl
        );
        return { ok: true, data };
      } catch (err) {
        return { ok: false, error: err.message };
      }
    }

    case "SHUTDOWN_SERVER": {
      try {
        const data = await fetchJson(`${serverBase}/shutdown`, { method: "POST" }, fetchImpl);
        return { ok: true, data };
      } catch (err) {
        return { ok: false, error: err.message };
      }
    }

    case "CANCEL_TRANSLATE": {
      const requestId = message.requestId;
      if (requestId && inflightControllers.has(requestId)) {
        inflightControllers.get(requestId).abort();
        inflightControllers.delete(requestId);
      }
      if (requestId) {
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
      return { ok: true };
    }

    case "TRANSLATE_IMAGE": {
      const requestId = message.requestId || `bg-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const controller = new AbortController();
      inflightControllers.set(requestId, controller);
      let completedOk = false;

      try {
        let imageBase64 = message.imageBase64 || "";
        if (!imageBase64 && message.imageUrl) {
          const referer = message.refererUrl || sender?.tab?.url || "";
          imageBase64 = await fetchImageAsBase64(message.imageUrl, referer, {
            ...deps,
            fetch: fetchImpl,
            signal: controller.signal,
          });
        }
        if (controller.signal.aborted) {
          return { ok: false, aborted: true };
        }
        if (!imageBase64) {
          return { ok: false, error: "No image data provided" };
        }

        const data = await fetchJson(
          `${serverBase}/translate`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify({
              image_base64: imageBase64,
              lang: message.lang || "auto",
              chapter_id: message.chapterId || "default",
              priority: message.priority ?? 0,
              glossary: message.glossary || {},
              request_id: requestId,
            }),
          },
          fetchImpl
        );
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

function registerExtensionListeners() {
  if (typeof chrome === "undefined" || !chrome.runtime) return;

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
    });
  }

  if (chrome.runtime.onStartup?.addListener) {
    chrome.runtime.onStartup.addListener(() => {
      void ensureUpdateAlarm();
      void runScheduledUpdateCheck();
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
    arrayBufferToBase64,
    fetchImageAsBase64,
    getInflightCount,
    getUpdateStorageState,
    maybeAutoReloadForBuild,
    ensureUpdateAlarm,
    runScheduledUpdateCheck,
    handleMessage,
    registerExtensionListeners,
  };
}
