"use strict";

const SERVER_BASE_URL = "http://127.0.0.1:8765";
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
      await dnr.updateSessionRules({ removeRuleIds: [addedRuleId] }).catch(() => {});
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
  const storage = deps.storage ?? (typeof chrome !== "undefined" ? chrome.storage?.local : null);
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

async function handleMessage(message, sender = {}, deps = {}) {
  const fetchImpl = deps.fetch || fetch;
  const serverBase = await getServerBaseUrl(deps);
  const type = message?.type;

  switch (type) {
    case "GET_HEALTH": {
      try {
        const data = await fetchJson(`${serverBase}/health`, { method: "GET" }, fetchImpl);
        return { ok: true, data };
      } catch (err) {
        return { ok: false, offline: true, error: err.message };
      }
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
        await fetchImpl(`${serverBase}/translate/cancel`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ request_id: requestId }),
        }).catch(() => {});
      }
      return { ok: true };
    }

    case "TRANSLATE_IMAGE": {
      const requestId = message.requestId || `bg-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const controller = new AbortController();
      inflightControllers.set(requestId, controller);

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
        return { ok: true, data };
      } catch (err) {
        if (err?.name === "AbortError" || controller.signal.aborted) {
          return { ok: false, aborted: true };
        }
        return { ok: false, error: err.message };
      } finally {
        inflightControllers.delete(requestId);
      }
    }

    default:
      return { ok: false, error: `Unknown message type: ${type}` };
  }
}

function registerExtensionListeners() {
  if (typeof chrome === "undefined" || !chrome.runtime) return;

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    handleMessage(message, sender)
      .then((res) => sendResponse(res))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  });

  if (chrome.runtime.onInstalled && chrome.contextMenus) {
    chrome.runtime.onInstalled.addListener(() => {
      chrome.contextMenus.removeAll(() => {
        chrome.contextMenus.create({
          id: "mlt-translate-image",
          title: "Translate this manga page",
          contexts: ["image"],
        });
      });
    });
  }

  if (chrome.contextMenus?.onClicked) {
    chrome.contextMenus.onClicked.addListener((info, tab) => {
      if (info.menuItemId === "mlt-translate-image" && tab?.id) {
        chrome.tabs
          .sendMessage(tab.id, {
            type: "MANUAL_TRANSLATE_IMAGE",
            srcUrl: info.srcUrl || "",
          })
          .catch(() => {});
      }
    });
  }
}

registerExtensionListeners();

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    SERVER_BASE_URL,
    arrayBufferToBase64,
    fetchImageAsBase64,
    handleMessage,
  };
}
