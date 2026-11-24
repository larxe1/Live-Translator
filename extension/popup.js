"use strict";

const DEFAULT_SERVER_URL = "http://127.0.0.1:8765";
const DEFAULT_PRELOAD_WHOLE_CHAPTER = false;
const DEFAULT_INCLUDE_FLOATING = false;
const hasDom = typeof document !== "undefined";

const statusDot = hasDom ? document.getElementById("statusDot") : null;
const statusText = hasDom ? document.getElementById("statusText") : null;
const statusVram = hasDom ? document.getElementById("statusVram") : null;
const gpuGuardNotice = hasDom ? document.getElementById("gpuGuardNotice") : null;
const versionBadge = hasDom ? document.getElementById("versionBadge") : null;
const siteHostEl = hasDom ? document.getElementById("siteHost") : null;
const siteToggle = hasDom ? document.getElementById("siteToggle") : null;
const floatingToggle = hasDom ? document.getElementById("floatingToggle") : null;
const preloadToggle = hasDom ? document.getElementById("preloadToggle") : null;
const langButtons = hasDom ? Array.from(document.querySelectorAll(".lang-btn")) : [];
const engineModeSelect = hasDom ? document.getElementById("engineModeSelect") : null;
const geminiKeyInput = hasDom ? document.getElementById("geminiKeyInput") : null;
const saveGeminiKeyBtn = hasDom ? document.getElementById("saveGeminiKeyBtn") : null;
const clearGeminiKeyBtn = hasDom ? document.getElementById("clearGeminiKeyBtn") : null;
const geminiKeyStatus = hasDom ? document.getElementById("geminiKeyStatus") : null;
const glossaryLookupForm = hasDom ? document.getElementById("glossaryLookupForm") : null;
const glossaryLookupInput = hasDom ? document.getElementById("glossaryLookupInput") : null;
const glossaryLookupBtn = hasDom ? document.getElementById("glossaryLookupBtn") : null;
const glossaryClearBtn = hasDom ? document.getElementById("glossaryClearBtn") : null;
const glossaryForm = hasDom ? document.getElementById("glossaryForm") : null;
const glossaryOrigInput = hasDom ? document.getElementById("glossaryOrig") : null;
const glossaryEngInput = hasDom ? document.getElementById("glossaryEng") : null;
const glossaryList = hasDom ? document.getElementById("glossaryList") : null;
const clearMemoryBtn = hasDom ? document.getElementById("clearMemoryBtn") : null;
const openTestBtn = hasDom ? document.getElementById("openTestBtn") : null;
const shutdownBtn = hasDom ? document.getElementById("shutdownBtn") : null;
const startEngineBtn = hasDom ? document.getElementById("startEngineBtn") : null;
const autoUpdateToggle = hasDom ? document.getElementById("autoUpdateToggle") : null;
const updateStatusText = hasDom ? document.getElementById("updateStatusText") : null;
const checkUpdatesBtn = hasDom ? document.getElementById("checkUpdatesBtn") : null;
const installUpdateBtn = hasDom ? document.getElementById("installUpdateBtn") : null;
const serverUrlInput = hasDom ? document.getElementById("serverUrlInput") : null;
const saveServerUrlBtn = hasDom ? document.getElementById("saveServerUrlBtn") : null;
const feedbackEl = hasDom ? document.getElementById("feedback") : null;

let currentHostname = "";
let currentTabId = null;
let currentChapterId = "default";
let currentSeriesId = "default";
let currentServerUrl = DEFAULT_SERVER_URL;
let currentGlossary = {};
let lastKnownUpdateStatus = null;
let feedbackTimer = null;

const HEALTH_POLL_INTERVAL_MS = 1500;
const HEALTH_POLL_MAX_MS = 180000;
const OFFLINE_TEXT = "Offline — click Start Engine";
const STAGE_LABELS = {
  initializing: "Initializing...",
  starting_llm: "Loading LLM...",
  restarting_llm: "Reloading LLM...",
  loading_detector: "Loading detector...",
  loading_ocr: "Loading OCR...",
  initializing_pipeline: "Preparing pipeline...",
  idle_unloaded: "LLM unloaded (idle)",
  unloaded: "Models not loaded",
};

let engineStarting = false;
let healthPollTimer = null;
let healthPollActive = false;
let healthPollStartedAt = 0;

function getLocalVersion() {
  try {
    return chrome.runtime?.getManifest?.()?.version || "1.3.1";
  } catch (_err) {
    return "1.3.1";
  }
}

async function sendBgMessage(message) {
  try {
    return await chrome.runtime.sendMessage(message);
  } catch (_err) {
    return null;
  }
}

function showFeedback(msg) {
  feedbackEl.textContent = msg;
  if (feedbackTimer) clearTimeout(feedbackTimer);
  feedbackTimer = setTimeout(() => {
    if (feedbackEl.textContent === msg) feedbackEl.textContent = "";
    feedbackTimer = null;
  }, 4000);
}

const SERIES_CONTAINER_SEGMENTS = new Set([
  "manga",
  "manhwa",
  "manhua",
  "comic",
  "comics",
  "webtoon",
  "webtoons",
  "series",
  "title",
  "titles",
  "read",
  "reader",
  "chapter",
  "chapters",
  "ch",
  "episode",
  "episodes",
  "ep",
  "book",
  "books",
  "novel",
  "novels",
  "en",
  "ja",
  "jp",
  "ko",
  "kr",
]);
const CHAPTER_KEYWORD_SEGMENTS = new Set([
  "chapter",
  "chapters",
  "chap",
  "ch",
  "episode",
  "episodes",
  "ep",
  "page",
  "pages",
  "pg",
  "p",
  "vol",
  "volume",
  "part",
  "issue",
]);

function deriveSeriesId(hostname, pathname) {
  const host = String(hostname || "").trim().toLowerCase();
  const rawPath = String(pathname || "/").split("?")[0].split("#")[0].trim();
  const normalizedPath = rawPath ? (rawPath.startsWith("/") ? rawPath : `/${rawPath}`) : "/";
  const fallback = host ? `${host}${normalizedPath}` : normalizedPath || "default";

  const parts = normalizedPath.split("/").filter(Boolean);
  if (parts.length === 0) return fallback;

  let end = parts.length;
  while (end > 0) {
    const seg = parts[end - 1].toLowerCase();
    if (/^(?:viewer|view|read|index\.html?)$/.test(seg)) {
      end -= 1;
      continue;
    }
    if (
      /^(?:chapter|chapters|chap|ch|episode|episodes|ep|page|pages|pg|vol|volume|part|issue)[-_.\s]*\d+(?:[-_.][a-z0-9]+)*$/.test(
        seg
      ) ||
      /^(?:c|e|p)[-_.]?\d+(?:[-_.]\d+)*$/.test(seg) ||
      /^(?:v|s)\d+[-_.]?(?:ch|ep|c|e)[-_.]?\d+(?:[-_.][a-z0-9]+)*$/.test(seg)
    ) {
      end -= 1;
      continue;
    }
    if (/^\d+(?:[-_.]\d+)*(?:[a-z])?$/.test(seg)) {
      const prev = end >= 2 ? parts[end - 2].toLowerCase() : "";
      if (CHAPTER_KEYWORD_SEGMENTS.has(prev)) {
        end -= 2;
        continue;
      }
      if (end >= 3 && !SERIES_CONTAINER_SEGMENTS.has(prev)) {
        end -= 1;
        continue;
      }
    }
    break;
  }

  if (end === 0 || end === parts.length) {
    return fallback;
  }
  const lastKept = parts[end - 1].toLowerCase();
  if (SERIES_CONTAINER_SEGMENTS.has(lastKept)) {
    return fallback;
  }

  const strippedPath = `/${parts.slice(0, end).join("/")}`;
  return host ? `${host}${strippedPath}` : strippedPath;
}

function renderVersionBadge(buildId = "", version = getLocalVersion()) {
  if (!versionBadge) return;
  const shortHash = buildId ? String(buildId).slice(0, 8) : "";
  versionBadge.textContent = shortHash ? `v${version} · ${shortHash}` : `v${version}`;
}

function renderUpdateUI(statusData, autoEnabled, localVersion = getLocalVersion()) {
  if (autoUpdateToggle) {
    autoUpdateToggle.checked = autoEnabled !== false;
  }
  renderVersionBadge(statusData?.extension_build_id || "", localVersion);

  if (!updateStatusText) return;
  updateStatusText.classList.remove("has-update");

  if (!statusData) {
    updateStatusText.textContent = "Server offline";
    if (installUpdateBtn) installUpdateBtn.hidden = true;
    return;
  }

  if (statusData.updating) {
    updateStatusText.textContent = "Installing update...";
    if (installUpdateBtn) installUpdateBtn.hidden = true;
    return;
  }

  if (statusData.update_available) {
    updateStatusText.classList.add("has-update");
    const behind = Number(statusData.commits_behind || 0);
    updateStatusText.textContent =
      behind > 0 ? `Update available! (${behind} behind)` : "Update available!";
    if (installUpdateBtn) installUpdateBtn.hidden = false;
    return;
  }

  if (installUpdateBtn) installUpdateBtn.hidden = true;

  if (statusData.has_remote) {
    updateStatusText.textContent = "Up to date with GitHub";
    return;
  }

  updateStatusText.textContent =
    autoEnabled === false
      ? "Auto-update paused (manual mode)"
      : "Auto-update active (local live-reload)";
}

function formatLoadingStage(stage) {
  const raw = String(stage || "").trim();
  if (!raw) return "Loading models...";
  if (STAGE_LABELS[raw]) return STAGE_LABELS[raw];
  if (raw.startsWith("error")) return `Engine error: ${raw.replace(/^error:\s*/, "")}`;
  return `Loading models... (${raw})`;
}

/**
 * state: "offline" | "starting" | "loading" | "online"
 */
function setEngineControls(state) {
  const showStart = state === "offline" || state === "starting";
  if (startEngineBtn) {
    startEngineBtn.hidden = !showStart;
    startEngineBtn.disabled = state === "starting";
    startEngineBtn.textContent = state === "starting" ? "Starting Engine..." : "Start Local Engine";
  }
  if (shutdownBtn) {
    shutdownBtn.hidden = showStart;
  }
}

function formatGpuTelemetry(health) {
  if (!health || typeof health !== "object") return "";
  const guard = health.gpu_guard && typeof health.gpu_guard === "object" ? health.gpu_guard : null;

  const rawTotal = Number(health.vram_total_mb || guard?.vram_total_mb || 0);
  const rawUsed = Number(
    health.vram_total_mb > 0 ? health.vram_used_mb : guard?.vram_used_mb ?? health.vram_used_mb ?? 0
  );
  const vramStr =
    Number.isFinite(rawTotal) && rawTotal > 0
      ? `${Math.max(0, Math.round(rawUsed))}/${Math.round(rawTotal)} MB`
      : "";

  const rawTemp = guard && guard.temp_c !== undefined && guard.temp_c !== null ? Number(guard.temp_c) : NaN;
  const tempStr = Number.isFinite(rawTemp) && rawTemp > 0 ? `${Math.round(rawTemp)}°C` : "";

  if (vramStr && tempStr) return `${vramStr} · ${tempStr}`;
  return vramStr || tempStr;
}

function formatGpuGuardNotice(gpuGuard) {
  if (!gpuGuard || typeof gpuGuard !== "object") {
    return { visible: false, state: "ok", text: "", title: "" };
  }
  const state = String(gpuGuard.state || "").trim().toLowerCase();
  const rawTemp =
    gpuGuard.temp_c !== undefined && gpuGuard.temp_c !== null ? Number(gpuGuard.temp_c) : NaN;
  const tempC = Number.isFinite(rawTemp) && rawTemp > 0 ? Math.round(rawTemp) : null;
  const msg = String(gpuGuard.message || "").trim();

  if (state === "throttled") {
    return {
      visible: true,
      state: "throttled",
      text: tempC !== null ? `Cooling GPU (${tempC}°C)...` : msg || "Cooling GPU...",
      title: msg,
    };
  }
  if (state === "paused") {
    return {
      visible: true,
      state: "paused",
      text: tempC !== null ? `Paused: GPU hot (${tempC}°C)` : msg || "Paused: GPU hot",
      title: msg,
    };
  }
  return { visible: false, state: state || "ok", text: "", title: "" };
}

function applyGpuGuardNotice(gpuGuard, noticeEl = gpuGuardNotice) {
  const info = formatGpuGuardNotice(gpuGuard);
  if (!noticeEl) return info;
  noticeEl.hidden = !info.visible;
  noticeEl.textContent = info.text;
  noticeEl.title = info.title || "";
  noticeEl.className =
    info.state === "paused" ? "gpu-guard-notice paused" : "gpu-guard-notice";
  return info;
}

const VALID_TRANSLATION_MODES = new Set(["hybrid", "local", "gemini"]);

function formatLocalModelLabel(modelName) {
  const raw = String(modelName || "").trim();
  if (!raw) return "";
  if (/shisa/i.test(raw)) return "Shisa JP";
  if (/exaone/i.test(raw)) return "EXAONE KR";
  if (/qwen2\.5.*abliterated/i.test(raw)) return "Qwen2.5 Abliterated";
  const base = raw.split(/[\\/]/).pop().replace(/\.gguf$/i, "");
  return base.length > 22 ? `${base.slice(0, 19)}...` : base;
}

function formatEngineStatusLabel(h) {
  if (!h || typeof h !== "object") return "Ready";
  const mode = String(h.translation_mode || "").trim().toLowerCase();
  if (!VALID_TRANSLATION_MODES.has(mode)) {
    return `Ready — ${h.gpu_name || "Local GPU"}`;
  }
  const localModel = formatLocalModelLabel(h.active_local_model || h.llm_model);
  if (mode === "gemini") {
    return h.gemini_configured ? "Gemini Vision Only" : "Gemini (Key needed)";
  }
  if (mode === "local") {
    return localModel ? `Local · ${localModel}` : "Local · Shisa/EXAONE";
  }
  if (h.gemini_configured) {
    return h.llm_ready && localModel
      ? `Hybrid · Gemini + ${localModel}`
      : "Hybrid · Gemini + Shisa/EXAONE";
  }
  return localModel ? `Hybrid (No Key) · ${localModel}` : "Hybrid (No Key) · Local";
}

function formatGeminiKeyStatus(data) {
  if (data && data.gemini_configured) {
    const hint = String(data.gemini_key_hint || "").trim();
    return {
      configured: true,
      text: hint ? `Saved (${hint})` : "Saved",
      className: "key-status configured",
    };
  }
  return {
    configured: false,
    text: "Not configured",
    className: "key-status",
  };
}

function applySettingsResponse(data, dom = { engineModeSelect, geminiKeyStatus }) {
  if (!data || typeof data !== "object") return;
  const mode = String(data.translation_mode || "").trim().toLowerCase();
  if (dom.engineModeSelect && VALID_TRANSLATION_MODES.has(mode)) {
    const isFocused = hasDom && document.activeElement === dom.engineModeSelect;
    if (!isFocused) {
      dom.engineModeSelect.value = mode;
    }
  }
  if (dom.geminiKeyStatus && ("gemini_configured" in data || "gemini_key_hint" in data)) {
    const keyInfo = formatGeminiKeyStatus(data);
    dom.geminiKeyStatus.textContent = keyInfo.text;
    dom.geminiKeyStatus.className = keyInfo.className;
  }
}

function applyHealthUI(
  h,
  dom = {
    statusDot,
    statusText,
    statusVram,
    gpuGuardNotice,
    engineModeSelect,
    geminiKeyStatus,
  }
) {
  if (!h || typeof h !== "object") return { ready: false, notice: formatGpuGuardNotice(null) };
  const notice = applyGpuGuardNotice(h.gpu_guard, dom.gpuGuardNotice);
  const telemetry = formatGpuTelemetry(h);
  applySettingsResponse(h, dom);
  if (h.ready) {
    if (dom.statusDot) {
      dom.statusDot.className =
        notice.state === "paused" ? "status-dot loading" : "status-dot ready";
    }
    if (dom.statusText) {
      dom.statusText.textContent = formatEngineStatusLabel(h);
    }
    if (dom.statusVram) {
      dom.statusVram.textContent = telemetry;
    }
    return { ready: true, notice };
  }
  if (dom.statusDot) {
    dom.statusDot.className = "status-dot loading";
  }
  if (dom.statusText) {
    dom.statusText.textContent = formatLoadingStage(h.loading_stage);
  }
  if (dom.statusVram) {
    dom.statusVram.textContent = telemetry;
  }
  return { ready: false, notice };
}

function renderOffline(message = OFFLINE_TEXT) {
  if (statusDot) statusDot.className = "status-dot";
  if (statusText) statusText.textContent = message;
  if (statusVram) statusVram.textContent = "";
  applyGpuGuardNotice(null);
  setEngineControls("offline");
}

function renderStarting(message = "Starting local engine...") {
  if (statusDot) statusDot.className = "status-dot loading";
  if (statusText) statusText.textContent = message;
  if (statusVram) statusVram.textContent = "";
  applyGpuGuardNotice(null);
  setEngineControls("starting");
}

async function notifyActiveTabEngineReady() {
  if (!currentTabId || !siteToggle?.checked || typeof chrome?.tabs?.sendMessage !== "function") {
    return;
  }
  try {
    await chrome.tabs.sendMessage(currentTabId, { type: "ENGINE_READY" });
  } catch (_err) {
    // Ignore tabs without an active content script.
  }
}

/**
 * Renders current server health and returns { online, ready, starting }.
 * Never schedules polling itself; callers decide.
 */
async function refreshHealth() {
  const res = await sendBgMessage({ type: "GET_HEALTH" });
  if (!res || !res.ok || !res.data) {
    const starting = engineStarting || Boolean(res?.starting);
    if (starting) {
      renderStarting();
    } else {
      renderOffline();
    }
    return { online: false, ready: false, starting };
  }
  const h = res.data;
  if (h.extension_build_id) {
    renderVersionBadge(h.extension_build_id, h.extension_version || getLocalVersion());
  }
  if (h.ready) {
    engineStarting = false;
    applyHealthUI(h);
    setEngineControls("online");
    void notifyActiveTabEngineReady();
    return { online: true, ready: true, starting: false };
  }
  applyHealthUI(h);
  setEngineControls("loading");
  const failed = String(h.loading_stage || "").startsWith("error");
  return { online: true, ready: false, starting: !failed, failed };
}

function stopHealthPolling() {
  healthPollActive = false;
  healthPollStartedAt = 0;
  if (healthPollTimer !== null) {
    clearTimeout(healthPollTimer);
    healthPollTimer = null;
  }
}

function scheduleHealthPoll() {
  if (healthPollTimer !== null) clearTimeout(healthPollTimer);
  healthPollTimer = setTimeout(() => {
    healthPollTimer = null;
    void pollHealthTick();
  }, HEALTH_POLL_INTERVAL_MS);
}

function startHealthPolling() {
  if (healthPollActive) return;
  healthPollActive = true;
  healthPollStartedAt = Date.now();
  scheduleHealthPoll();
}

async function pollHealthTick() {
  if (!healthPollActive) return;
  const state = await refreshHealth();
  if (!healthPollActive) return;

  if (state.ready) {
    stopHealthPolling();
    showFeedback("Local engine ready");
    void refreshUpdateStatus();
    return;
  }
  if (state.failed) {
    engineStarting = false;
    stopHealthPolling();
    return;
  }
  if (Date.now() - healthPollStartedAt > HEALTH_POLL_MAX_MS) {
    engineStarting = false;
    stopHealthPolling();
    renderOffline("Engine did not start — check cache/server.log");
    return;
  }
  if (!state.online && !state.starting) {
    stopHealthPolling();
    return;
  }
  scheduleHealthPoll();
}

async function startEngine() {
  if (engineStarting) return false;
  engineStarting = true;
  renderStarting();

  const res = await sendBgMessage({ type: "START_SERVER" });
  if (!res || !res.ok) {
    engineStarting = false;
    stopHealthPolling();
    if (res?.hostMissing) {
      renderOffline("Launcher not installed — re-run setup.ps1");
      showFeedback("Native launcher missing. Run setup.ps1, then reload the extension.");
    } else {
      renderOffline(OFFLINE_TEXT);
      showFeedback(res?.error ? `Start failed: ${res.error}` : "Start failed");
    }
    return false;
  }

  showFeedback(res.already_running ? "Local engine already running" : "Starting local engine...");
  startHealthPolling();
  return true;
}

async function refreshHealthAndPoll() {
  const state = await refreshHealth();
  if (!state.ready && (state.starting || (state.online && !state.failed))) {
    startHealthPolling();
  }
  return state;
}

async function refreshUpdateStatus() {
  const res = await sendBgMessage({ type: "GET_UPDATE_STATUS" });
  const localVersion = res?.localVersion || getLocalVersion();
  const autoEnabled = res?.autoUpdateEnabled !== false;
  lastKnownUpdateStatus = res?.data || null;
  renderUpdateUI(lastKnownUpdateStatus, autoEnabled, localVersion);
}

function renderLangButtons(langMode) {
  for (const btn of langButtons) {
    btn.classList.toggle("active", btn.dataset.lang === langMode);
  }
}

function renderGlossary() {
  glossaryList.textContent = "";
  const entries = Object.entries(currentGlossary);
  if (entries.length === 0) {
    const empty = document.createElement("div");
    empty.className = "glossary-empty";
    empty.textContent = "No pinned character names yet.";
    glossaryList.appendChild(empty);
    return;
  }

  for (const [orig, eng] of entries) {
    const item = document.createElement("div");
    item.className = "glossary-item";

    const pair = document.createElement("span");
    pair.className = "glossary-pair";
    pair.textContent = `${orig} → ${eng}`;

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "glossary-del";
    delBtn.textContent = "×";
    delBtn.title = "Remove";
    delBtn.addEventListener("click", async () => {
      delete currentGlossary[orig];
      await syncGlossary();
    });

    item.appendChild(pair);
    item.appendChild(delBtn);
    glossaryList.appendChild(item);
  }
}

async function syncGlossary() {
  try {
    const stored = (await chrome.storage.local.get(["glossariesBySeries"])) || {};
    const nextBySeries = { ...(stored.glossariesBySeries || {}) };
    nextBySeries[currentSeriesId] = { ...currentGlossary };
    await chrome.storage.local.set({
      glossariesBySeries: nextBySeries,
      glossary: { ...currentGlossary },
    });
  } catch (_err) {
    // Ignore storage write failures.
  }
  renderGlossary();
  await sendBgMessage({
    type: "SET_GLOSSARY",
    chapterId: currentChapterId,
    seriesId: currentSeriesId,
    entries: currentGlossary,
  });
}

async function initPopup() {
  renderVersionBadge("", getLocalVersion());

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.url) {
      const u = new URL(tab.url);
      if (u.protocol === "http:" || u.protocol === "https:") {
        currentTabId = tab.id ?? null;
        currentHostname = u.hostname;
        currentChapterId = `${u.hostname}${u.pathname}`;
        currentSeriesId = deriveSeriesId(u.hostname, u.pathname);
      }
    }
  } catch (_err) {
    // Ignore tab query or malformed URL failures.
  }

  let stored = {};
  try {
    stored =
      (await chrome.storage.local.get([
        "enabledDomains",
        "langMode",
        "translationMode",
        "includeFloating",
        "preloadWholeChapter",
        "glossary",
        "glossariesBySeries",
        "serverUrl",
        "autoUpdateEnabled",
      ])) || {};
  } catch (_err) {
    stored = {};
  }

  const enabledDomains = stored.enabledDomains || {};
  if (currentHostname) {
    siteHostEl.textContent = currentHostname;
    siteToggle.checked = Boolean(enabledDomains[currentHostname]);
    siteToggle.disabled = false;
  } else {
    siteHostEl.textContent = "Unavailable on browser system pages";
    siteToggle.checked = false;
    siteToggle.disabled = true;
  }

  if (floatingToggle) {
    floatingToggle.checked = stored.includeFloating === true;
  }
  if (preloadToggle) {
    preloadToggle.checked = stored.preloadWholeChapter === true;
  }

  renderLangButtons(stored.langMode || "auto");
  if (engineModeSelect) {
    const storedMode = String(stored.translationMode || "").trim().toLowerCase();
    if (VALID_TRANSLATION_MODES.has(storedMode)) {
      engineModeSelect.value = storedMode;
    }
  }
  const bySeries = stored.glossariesBySeries;
  if (bySeries && Object.prototype.hasOwnProperty.call(bySeries, currentSeriesId)) {
    currentGlossary = { ...(bySeries[currentSeriesId] || {}) };
  } else if (!bySeries) {
    currentGlossary = { ...(stored.glossary || {}) };
  } else {
    currentGlossary = {};
  }
  renderGlossary();

  currentServerUrl = stored.serverUrl || DEFAULT_SERVER_URL;
  serverUrlInput.value = currentServerUrl;
  if (autoUpdateToggle) {
    autoUpdateToggle.checked = stored.autoUpdateEnabled !== false;
  }

  if (currentHostname && currentTabId) {
    void sendBgMessage({ type: "ENSURE_CONTENT_SCRIPT", tabId: currentTabId });
  }
  await Promise.all([refreshHealthAndPoll(), refreshUpdateStatus()]);
}

if (siteToggle) {
  siteToggle.addEventListener("change", async () => {
    if (!currentHostname) return;
    const stored = await chrome.storage.local.get(["enabledDomains"]);
    const next = { ...(stored.enabledDomains || {}), [currentHostname]: siteToggle.checked };
    await chrome.storage.local.set({ enabledDomains: next });
    if (siteToggle.checked && currentTabId) {
      await sendBgMessage({ type: "ENSURE_CONTENT_SCRIPT", tabId: currentTabId });
    }
    showFeedback(
      siteToggle.checked
        ? `Enabled on ${currentHostname}`
        : `Disabled on ${currentHostname}`
    );
    if (!siteToggle.checked) return;

    const state = await refreshHealth();
    if (state.ready) {
      void notifyActiveTabEngineReady();
      return;
    }
    if (!state.online && !state.starting) {
      await startEngine();
    } else if (!state.failed) {
      startHealthPolling();
    }
  });
}

if (startEngineBtn) {
  startEngineBtn.addEventListener("click", async () => {
    await startEngine();
  });
}

for (const btn of langButtons) {
  btn.addEventListener("click", async () => {
    const langMode = btn.dataset.lang || "auto";
    await chrome.storage.local.set({ langMode });
    renderLangButtons(langMode);
    showFeedback(`Language set to ${btn.textContent}`);
  });
}

if (engineModeSelect) {
  engineModeSelect.addEventListener("change", async () => {
    const translationMode = String(engineModeSelect.value || "hybrid").trim().toLowerCase();
    try {
      await chrome.storage.local.set({ translationMode });
    } catch (_err) {
      // Ignore storage write failures.
    }
    const res = await sendBgMessage({
      type: "SAVE_SETTINGS",
      translation_mode: translationMode,
    });
    if (res?.ok) {
      applySettingsResponse(res);
      const selectedLabel =
        engineModeSelect.options?.[engineModeSelect.selectedIndex]?.text || translationMode;
      showFeedback(`Engine set to ${selectedLabel}`);
      void refreshHealthAndPoll();
    } else {
      showFeedback("Engine mode saved locally (server offline)");
    }
  });
}

if (saveGeminiKeyBtn) {
  saveGeminiKeyBtn.addEventListener("click", async () => {
    const apiKey = (geminiKeyInput?.value || "").trim();
    if (!apiKey) {
      showFeedback("Paste a Google AI Studio key first");
      return;
    }
    saveGeminiKeyBtn.disabled = true;
    if (clearGeminiKeyBtn) clearGeminiKeyBtn.disabled = true;
    try {
      const res = await sendBgMessage({
        type: "SAVE_SETTINGS",
        gemini_api_key: apiKey,
      });
      if (!res?.ok) {
        showFeedback(res?.error || "Failed to save key — is the server running?");
        return;
      }
      if (geminiKeyInput) geminiKeyInput.value = "";
      applySettingsResponse(res);
      showFeedback("Gemini API key saved");
      await refreshHealthAndPoll();
    } finally {
      saveGeminiKeyBtn.disabled = false;
      if (clearGeminiKeyBtn) clearGeminiKeyBtn.disabled = false;
    }
  });
}

if (clearGeminiKeyBtn) {
  clearGeminiKeyBtn.addEventListener("click", async () => {
    if (saveGeminiKeyBtn) saveGeminiKeyBtn.disabled = true;
    clearGeminiKeyBtn.disabled = true;
    try {
      const res = await sendBgMessage({
        type: "SAVE_SETTINGS",
        clear_gemini_api_key: true,
      });
      if (!res?.ok) {
        showFeedback(res?.error || "Failed to clear key — is the server running?");
        return;
      }
      if (geminiKeyInput) geminiKeyInput.value = "";
      applySettingsResponse(res);
      showFeedback("Gemini API key cleared");
      await refreshHealthAndPoll();
    } finally {
      if (saveGeminiKeyBtn) saveGeminiKeyBtn.disabled = false;
      clearGeminiKeyBtn.disabled = false;
    }
  });
}

if (floatingToggle) {
  floatingToggle.addEventListener("change", async () => {
    const includeFloating = floatingToggle.checked;
    try {
      await chrome.storage.local.set({ includeFloating });
    } catch (_err) {
      // Ignore storage write failures.
    }
    showFeedback(
      includeFloating ? "Floating dialogue & SFX enabled" : "Floating dialogue & SFX hidden"
    );
  });
}

if (preloadToggle) {
  preloadToggle.addEventListener("change", async () => {
    const preloadWholeChapter = Boolean(preloadToggle.checked);
    try {
      await chrome.storage.local.set({ preloadWholeChapter });
    } catch (_err) {
      // Ignore storage write failures.
    }
    showFeedback(
      preloadWholeChapter
        ? "Background chapter preloading enabled"
        : "Background chapter preloading disabled"
    );
  });
}

function formatLookupSourceName(source) {
  const map = {
    anilist: "AniList",
    wikipedia: "Wikipedia",
    fandom: "Fandom",
  };
  const key = String(source || "").trim().toLowerCase();
  return map[key] || source || "Wiki";
}

if (glossaryLookupForm) {
  glossaryLookupForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = (glossaryLookupInput?.value || "").trim();
    if (!query) {
      showFeedback("Enter a title or Wikipedia / Fandom / AniList URL");
      return;
    }

    if (glossaryLookupBtn) {
      glossaryLookupBtn.disabled = true;
      glossaryLookupBtn.textContent = "Looking up...";
    }
    try {
      const res = await sendBgMessage({
        type: "LOOKUP_GLOSSARY",
        query,
        source: "auto",
      });
      if (!res || !res.ok) {
        showFeedback(res?.error || "Lookup failed — is the local engine running?");
        return;
      }

      const data = res.data || {};
      const entries =
        data.entries && typeof data.entries === "object" ? data.entries : {};
      const pairs = Object.entries(entries);
      if (pairs.length === 0) {
        showFeedback("No character names found — try pasting a Wikipedia or Fandom URL");
        return;
      }

      let added = 0;
      for (const [orig, eng] of pairs) {
        if (!orig || !eng) continue;
        if (!Object.prototype.hasOwnProperty.call(currentGlossary, orig)) {
          currentGlossary[orig] = eng;
          added += 1;
        }
      }
      await syncGlossary();

      const sourceLabel = formatLookupSourceName(data.source);
      const titleSuffix = data.title ? ` (${data.title})` : "";
      if (added > 0) {
        showFeedback(
          `Imported ${added} name${added === 1 ? "" : "s"} from ${sourceLabel}${titleSuffix}`
        );
      } else {
        showFeedback(
          `All ${pairs.length} names from ${sourceLabel}${titleSuffix} already pinned`
        );
      }
    } finally {
      if (glossaryLookupBtn) {
        glossaryLookupBtn.disabled = false;
        glossaryLookupBtn.textContent = "Import";
      }
    }
  });
}

if (glossaryClearBtn) {
  glossaryClearBtn.addEventListener("click", async () => {
    currentGlossary = {};
    await syncGlossary();
    showFeedback("Glossary cleared");
  });
}

if (glossaryForm) {
  glossaryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const orig = glossaryOrigInput.value.trim();
    const eng = glossaryEngInput.value.trim();
    if (!orig || !eng) return;
    currentGlossary[orig] = eng;
    glossaryOrigInput.value = "";
    glossaryEngInput.value = "";
    await syncGlossary();
    showFeedback(`Saved ${orig} → ${eng}`);
  });
}

if (clearMemoryBtn) {
  clearMemoryBtn.addEventListener("click", async () => {
    const res = await sendBgMessage({
      type: "CLEAR_MEMORY",
      chapterId: currentChapterId,
      seriesId: currentSeriesId,
    });
    showFeedback(res?.ok ? "Chapter memory cleared" : "Server offline");
  });
}

if (openTestBtn) {
  openTestBtn.addEventListener("click", () => {
    const base = (currentServerUrl || DEFAULT_SERVER_URL).replace(/\/+$/, "");
    chrome.tabs.create({ url: `${base}/test` });
  });
}

if (shutdownBtn) {
  shutdownBtn.addEventListener("click", async () => {
    stopHealthPolling();
    engineStarting = false;
    const res = await sendBgMessage({ type: "SHUTDOWN_SERVER" });
    if (res?.ok) {
      renderOffline("Offline — VRAM Freed");
      if (updateStatusText) updateStatusText.textContent = "Server offline";
      showFeedback("Local GPU engine stopped");
    } else {
      renderOffline();
      showFeedback("Server already offline");
    }
  });
}

if (autoUpdateToggle) {
  autoUpdateToggle.addEventListener("change", async () => {
    const enabled = autoUpdateToggle.checked;
    const res = await sendBgMessage({ type: "SET_AUTO_UPDATE", enabled });
    const actual = res?.ok ? res.autoUpdateEnabled !== false : enabled;
    renderUpdateUI(lastKnownUpdateStatus, actual);
    showFeedback(actual ? "Auto-update enabled" : "Auto-update paused");
  });
}

if (checkUpdatesBtn) {
  checkUpdatesBtn.addEventListener("click", async () => {
    checkUpdatesBtn.disabled = true;
    const prevLabel = checkUpdatesBtn.textContent;
    checkUpdatesBtn.textContent = "Checking...";
    try {
      const res = await sendBgMessage({ type: "CHECK_FOR_UPDATES" });
      if (!res?.ok) {
        renderUpdateUI(null, autoUpdateToggle?.checked !== false);
        showFeedback(res?.error ? `Check failed: ${res.error}` : "Server offline");
        return;
      }
      lastKnownUpdateStatus = res.data || null;
      renderUpdateUI(lastKnownUpdateStatus, autoUpdateToggle?.checked !== false);
      if (res.reloaded) {
        showFeedback("Extension updated — reloading...");
      } else if (res.data?.update_available) {
        showFeedback("Update available!");
      } else if (res.data?.has_remote) {
        showFeedback("Up to date with GitHub");
      } else {
        showFeedback("Checked local extension build");
      }
    } finally {
      checkUpdatesBtn.disabled = false;
      checkUpdatesBtn.textContent = prevLabel;
    }
  });
}

if (installUpdateBtn) {
  installUpdateBtn.addEventListener("click", async () => {
    installUpdateBtn.disabled = true;
    const prevLabel = installUpdateBtn.textContent;
    installUpdateBtn.textContent = "Installing...";
    try {
      const res = await sendBgMessage({ type: "APPLY_UPDATE" });
      if (!res?.ok) {
        showFeedback(res?.error || "Update failed");
        return;
      }
      if (res.data?.status) {
        lastKnownUpdateStatus = res.data.status;
        renderUpdateUI(lastKnownUpdateStatus, autoUpdateToggle?.checked !== false);
      }
      if (res.reloaded) {
        showFeedback("Extension updated — reloading...");
      } else {
        showFeedback(res.data?.message || "Update applied");
      }
    } finally {
      installUpdateBtn.disabled = false;
      installUpdateBtn.textContent = prevLabel;
    }
  });
}

if (saveServerUrlBtn) {
  saveServerUrlBtn.addEventListener("click", async () => {
    const raw = serverUrlInput.value.trim().replace(/\/+$/, "");
    const clean = raw || DEFAULT_SERVER_URL;
    if (!/^https?:\/\/[^/\s]+/i.test(clean)) {
      showFeedback("Enter a valid http:// or https:// URL");
      return;
    }
    currentServerUrl = clean;
    serverUrlInput.value = clean;
    try {
      await chrome.storage.local.set({ serverUrl: clean });
    } catch (_err) {
      // Ignore storage failures.
    }
    showFeedback("Server URL saved");
    await Promise.all([refreshHealthAndPoll(), refreshUpdateStatus()]);
  });
}

if (typeof module === "undefined" && typeof document !== "undefined") {
  if (typeof window !== "undefined") {
    window.addEventListener("pagehide", stopHealthPolling);
  }

  void (async () => {
    try {
      await initPopup();
    } catch (_err) {
      // Ignore popup init errors when closed mid-load.
    }
  })();
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    DEFAULT_PRELOAD_WHOLE_CHAPTER,
    DEFAULT_INCLUDE_FLOATING,
    deriveSeriesId,
    formatGpuTelemetry,
    formatGpuGuardNotice,
    applyGpuGuardNotice,
    formatLocalModelLabel,
    formatEngineStatusLabel,
    formatGeminiKeyStatus,
    applySettingsResponse,
    applyHealthUI,
  };
}
