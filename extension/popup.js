"use strict";

const DEFAULT_SERVER_URL = "http://127.0.0.1:8765";

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const statusVram = document.getElementById("statusVram");
const versionBadge = document.getElementById("versionBadge");
const siteHostEl = document.getElementById("siteHost");
const siteToggle = document.getElementById("siteToggle");
const langButtons = Array.from(document.querySelectorAll(".lang-btn"));
const glossaryForm = document.getElementById("glossaryForm");
const glossaryOrigInput = document.getElementById("glossaryOrig");
const glossaryEngInput = document.getElementById("glossaryEng");
const glossaryList = document.getElementById("glossaryList");
const clearMemoryBtn = document.getElementById("clearMemoryBtn");
const openTestBtn = document.getElementById("openTestBtn");
const shutdownBtn = document.getElementById("shutdownBtn");
const autoUpdateToggle = document.getElementById("autoUpdateToggle");
const updateStatusText = document.getElementById("updateStatusText");
const checkUpdatesBtn = document.getElementById("checkUpdatesBtn");
const installUpdateBtn = document.getElementById("installUpdateBtn");
const serverUrlInput = document.getElementById("serverUrlInput");
const saveServerUrlBtn = document.getElementById("saveServerUrlBtn");
const feedbackEl = document.getElementById("feedback");

let currentHostname = "";
let currentChapterId = "default";
let currentServerUrl = DEFAULT_SERVER_URL;
let currentGlossary = {};
let lastKnownUpdateStatus = null;

function getLocalVersion() {
  try {
    return chrome.runtime?.getManifest?.()?.version || "1.1.0";
  } catch (_err) {
    return "1.1.0";
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
  setTimeout(() => {
    if (feedbackEl.textContent === msg) feedbackEl.textContent = "";
  }, 3000);
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

async function refreshHealth() {
  const res = await sendBgMessage({ type: "GET_HEALTH" });
  if (!res || !res.ok || !res.data) {
    statusDot.className = "status-dot";
    statusText.textContent = "Offline — open Start Manga Translator";
    statusVram.textContent = "";
    return;
  }
  const h = res.data;
  if (h.extension_build_id) {
    renderVersionBadge(h.extension_build_id, h.extension_version || getLocalVersion());
  }
  if (h.ready) {
    statusDot.className = "status-dot ready";
    statusText.textContent = `Ready — ${h.gpu_name || "Local GPU"}`;
    statusVram.textContent =
      h.vram_total_mb > 0 ? `${h.vram_used_mb}/${h.vram_total_mb} MB` : "";
  } else {
    statusDot.className = "status-dot loading";
    const stage = h.loading_stage ? ` (${h.loading_stage})` : "";
    statusText.textContent = `Loading models...${stage}`;
    statusVram.textContent = "";
  }
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
    await chrome.storage.local.set({ glossary: currentGlossary });
  } catch (_err) {
    // Ignore storage write failures.
  }
  renderGlossary();
  await sendBgMessage({
    type: "SET_GLOSSARY",
    chapterId: currentChapterId,
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
        currentHostname = u.hostname;
        currentChapterId = `${u.hostname}${u.pathname}`;
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
        "glossary",
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

  renderLangButtons(stored.langMode || "auto");
  currentGlossary = { ...(stored.glossary || {}) };
  renderGlossary();

  currentServerUrl = stored.serverUrl || DEFAULT_SERVER_URL;
  serverUrlInput.value = currentServerUrl;
  if (autoUpdateToggle) {
    autoUpdateToggle.checked = stored.autoUpdateEnabled !== false;
  }

  await Promise.all([refreshHealth(), refreshUpdateStatus()]);
}

siteToggle.addEventListener("change", async () => {
  if (!currentHostname) return;
  const stored = await chrome.storage.local.get(["enabledDomains"]);
  const next = { ...(stored.enabledDomains || {}), [currentHostname]: siteToggle.checked };
  await chrome.storage.local.set({ enabledDomains: next });
  showFeedback(
    siteToggle.checked
      ? `Enabled on ${currentHostname}`
      : `Disabled on ${currentHostname}`
  );
});

for (const btn of langButtons) {
  btn.addEventListener("click", async () => {
    const langMode = btn.dataset.lang || "auto";
    await chrome.storage.local.set({ langMode });
    renderLangButtons(langMode);
    showFeedback(`Language set to ${btn.textContent}`);
  });
}

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

clearMemoryBtn.addEventListener("click", async () => {
  const res = await sendBgMessage({ type: "CLEAR_MEMORY", chapterId: currentChapterId });
  showFeedback(res?.ok ? "Chapter memory cleared" : "Server offline");
});

openTestBtn.addEventListener("click", () => {
  const base = (currentServerUrl || DEFAULT_SERVER_URL).replace(/\/+$/, "");
  chrome.tabs.create({ url: `${base}/test` });
});

shutdownBtn.addEventListener("click", async () => {
  const res = await sendBgMessage({ type: "SHUTDOWN_SERVER" });
  if (res?.ok) {
    statusDot.className = "status-dot";
    statusText.textContent = "Offline — VRAM Freed";
    statusVram.textContent = "";
    if (updateStatusText) updateStatusText.textContent = "Server offline";
    showFeedback("Local GPU engine stopped");
  } else {
    showFeedback("Server already offline");
  }
});

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
  await Promise.all([refreshHealth(), refreshUpdateStatus()]);
});

void (async () => {
  try {
    await initPopup();
  } catch (_err) {
    // Ignore popup init errors when closed mid-load.
  }
})();
