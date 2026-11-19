"use strict";

const DEFAULT_SERVER_URL = "http://127.0.0.1:8765";

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const statusVram = document.getElementById("statusVram");
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
const serverUrlInput = document.getElementById("serverUrlInput");
const saveServerUrlBtn = document.getElementById("saveServerUrlBtn");
const feedbackEl = document.getElementById("feedback");

let currentHostname = "";
let currentChapterId = "default";
let currentServerUrl = DEFAULT_SERVER_URL;
let currentGlossary = {};

function showFeedback(msg) {
  feedbackEl.textContent = msg;
  setTimeout(() => {
    if (feedbackEl.textContent === msg) feedbackEl.textContent = "";
  }, 3000);
}

async function refreshHealth() {
  const res = await chrome.runtime.sendMessage({ type: "GET_HEALTH" }).catch(() => null);
  if (!res || !res.ok || !res.data) {
    statusDot.className = "status-dot";
    statusText.textContent = "Offline — open Start Manga Translator";
    statusVram.textContent = "";
    return;
  }
  const h = res.data;
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
  await chrome.storage.local.set({ glossary: currentGlossary });
  renderGlossary();
  await chrome.runtime
    .sendMessage({
      type: "SET_GLOSSARY",
      chapterId: currentChapterId,
      entries: currentGlossary,
    })
    .catch(() => {});
}

async function initPopup() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab && tab.url) {
    try {
      const u = new URL(tab.url);
      if (u.protocol === "http:" || u.protocol === "https:") {
        currentHostname = u.hostname;
        currentChapterId = `${u.hostname}${u.pathname}`;
      }
    } catch (_err) {
      // Ignore malformed tab URLs.
    }
  }

  const stored = await chrome.storage.local.get([
    "enabledDomains",
    "langMode",
    "glossary",
    "serverUrl",
  ]);

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

  await refreshHealth();
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
  const res = await chrome.runtime
    .sendMessage({ type: "CLEAR_MEMORY", chapterId: currentChapterId })
    .catch(() => null);
  showFeedback(res?.ok ? "Chapter memory cleared" : "Server offline");
});

openTestBtn.addEventListener("click", () => {
  const base = (currentServerUrl || DEFAULT_SERVER_URL).replace(/\/+$/, "");
  chrome.tabs.create({ url: `${base}/test` });
});

shutdownBtn.addEventListener("click", async () => {
  const res = await chrome.runtime.sendMessage({ type: "SHUTDOWN_SERVER" }).catch(() => null);
  if (res?.ok) {
    statusDot.className = "status-dot";
    statusText.textContent = "Offline — VRAM Freed";
    statusVram.textContent = "";
    showFeedback("Local GPU engine stopped");
  } else {
    showFeedback("Server already offline");
  }
});

saveServerUrlBtn.addEventListener("click", async () => {
  const raw = serverUrlInput.value.trim().replace(/\/+$/, "");
  const clean = raw || DEFAULT_SERVER_URL;
  if (!/^https?:\/\/[^/\s]+/i.test(clean)) {
    showFeedback("Enter a valid http:// or https:// URL");
    return;
  }
  currentServerUrl = clean;
  serverUrlInput.value = clean;
  await chrome.storage.local.set({ serverUrl: clean });
  showFeedback("Server URL saved");
  await refreshHealth();
});

initPopup().catch(() => {});
