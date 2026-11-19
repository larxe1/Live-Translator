(function (root) {
  "use strict";

  const MAX_LOOKAHEAD = 3;
  const MAX_PAGE_CACHE = 50;

  function isDomainEnabled(hostname, enabledDomains) {
    if (!hostname || !enabledDomains) return false;
    if (Array.isArray(enabledDomains)) {
      return enabledDomains.includes(hostname);
    }
    if (typeof enabledDomains === "object") {
      return Boolean(enabledDomains[hostname]);
    }
    return false;
  }

  function getElementDimensions(el) {
    if (!el) return { width: 0, height: 0 };
    const rect = typeof el.getBoundingClientRect === "function" ? el.getBoundingClientRect() : null;
    const width =
      rect && rect.width > 0
        ? rect.width
        : Number(el.naturalWidth || el.width || 0);
    const height =
      rect && rect.height > 0
        ? rect.height
        : Number(el.naturalHeight || el.height || 0);
    return { width, height };
  }

  function isCandidateMangaElement(el) {
    if (!el || !el.tagName) return false;
    const tag = String(el.tagName).toUpperCase();
    if (tag !== "IMG" && tag !== "CANVAS") return false;
    if (el.closest && (el.closest(".mlt-overlay") || el.closest(".mlt-pill"))) {
      return false;
    }
    if (el.style && el.style.display === "none") return false;

    if (tag === "IMG") {
      if (el.complete === false) return false;
      const src = getElementSourceKey(el);
      if (!src) return false;
    }

    const { width, height } = getElementDimensions(el);
    if (width <= 0 || height <= 0) return false;

    const isStandardPage = width >= 300 && height >= 350;
    const isTallStrip = width >= 250 && height / width >= 1.8;
    return isStandardPage || isTallStrip;
  }

  function getElementSourceKey(el) {
    if (!el) return "";
    const tag = String(el.tagName || "").toUpperCase();
    if (tag === "CANVAS") {
      return `canvas:${el.dataset?.mltCanvasId || el.id || `${el.width}x${el.height}`}`;
    }
    const attrDataSrc =
      typeof el.getAttribute === "function" ? el.getAttribute("data-src") : "";
    const attrSrc = typeof el.getAttribute === "function" ? el.getAttribute("src") : "";
    return String(el.currentSrc || el.src || attrSrc || attrDataSrc || "").trim();
  }

  /**
   * Computes priority 0 (currently visible) and priority 1 (next 2-3 pages
   * below in DOM order) targets, deferring distant pages until scrolled closer.
   */
  function computeLookaheadTargets(candidates, visibleSet, maxLookahead = MAX_LOOKAHEAD) {
    if (!Array.isArray(candidates) || candidates.length === 0) {
      return { immediate: [], lookahead: [] };
    }
    const immediate = [];
    let lastVisibleIdx = -1;

    for (let i = 0; i < candidates.length; i += 1) {
      const el = candidates[i];
      if (visibleSet && visibleSet.has(el)) {
        immediate.push(el);
        lastVisibleIdx = i;
      }
    }

    if (lastVisibleIdx === -1) {
      return { immediate: [], lookahead: [] };
    }

    const lookahead = [];
    for (let i = lastVisibleIdx + 1; i < candidates.length && lookahead.length < maxLookahead; i += 1) {
      const el = candidates[i];
      if (!visibleSet.has(el)) {
        lookahead.push(el);
      }
    }

    return { immediate, lookahead };
  }

  function isContextInvalidatedError(err) {
    if (typeof chrome !== "undefined" && !chrome?.runtime?.id) {
      return true;
    }
    const msg = String(err?.message || err || "");
    return /extension context invalidated/i.test(msg);
  }

  function stableGlossaryKey(glossary) {
    if (!glossary || typeof glossary !== "object") return "{}";
    const keys = Object.keys(glossary).sort();
    const out = {};
    for (const k of keys) out[k] = glossary[k];
    return JSON.stringify(out);
  }

  async function blobUrlToBase64(blobUrl, fetchImpl = fetch) {
    const res = await fetchImpl(blobUrl);
    if (!res.ok) throw new Error(`Blob fetch failed (${res.status})`);
    const buffer = await res.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x4000;
    const chunks = [];
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const sub = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
      chunks.push(String.fromCharCode.apply(null, sub));
    }
    return btoa(chunks.join(""));
  }

  function createPageController(deps) {
    const doc = deps.document || document;
    const win = deps.window || window;
    const hostname = deps.hostname ?? win.location?.hostname ?? "";
    const chapterId = deps.chapterId ?? `${hostname}${win.location?.pathname || "/"}`;
    const storage = deps.storage;
    const sendMessage = deps.sendMessage;
    const overlayApi = deps.overlayApi || root.MangaOverlay;
    const MutationObserverCtor =
      deps.MutationObserver !== undefined ? deps.MutationObserver : win.MutationObserver;
    const IntersectionObserverCtor =
      deps.IntersectionObserver !== undefined
        ? deps.IntersectionObserver
        : win.IntersectionObserver;

    let enabled = false;
    let contextInvalidated = false;
    let langMode = "auto";
    let glossary = {};
    let reqCounter = 0;
    let mutationObserver = null;
    let intersectionObserver = null;
    let pillEl = null;
    let pillDot = null;
    let pillStatusWrap = null;
    let pillStatusText = null;
    let pillLangSelect = null;
    let pillPeekBtn = null;
    let isPeeking = false;
    let collapsed = false;

    const candidates = [];
    const candidateSet = new Set();
    const visibleSet = new Set();
    const generations = new WeakMap();
    const lastSourceByEl = new WeakMap();
    const inFlightByEl = new Map();
    const completedByEl = new WeakMap();
    const loadListeners = new WeakMap();
    const pageCache = new Map();

    function getCacheKey(srcKey) {
      if (!srcKey || srcKey.startsWith("data:") || srcKey.startsWith("canvas:")) {
        return null;
      }
      return `${srcKey}::${langMode}::${stableGlossaryKey(glossary)}`;
    }

    function setCachedResponse(cacheKey, response) {
      if (!cacheKey) return;
      if (pageCache.has(cacheKey)) {
        pageCache.delete(cacheKey);
      }
      pageCache.set(cacheKey, response);
      while (pageCache.size > MAX_PAGE_CACHE) {
        const oldest = pageCache.keys().next().value;
        pageCache.delete(oldest);
      }
    }

    function updatePillStatus(state, label) {
      if (!pillDot || !pillStatusText) return;
      if (pillDot.dataset) pillDot.dataset.state = state;
      pillStatusText.textContent = label;
    }

    function applyUpdatedPillUI() {
      if (!pillEl) return;
      if (pillEl.dataset) pillEl.dataset.updated = "true";
      if (pillDot?.style) pillDot.style.background = "#6366f1";
      if (pillStatusWrap) {
        pillStatusWrap.title = "Extension updated — Click to refresh this page";
        if (pillStatusWrap.style) pillStatusWrap.style.cursor = "pointer";
      }
      if (pillStatusText?.style) {
        pillStatusText.style.textDecoration = "underline";
        pillStatusText.style.cursor = "pointer";
      }
      updatePillStatus("updated", "Updated — Click to refresh");
    }

    function handleContextInvalidated() {
      if (contextInvalidated) return;
      contextInvalidated = true;
      stopObservers();
      inFlightByEl.clear();
      mountPill();
      applyUpdatedPillUI();
    }

    function refreshStatusIndicator() {
      if (!enabled || contextInvalidated) return;
      if (inFlightByEl.size > 0) {
        updatePillStatus("busy", `Translating (${inFlightByEl.size})`);
      } else {
        updatePillStatus("ready", "Ready");
      }
    }

    function mountPill() {
      if (pillEl || !doc || !doc.createElement) return;
      const host = doc.body || doc.documentElement;
      if (!host || typeof host.appendChild !== "function") return;

      const pill = doc.createElement("div");
      pill.className = "mlt-pill";
      if (pill.dataset) pill.dataset.mltPill = "true";

      const header = doc.createElement("div");
      header.className = "mlt-pill-header";

      const statusWrap = doc.createElement("div");
      statusWrap.className = "mlt-pill-status-wrap";
      const dot = doc.createElement("span");
      dot.className = "mlt-pill-dot";
      if (dot.dataset) dot.dataset.state = "ready";
      const statusSpan = doc.createElement("span");
      statusSpan.textContent = "Ready";
      statusWrap.appendChild(dot);
      statusWrap.appendChild(statusSpan);
      statusWrap.addEventListener("click", () => {
        if (contextInvalidated && typeof win?.location?.reload === "function") {
          win.location.reload();
        }
      });

      const collapseBtn = doc.createElement("button");
      collapseBtn.type = "button";
      collapseBtn.className = "mlt-pill-btn";
      collapseBtn.textContent = "—";
      collapseBtn.title = "Collapse / Expand";
      collapseBtn.addEventListener("click", () => {
        collapsed = !collapsed;
        pill.classList.toggle("mlt-pill-collapsed", collapsed);
        collapseBtn.textContent = collapsed ? "+" : "—";
      });

      pill.addEventListener("click", (evt) => {
        if (contextInvalidated && evt?.target !== collapseBtn && typeof win?.location?.reload === "function") {
          win.location.reload();
        }
      });

      header.appendChild(statusWrap);
      header.appendChild(collapseBtn);
      pill.appendChild(header);

      const controls = doc.createElement("div");
      controls.className = "mlt-pill-controls";

      const peekBtn = doc.createElement("button");
      peekBtn.type = "button";
      peekBtn.className = "mlt-pill-btn";
      peekBtn.textContent = "Peek (Alt)";
      peekBtn.addEventListener("click", () => {
        isPeeking = !isPeeking;
        if (peekBtn.dataset) peekBtn.dataset.active = isPeeking ? "true" : "false";
        overlayApi?.setPeekOriginal(isPeeking, doc);
      });

      const langSelect = doc.createElement("select");
      langSelect.className = "mlt-pill-select";
      for (const [val, text] of [
        ["auto", "Auto"],
        ["ja", "JP"],
        ["ko", "KO"],
      ]) {
        const opt = doc.createElement("option");
        opt.value = val;
        opt.textContent = text;
        langSelect.appendChild(opt);
      }
      langSelect.value = langMode;
      langSelect.addEventListener("change", async () => {
        if (contextInvalidated) return;
        langMode = langSelect.value;
        if (storage?.set) {
          try {
            await storage.set({ langMode });
          } catch (err) {
            if (isContextInvalidatedError(err)) {
              handleContextInvalidated();
              return;
            }
          }
        }
        retranslateAllActive();
      });

      const offBtn = doc.createElement("button");
      offBtn.type = "button";
      offBtn.className = "mlt-pill-btn mlt-pill-btn-off";
      offBtn.textContent = "Off";
      offBtn.addEventListener("click", async () => {
        if (contextInvalidated) return;
        if (storage?.get && storage?.set) {
          try {
            const cur = await storage.get(["enabledDomains"]);
            const next = { ...(cur?.enabledDomains || {}), [hostname]: false };
            await storage.set({ enabledDomains: next });
          } catch (err) {
            if (isContextInvalidatedError(err)) {
              handleContextInvalidated();
              return;
            }
          }
        }
        disable();
      });

      controls.appendChild(peekBtn);
      controls.appendChild(langSelect);
      controls.appendChild(offBtn);
      pill.appendChild(controls);

      host.appendChild(pill);
      pillEl = pill;
      pillDot = dot;
      pillStatusWrap = statusWrap;
      pillStatusText = statusSpan;
      pillLangSelect = langSelect;
      pillPeekBtn = peekBtn;
      if (contextInvalidated) {
        applyUpdatedPillUI();
      }
    }

    function unmountPill() {
      if (pillEl && pillEl.parentNode) {
        pillEl.parentNode.removeChild(pillEl);
      }
      pillEl = null;
      pillDot = null;
      pillStatusWrap = null;
      pillStatusText = null;
      pillLangSelect = null;
      pillPeekBtn = null;
      isPeeking = false;
      overlayApi?.setPeekOriginal(false, doc);
    }

    function bumpGeneration(el) {
      const next = (generations.get(el) || 0) + 1;
      generations.set(el, next);
      return next;
    }

    async function sendCancelTranslate(requestId) {
      if (contextInvalidated || typeof sendMessage !== "function") return;
      try {
        await sendMessage({ type: "CANCEL_TRANSLATE", requestId });
      } catch (err) {
        if (isContextInvalidatedError(err)) {
          handleContextInvalidated();
        }
      }
    }

    function cancelElementInFlight(el) {
      const active = inFlightByEl.get(el);
      if (!active) return;
      inFlightByEl.delete(el);
      if (!contextInvalidated && typeof sendMessage === "function") {
        void sendCancelTranslate(active.requestId);
      }
    }

    function registerElement(el) {
      if (!el || !el.tagName) return;
      const tag = String(el.tagName).toUpperCase();
      if (tag !== "IMG" && tag !== "CANVAS") return;

      if (tag === "IMG" && el.complete === false && !loadListeners.has(el)) {
        const onLoad = () => {
          if (!enabled) return;
          if (isCandidateMangaElement(el)) {
            addCandidate(el);
            refreshVisibleAndLookahead();
          }
        };
        loadListeners.set(el, onLoad);
        if (typeof el.addEventListener === "function") {
          el.addEventListener("load", onLoad, { once: true });
        }
      }

      if (isCandidateMangaElement(el)) {
        addCandidate(el);
      }
    }

    function addCandidate(el) {
      if (candidateSet.has(el)) return;
      candidateSet.add(el);
      candidates.push(el);
      if (!generations.has(el)) {
        generations.set(el, 1);
      }
      lastSourceByEl.set(el, getElementSourceKey(el));
      if (intersectionObserver && typeof intersectionObserver.observe === "function") {
        intersectionObserver.observe(el);
      }
    }

    function removeCandidate(el) {
      if (!el) return;
      cancelElementInFlight(el);
      overlayApi?.clearOverlay(el);
      visibleSet.delete(el);
      completedByEl.delete(el);
      if (candidateSet.has(el)) {
        candidateSet.delete(el);
        const idx = candidates.indexOf(el);
        if (idx !== -1) candidates.splice(idx, 1);
      }
      if (intersectionObserver && typeof intersectionObserver.unobserve === "function") {
        intersectionObserver.unobserve(el);
      }
    }

    function scanDocumentCandidates() {
      if (!doc || typeof doc.querySelectorAll !== "function") return;
      const els = doc.querySelectorAll("img, canvas");
      for (const el of els) {
        registerElement(el);
      }
    }

    function handleSourceMutation(el) {
      if (!el) return;
      const nextSrc = getElementSourceKey(el);
      const prevSrc = lastSourceByEl.get(el);
      if (nextSrc === prevSrc) return;

      lastSourceByEl.set(el, nextSrc);
      bumpGeneration(el);
      overlayApi?.clearOverlay(el);
      completedByEl.delete(el);
      cancelElementInFlight(el);

      if (isCandidateMangaElement(el)) {
        addCandidate(el);
        refreshVisibleAndLookahead();
      }
    }

    async function translateElement(el, priority = 0, force = false) {
      if (!enabled && !force) return null;
      if (!el) return null;

      const srcKey = getElementSourceKey(el);
      const gen = generations.get(el) || 1;
      const tag = String(el.tagName || "").toUpperCase();

      if (!force) {
        const done = completedByEl.get(el);
        if (done && done.gen === gen && done.srcKey === srcKey) {
          return done.response;
        }
        const active = inFlightByEl.get(el);
        if (active && active.gen === gen && active.srcKey === srcKey) {
          return null;
        }
      }

      const cacheKey = !force ? getCacheKey(srcKey) : null;
      if (cacheKey && pageCache.has(cacheKey)) {
        const cachedResp = pageCache.get(cacheKey);
        completedByEl.set(el, { gen, srcKey, response: cachedResp });
        overlayApi?.renderOverlay(el, cachedResp, { document: doc, window: win });
        return cachedResp;
      }

      let imageUrl = "";
      let imageBase64 = "";

      if (tag === "CANVAS") {
        try {
          const dataUrl = el.toDataURL("image/png");
          imageBase64 = dataUrl.split(",")[1] || "";
        } catch (_err) {
          return null; // Tainted DRM canvas; skip cleanly.
        }
      } else {
        if (srcKey.startsWith("data:")) {
          imageBase64 = srcKey.split(",")[1] || "";
        } else if (srcKey.startsWith("blob:")) {
          try {
            imageBase64 = await blobUrlToBase64(srcKey, deps.fetch || win.fetch);
          } catch (_err) {
            return null;
          }
        } else {
          imageUrl = srcKey;
        }
      }

      if (!imageUrl && !imageBase64) return null;

      if (contextInvalidated) return null;
      if (typeof chrome !== "undefined" && !chrome?.runtime?.id) {
        handleContextInvalidated();
        return null;
      }

      const requestId = `mlt-${++reqCounter}-${Date.now()}`;
      inFlightByEl.set(el, { gen, srcKey, requestId });
      refreshStatusIndicator();

      try {
        const result = await sendMessage({
          type: "TRANSLATE_IMAGE",
          imageUrl,
          imageBase64,
          refererUrl: win.location?.href || "",
          lang: langMode,
          chapterId,
          priority,
          glossary,
          requestId,
        });

        // Stale-response guard: discard if disabled, invalidated, generation changed, or src mutated.
        if (contextInvalidated) return null;
        if (!enabled && !force) return null;
        if ((generations.get(el) || 1) !== gen) return null;
        if (getElementSourceKey(el) !== srcKey) return null;

        if (!result || result.ok === false || !result.data) {
          if (result && !result.aborted) {
            if (isContextInvalidatedError(result.error)) {
              handleContextInvalidated();
            } else {
              updatePillStatus("offline", "Server Offline");
            }
          }
          return null;
        }

        const data = result.data;
        completedByEl.set(el, { gen, srcKey, response: data });
        setCachedResponse(cacheKey, data);
        overlayApi?.renderOverlay(el, data, { document: doc, window: win });
        return data;
      } catch (err) {
        if (isContextInvalidatedError(err)) {
          handleContextInvalidated();
          return null;
        }
        updatePillStatus("offline", "Server Offline");
        return null;
      } finally {
        const cur = inFlightByEl.get(el);
        if (cur && cur.requestId === requestId) {
          inFlightByEl.delete(el);
        }
        if (!contextInvalidated) {
          refreshStatusIndicator();
        }
      }
    }

    function refreshVisibleAndLookahead() {
      if (!enabled || contextInvalidated) return;

      // Sort candidates in DOM order if compareDocumentPosition is available.
      if (candidates.length > 1 && typeof candidates[0]?.compareDocumentPosition === "function") {
        candidates.sort((a, b) => {
          if (a === b) return 0;
          const pos = a.compareDocumentPosition(b);
          return pos & 4 ? -1 : 1;
        });
      }

      const { immediate, lookahead } = computeLookaheadTargets(
        candidates,
        visibleSet,
        MAX_LOOKAHEAD
      );
      const activeTargets = new Set([...immediate, ...lookahead]);

      // Cancel off-screen requests scrolled far away.
      for (const [el] of inFlightByEl.entries()) {
        if (!activeTargets.has(el)) {
          cancelElementInFlight(el);
        }
      }

      const promises = [];
      for (const el of immediate) {
        promises.push(translateElement(el, 0));
      }
      for (const el of lookahead) {
        promises.push(translateElement(el, 1));
      }
      return Promise.all(promises);
    }

    function retranslateAllActive() {
      if (!enabled || contextInvalidated) return;
      for (const el of candidates) {
        bumpGeneration(el);
        completedByEl.delete(el);
        overlayApi?.clearOverlay(el);
        cancelElementInFlight(el);
      }
      refreshVisibleAndLookahead();
    }

    function setVisibility(el, isVisible) {
      if (isVisible) {
        visibleSet.add(el);
      } else {
        visibleSet.delete(el);
      }
    }

    function startObservers() {
      if (typeof IntersectionObserverCtor === "function" && !intersectionObserver) {
        intersectionObserver = new IntersectionObserverCtor(
          (entries) => {
            if (!enabled) return;
            for (const entry of entries) {
              setVisibility(entry.target, Boolean(entry.isIntersecting));
            }
            refreshVisibleAndLookahead();
          },
          { threshold: 0.05 }
        );
        for (const el of candidates) {
          intersectionObserver.observe(el);
        }
      }

      if (typeof MutationObserverCtor === "function" && !mutationObserver) {
        const rootNode = doc.body || doc.documentElement;
        if (rootNode) {
          mutationObserver = new MutationObserverCtor((mutations) => {
            if (!enabled) return;
            let added = false;
            for (const m of mutations) {
              if (m.type === "attributes" && m.target) {
                handleSourceMutation(m.target);
              } else if (m.type === "childList") {
                for (const node of m.removedNodes || []) {
                  if (candidateSet.has(node)) {
                    removeCandidate(node);
                  } else if (node?.querySelectorAll) {
                    for (const child of node.querySelectorAll("img, canvas")) {
                      removeCandidate(child);
                    }
                  }
                }
                for (const node of m.addedNodes || []) {
                  if (node?.tagName) {
                    const tag = String(node.tagName).toUpperCase();
                    if (tag === "IMG" || tag === "CANVAS") {
                      registerElement(node);
                      added = true;
                    } else if (node.querySelectorAll) {
                      for (const child of node.querySelectorAll("img, canvas")) {
                        registerElement(child);
                        added = true;
                      }
                    }
                  }
                }
              }
            }
            if (added) {
              refreshVisibleAndLookahead();
            }
          });
          mutationObserver.observe(rootNode, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ["src", "srcset", "data-src"],
          });
        }
      }
    }

    function stopObservers() {
      if (mutationObserver) {
        mutationObserver.disconnect();
        mutationObserver = null;
      }
      if (intersectionObserver) {
        intersectionObserver.disconnect();
        intersectionObserver = null;
      }
    }

    function enable() {
      if (enabled || contextInvalidated) return;
      enabled = true;
      mountPill();
      scanDocumentCandidates();
      startObservers();
      refreshVisibleAndLookahead();
    }

    function disable() {
      enabled = false;
      stopObservers();
      for (const el of [...candidates]) {
        removeCandidate(el);
      }
      for (const [el] of [...inFlightByEl.entries()]) {
        cancelElementInFlight(el);
      }
      visibleSet.clear();
      unmountPill();
    }

    async function init() {
      if (!storage || typeof storage.get !== "function") {
        return { enabled: false };
      }
      const data = await storage.get(["enabledDomains", "langMode", "glossary"]);
      langMode = data?.langMode || "auto";
      glossary = data?.glossary || {};
      if (isDomainEnabled(hostname, data?.enabledDomains)) {
        enable();
        return { enabled: true };
      }
      return { enabled: false };
    }

    function handleStorageChanges(changes) {
      if (!changes || contextInvalidated) return;
      if (changes.langMode) {
        langMode = changes.langMode.newValue || "auto";
        if (pillLangSelect) pillLangSelect.value = langMode;
        if (enabled) retranslateAllActive();
      }
      if (changes.glossary) {
        glossary = changes.glossary.newValue || {};
        if (enabled) retranslateAllActive();
      }
      if (changes.enabledDomains) {
        const shouldEnable = isDomainEnabled(hostname, changes.enabledDomains.newValue);
        if (shouldEnable && !enabled) {
          enable();
        } else if (!shouldEnable && enabled) {
          disable();
        }
      }
    }

    function handleManualTranslate(srcUrl) {
      if (contextInvalidated || !doc || typeof doc.querySelectorAll !== "function") return null;
      const imgs = doc.querySelectorAll("img");
      for (const img of imgs) {
        const key = getElementSourceKey(img);
        if (srcUrl && (key === srcUrl || img.src === srcUrl || img.currentSrc === srcUrl)) {
          return translateElement(img, 0, true);
        }
      }
      return null;
    }

    return {
      init,
      enable,
      disable,
      isEnabled: () => enabled,
      isContextInvalidated: () => contextInvalidated,
      handleContextInvalidated,
      registerElement,
      getCandidates: () => [...candidates],
      setVisibility,
      handleSourceMutation,
      translateElement,
      refreshVisibleAndLookahead,
      handleStorageChanges,
      handleManualTranslate,
      getCacheSize: () => pageCache.size,
    };
  }

  function bootstrapExtensionContent() {
    if (
      typeof window === "undefined" ||
      typeof document === "undefined" ||
      typeof chrome === "undefined" ||
      !chrome.runtime ||
      !chrome.storage?.local
    ) {
      return null;
    }

    const controller = createPageController({
      document,
      window,
      storage: chrome.storage.local,
      sendMessage: async (msg) => {
        if (!chrome?.runtime?.id) {
          throw new Error("Extension context invalidated");
        }
        return await chrome.runtime.sendMessage(msg);
      },
      overlayApi: root.MangaOverlay,
    });

    void (async () => {
      try {
        await controller.init();
      } catch (err) {
        if (isContextInvalidatedError(err)) {
          controller.handleContextInvalidated();
        }
      }
    })();

    try {
      chrome.storage.onChanged.addListener((changes, areaName) => {
        if (!chrome?.runtime?.id) {
          controller.handleContextInvalidated();
          return;
        }
        if (areaName === "local") {
          controller.handleStorageChanges(changes);
        }
      });

      chrome.runtime.onMessage.addListener((msg) => {
        if (!chrome?.runtime?.id) {
          controller.handleContextInvalidated();
          return;
        }
        if (msg?.type === "MANUAL_TRANSLATE_IMAGE") {
          void controller.handleManualTranslate(msg.srcUrl);
        }
      });
    } catch (err) {
      if (isContextInvalidatedError(err)) {
        controller.handleContextInvalidated();
      }
    }

    return controller;
  }

  if (typeof module === "undefined") {
    bootstrapExtensionContent();
  }

  const exportsApi = {
    MAX_LOOKAHEAD,
    MAX_PAGE_CACHE,
    isDomainEnabled,
    isCandidateMangaElement,
    getElementSourceKey,
    computeLookaheadTargets,
    isContextInvalidatedError,
    createPageController,
  };

  if (root) {
    root.MangaContent = exportsApi;
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = exportsApi;
  }
})(typeof window !== "undefined" ? window : globalThis);
