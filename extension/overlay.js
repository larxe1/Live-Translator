(function (root) {
  "use strict";

  const MIN_FONT_PX = 11;
  const MAX_FONT_PX = 26;
  const MAX_WIDTH_EXPANSION = 1.25;
  const MAX_HEIGHT_EXPANSION = 1.18;
  const PAGE_MARGIN_PX = 2;

  const overlaysByTarget = new WeakMap();
  let altListenersRegistered = false;
  let sharedMeasureCtx = null;

  function getMeasureContext(doc) {
    if (sharedMeasureCtx) return sharedMeasureCtx;
    const d = doc || (typeof document !== "undefined" ? document : null);
    if (!d || typeof d.createElement !== "function") return null;
    try {
      const canvas = d.createElement("canvas");
      if (canvas && typeof canvas.getContext === "function") {
        sharedMeasureCtx = canvas.getContext("2d");
        return sharedMeasureCtx;
      }
    } catch (_err) {
      // Ignore environments without 2D canvas support.
    }
    return null;
  }

  function measureWordWidth(word, fontSizePx, ctx) {
    if (ctx && typeof ctx.measureText === "function") {
      ctx.font = `700 ${fontSizePx}px "Comic Neue", "Anime Ace", "Segoe UI", sans-serif`;
      const metrics = ctx.measureText(word);
      if (metrics && typeof metrics.width === "number" && metrics.width > 0) {
        return metrics.width;
      }
    }
    // Conservative bold comic font width estimate for headless/Node environments.
    let width = 0;
    for (let i = 0; i < word.length; i += 1) {
      const ch = word[i];
      if ("ilI1!.,':;|".includes(ch)) {
        width += fontSizePx * 0.34;
      } else if ("mwMW@#%&".includes(ch)) {
        width += fontSizePx * 0.82;
      } else if (ch >= "A" && ch <= "Z") {
        width += fontSizePx * 0.66;
      } else {
        width += fontSizePx * 0.56;
      }
    }
    return width;
  }

  function defaultFitsBox(text, fontSizePx, innerW, innerH, ctx) {
    const trimmed = String(text || "").trim();
    if (!trimmed) return true;
    if (innerW <= 0 || innerH <= 0) return false;

    const spaceW = measureWordWidth(" ", fontSizePx, ctx);
    const lineHeight = fontSizePx * 1.15;
    const paragraphs = trimmed.split(/\r?\n/);
    let totalLines = 0;

    for (const para of paragraphs) {
      const words = para.trim().split(/\s+/).filter(Boolean);
      if (words.length === 0) {
        totalLines += 1;
        continue;
      }
      let currentLineW = 0;
      for (const word of words) {
        const wordW = measureWordWidth(word, fontSizePx, ctx);
        if (wordW > innerW) {
          return false;
        }
        if (currentLineW === 0) {
          currentLineW = wordW;
          totalLines += 1;
        } else if (currentLineW + spaceW + wordW <= innerW) {
          currentLineW += spaceW + wordW;
        } else {
          currentLineW = wordW;
          totalLines += 1;
        }
      }
    }

    return totalLines * lineHeight <= innerH;
  }

  function evaluateFit(text, fontSizePx, innerW, innerH, measureFn, ctx) {
    if (typeof measureFn === "function") {
      const res = measureFn(text, fontSizePx, innerW, innerH);
      if (typeof res === "boolean") return res;
      if (typeof res === "number") return res <= innerW;
      if (res && typeof res === "object") {
        if (typeof res.fits === "boolean") return res.fits;
        if (typeof res.width === "number" && typeof res.height === "number") {
          return res.width <= innerW && res.height <= innerH;
        }
      }
      return Boolean(res);
    }
    return defaultFitsBox(text, fontSizePx, innerW, innerH, ctx);
  }

  function findBestFontSize(text, boxW, boxH, measureFn, ctx) {
    const padX = Math.min(8, Math.max(3, boxW * 0.06));
    const padY = Math.min(8, Math.max(3, boxH * 0.06));
    const innerW = boxW - padX * 2;
    const innerH = boxH - padY * 2;

    if (!evaluateFit(text, MIN_FONT_PX, innerW, innerH, measureFn, ctx)) {
      return null;
    }
    if (evaluateFit(text, MAX_FONT_PX, innerW, innerH, measureFn, ctx)) {
      return MAX_FONT_PX;
    }

    let lo = MIN_FONT_PX;
    let hi = MAX_FONT_PX;
    let best = MIN_FONT_PX;
    while (lo <= hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (evaluateFit(text, mid, innerW, innerH, measureFn, ctx)) {
        best = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return best;
  }

  function normalizeBox(safeBox, imgW, imgH) {
    const rawX1 = Number(safeBox?.x1 ?? 0);
    const rawY1 = Number(safeBox?.y1 ?? 0);
    const rawX2 = Number(safeBox?.x2 ?? rawX1 + 20);
    const rawY2 = Number(safeBox?.y2 ?? rawY1 + 20);
    const x1 = Math.max(0, Math.min(imgW - 8, rawX1));
    const y1 = Math.max(0, Math.min(imgH - 8, rawY1));
    const x2 = Math.min(imgW, Math.max(x1 + 8, rawX2));
    const y2 = Math.min(imgH, Math.max(y1 + 8, rawY2));
    return { x1, y1, x2, y2 };
  }

  function computeControlledExpandedBox(baseBox, imgW, imgH) {
    const safeW = baseBox.x2 - baseBox.x1;
    const safeH = baseBox.y2 - baseBox.y1;
    const maxW = Math.min(safeW * MAX_WIDTH_EXPANSION, Math.max(8, imgW - PAGE_MARGIN_PX * 2));
    const maxH = Math.min(safeH * MAX_HEIGHT_EXPANSION, Math.max(8, imgH - PAGE_MARGIN_PX * 2));
    const cx = (baseBox.x1 + baseBox.x2) / 2;
    const cy = (baseBox.y1 + baseBox.y2) / 2;

    let ex1 = cx - maxW / 2;
    let ex2 = cx + maxW / 2;
    if (ex1 < PAGE_MARGIN_PX) {
      ex2 = Math.min(imgW - PAGE_MARGIN_PX, ex2 + (PAGE_MARGIN_PX - ex1));
      ex1 = PAGE_MARGIN_PX;
    }
    if (ex2 > imgW - PAGE_MARGIN_PX) {
      ex1 = Math.max(PAGE_MARGIN_PX, ex1 - (ex2 - (imgW - PAGE_MARGIN_PX)));
      ex2 = imgW - PAGE_MARGIN_PX;
    }

    let ey1 = cy - maxH / 2;
    let ey2 = cy + maxH / 2;
    if (ey1 < PAGE_MARGIN_PX) {
      ey2 = Math.min(imgH - PAGE_MARGIN_PX, ey2 + (PAGE_MARGIN_PX - ey1));
      ey1 = PAGE_MARGIN_PX;
    }
    if (ey2 > imgH - PAGE_MARGIN_PX) {
      ey1 = Math.max(PAGE_MARGIN_PX, ey1 - (ey2 - (imgH - PAGE_MARGIN_PX)));
      ey2 = imgH - PAGE_MARGIN_PX;
    }

    const x1 = Math.ceil(ex1);
    let x2 = Math.floor(ex2);
    if (x2 - x1 > safeW * MAX_WIDTH_EXPANSION) {
      x2 = Math.floor(x1 + safeW * MAX_WIDTH_EXPANSION);
    }

    const y1 = Math.ceil(ey1);
    let y2 = Math.floor(ey2);
    if (y2 - y1 > safeH * MAX_HEIGHT_EXPANSION) {
      y2 = Math.floor(y1 + safeH * MAX_HEIGHT_EXPANSION);
    }

    return {
      x1,
      y1,
      x2: Math.max(x1 + 8, x2),
      y2: Math.max(y1 + 8, y2),
    };
  }

  /**
   * Computes optimal font size and box bounds inside `safeBox`, allowing
   * controlled modest expansion (+25% width / +18% height) if text cannot
   * fit cleanly at >= 11px.
   */
  function computeFittedBox(text, safeBox, imgSize, measureFn) {
    const imgW = Math.max(16, Number(imgSize?.w ?? imgSize?.width ?? 1000));
    const imgH = Math.max(16, Number(imgSize?.h ?? imgSize?.height ?? 1000));
    const base = normalizeBox(safeBox, imgW, imgH);
    const trimmed = String(text || "").trim();

    if (!trimmed) {
      return { box: base, fontSizePx: 14, expanded: false };
    }

    const ctx = measureFn ? null : getMeasureContext();
    const safeW = base.x2 - base.x1;
    const safeH = base.y2 - base.y1;
    const directFont = findBestFontSize(trimmed, safeW, safeH, measureFn, ctx);
    if (directFont !== null) {
      return { box: base, fontSizePx: directFont, expanded: false };
    }

    const expandedBox = computeControlledExpandedBox(base, imgW, imgH);
    const expW = expandedBox.x2 - expandedBox.x1;
    const expH = expandedBox.y2 - expandedBox.y1;
    const expFont = findBestFontSize(trimmed, expW, expH, measureFn, ctx);

    return {
      box: expandedBox,
      fontSizePx: expFont !== null ? expFont : MIN_FONT_PX,
      expanded: true,
    };
  }

  function pct(value, total) {
    return `${((value / Math.max(1, total)) * 100).toFixed(4)}%`;
  }

  function setPeekOriginal(enabled, doc) {
    const d = doc || (typeof document !== "undefined" ? document : null);
    if (d && d.documentElement && d.documentElement.classList) {
      d.documentElement.classList.toggle("mlt-peek-original", Boolean(enabled));
    }
  }

  function initAltPeekListeners(win, doc) {
    const w = win || (typeof window !== "undefined" ? window : null);
    const d = doc || (typeof document !== "undefined" ? document : null);
    if (!w || !d || typeof w.addEventListener !== "function") return;
    if (altListenersRegistered && !win && !doc) return;
    if (!win && !doc) altListenersRegistered = true;

    const onKeyDown = (e) => {
      if (e && e.key === "Alt") {
        setPeekOriginal(true, d);
      }
    };
    const onKeyUp = (e) => {
      if (e && e.key === "Alt") {
        setPeekOriginal(false, d);
      }
    };
    const onBlur = () => setPeekOriginal(false, d);
    const onVisibilityChange = () => {
      if (d.hidden) setPeekOriginal(false, d);
    };

    w.addEventListener("keydown", onKeyDown, true);
    w.addEventListener("keyup", onKeyUp, true);
    w.addEventListener("blur", onBlur);
    if (typeof d.addEventListener === "function") {
      d.addEventListener("visibilitychange", onVisibilityChange);
    }

    return () => {
      w.removeEventListener("keydown", onKeyDown, true);
      w.removeEventListener("keyup", onKeyUp, true);
      w.removeEventListener("blur", onBlur);
      if (typeof d.removeEventListener === "function") {
        d.removeEventListener("visibilitychange", onVisibilityChange);
      }
    };
  }

  function clearOverlay(targetEl) {
    if (!targetEl) return;
    const entry = overlaysByTarget.get(targetEl);
    if (!entry) return;
    overlaysByTarget.delete(targetEl);

    if (entry.resizeObserver && typeof entry.resizeObserver.disconnect === "function") {
      entry.resizeObserver.disconnect();
    }
    if (entry.win && typeof entry.win.removeEventListener === "function") {
      entry.win.removeEventListener("resize", entry.sync);
    }
    if (entry.overlayEl && entry.overlayEl.parentNode) {
      entry.overlayEl.parentNode.removeChild(entry.overlayEl);
    }
  }

  function syncOverlayPosition(targetEl, overlayEl, imgW, bubbleEls, win, doc) {
    const hasRectFn = typeof targetEl.getBoundingClientRect === "function";
    const rect = hasRectFn ? targetEl.getBoundingClientRect() : null;
    const hasRealLayout = Boolean(rect && rect.width > 0 && rect.height > 0);

    if (hasRealLayout) {
      overlayEl.style.display = "block";
      overlayEl.style.width = `${rect.width}px`;
      overlayEl.style.height = `${rect.height}px`;

      const op = overlayEl.offsetParent;
      const getStyle = win && typeof win.getComputedStyle === "function" ? win.getComputedStyle : null;
      const opPosition = op && getStyle ? getStyle(op).position : "static";

      if (!op || op === doc.documentElement || opPosition === "static") {
        const docEl = doc.documentElement;
        const docRect =
          docEl && typeof docEl.getBoundingClientRect === "function"
            ? docEl.getBoundingClientRect()
            : { left: 0, top: 0 };
        overlayEl.style.left = `${rect.left - docRect.left}px`;
        overlayEl.style.top = `${rect.top - docRect.top}px`;
      } else {
        const opRect = op.getBoundingClientRect();
        overlayEl.style.left = `${rect.left - opRect.left + (op.scrollLeft || 0) - (op.clientLeft || 0)}px`;
        overlayEl.style.top = `${rect.top - opRect.top + (op.scrollTop || 0) - (op.clientTop || 0)}px`;
      }

      const scale = rect.width / Math.max(1, imgW);
      for (const bEl of bubbleEls) {
        const baseFont = Number(bEl.dataset.baseFontPx || 13);
        bEl.style.fontSize = `${Math.max(9, baseFont * scale).toFixed(2)}px`;
      }
    } else if (typeof ResizeObserver !== "undefined" && hasRectFn) {
      overlayEl.style.display = "none";
    } else {
      const fallbackW = Number(targetEl.width || targetEl.naturalWidth || imgW);
      const fallbackH = Number(targetEl.height || targetEl.naturalHeight || 1000);
      overlayEl.style.display = "block";
      overlayEl.style.left = "0px";
      overlayEl.style.top = "0px";
      overlayEl.style.width = `${fallbackW}px`;
      overlayEl.style.height = `${fallbackH}px`;
      const scale = fallbackW / Math.max(1, imgW);
      for (const bEl of bubbleEls) {
        const baseFont = Number(bEl.dataset.baseFontPx || 13);
        bEl.style.fontSize = `${Math.max(9, baseFont * scale).toFixed(2)}px`;
      }
    }
  }

  /**
   * Renders translated dialogue bubbles aligned pixel-accurate over `targetEl`.
   */
  function renderOverlay(targetEl, translateResponse, options = {}) {
    clearOverlay(targetEl);

    const doc = options.document || (targetEl && targetEl.ownerDocument) || (typeof document !== "undefined" ? document : null);
    const win = options.window || (doc && doc.defaultView) || (typeof window !== "undefined" ? window : null);

    if (!targetEl || !doc || !translateResponse || !Array.isArray(translateResponse.bubbles)) {
      return { element: null, sync() {}, destroy() {} };
    }

    initAltPeekListeners(win, doc);

    const imgW = Math.max(
      1,
      Number(translateResponse.width || targetEl.naturalWidth || targetEl.width || 1000)
    );
    const imgH = Math.max(
      1,
      Number(translateResponse.height || targetEl.naturalHeight || targetEl.height || 1000)
    );

    const overlayEl = doc.createElement("div");
    overlayEl.className = "mlt-overlay";
    if (overlayEl.dataset) {
      overlayEl.dataset.mltOverlay = "true";
      overlayEl.dataset.lang = String(translateResponse.lang || "auto");
    }

    const bubbleEls = [];

    for (const bubble of translateResponse.bubbles) {
      const translated = String(bubble?.translated ?? "").trim();
      if (!translated) continue;

      const safeBox = bubble.safe_box || bubble.bbox;
      if (!safeBox) continue;

      const bgColor = bubble.bg_color || (bubble.is_dark ? "#111111" : "#FFFFFF");
      const textColor = bubble.text_color || (bubble.is_dark ? "#FFFFFF" : "#111111");
      const isDark = Boolean(bubble.is_dark);

      // 1. Clean interior mask patch over safe_box to erase original text.
      const maskEl = doc.createElement("div");
      maskEl.className = "mlt-mask";
      maskEl.style.left = pct(safeBox.x1, imgW);
      maskEl.style.top = pct(safeBox.y1, imgH);
      maskEl.style.width = pct(Math.max(8, safeBox.x2 - safeBox.x1), imgW);
      maskEl.style.height = pct(Math.max(8, safeBox.y2 - safeBox.y1), imgH);
      maskEl.style.backgroundColor = bgColor;
      overlayEl.appendChild(maskEl);

      // 2. Fitted text box (with controlled expansion if cramped).
      const fit = computeFittedBox(translated, safeBox, { w: imgW, h: imgH }, options.measureFn);
      const bubbleEl = doc.createElement("div");
      bubbleEl.className = "mlt-bubble";
      if (isDark) bubbleEl.classList.add("mlt-bubble-dark");
      if (fit.expanded) bubbleEl.classList.add("mlt-bubble-expanded");

      bubbleEl.style.left = pct(fit.box.x1, imgW);
      bubbleEl.style.top = pct(fit.box.y1, imgH);
      bubbleEl.style.width = pct(fit.box.x2 - fit.box.x1, imgW);
      bubbleEl.style.height = pct(fit.box.y2 - fit.box.y1, imgH);
      bubbleEl.style.backgroundColor = bgColor;
      bubbleEl.style.color = textColor;
      bubbleEl.style.fontSize = `${fit.fontSizePx}px`;
      if (bubbleEl.dataset) {
        bubbleEl.dataset.bubbleId = String(bubble.id ?? "");
        bubbleEl.dataset.baseFontPx = String(fit.fontSizePx);
        bubbleEl.dataset.expanded = fit.expanded ? "true" : "false";
      }

      const textSpan = doc.createElement("span");
      textSpan.className = "mlt-bubble-text";
      textSpan.textContent = translated;
      bubbleEl.appendChild(textSpan);

      // 3. Hoverable hit region + tooltip showing English + original OCR text.
      const hitEl = doc.createElement("div");
      hitEl.className = "mlt-bubble-hit";
      if (typeof hitEl.setAttribute === "function") {
        hitEl.setAttribute("tabindex", "0");
        hitEl.setAttribute("aria-label", translated);
      }

      const tooltipEl = doc.createElement("div");
      tooltipEl.className = "mlt-tooltip";

      const enEl = doc.createElement("div");
      enEl.className = "mlt-tooltip-en";
      enEl.textContent = translated;
      tooltipEl.appendChild(enEl);

      const original = String(bubble.original ?? "").trim();
      if (original) {
        const origEl = doc.createElement("div");
        origEl.className = "mlt-tooltip-orig";
        origEl.textContent = original;
        tooltipEl.appendChild(origEl);
      }

      hitEl.appendChild(tooltipEl);
      bubbleEl.appendChild(hitEl);

      overlayEl.appendChild(bubbleEl);
      bubbleEls.push(bubbleEl);
    }

    const mountParent = targetEl.parentElement || doc.body || doc.documentElement;
    if (mountParent && typeof mountParent.appendChild === "function") {
      mountParent.appendChild(overlayEl);
    }

    const sync = () => {
      if (targetEl.isConnected === false) {
        clearOverlay(targetEl);
        return;
      }
      syncOverlayPosition(targetEl, overlayEl, imgW, bubbleEls, win, doc);
    };
    sync();

    let resizeObserver = null;
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => sync());
      resizeObserver.observe(targetEl);
      if (doc.body) {
        resizeObserver.observe(doc.body);
      }
    }
    if (win && typeof win.addEventListener === "function") {
      win.addEventListener("resize", sync);
    }

    overlaysByTarget.set(targetEl, { overlayEl, resizeObserver, sync, win });

    return {
      element: overlayEl,
      sync,
      destroy: () => clearOverlay(targetEl),
    };
  }

  // Register Alt-peek automatically when loaded inside a browser document.
  if (typeof window !== "undefined" && typeof document !== "undefined") {
    initAltPeekListeners();
  }

  const api = {
    computeFittedBox,
    renderOverlay,
    clearOverlay,
    setPeekOriginal,
    initAltPeekListeners,
  };

  if (root) {
    root.MangaOverlay = api;
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
