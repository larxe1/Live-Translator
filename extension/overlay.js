(function (root) {
  "use strict";

  const MIN_FONT_PX = 7;
  const MIN_RESIZE_FONT_PX = 6;
  const COMFORT_FONT_PX = 11;
  const MAX_FONT_PX = 26;
  const MAX_WIDTH_EXPANSION = 1.25;
  const MAX_HEIGHT_EXPANSION = 1.18;
  const PAGE_MARGIN_PX = 2;
  const EXPANDED_BORDER_PX = 1;

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

  function computeBoxPadding(boxW, boxH, _expanded) {
    const padX = Math.min(6, Math.max(2, Math.round(boxW * 0.04)));
    const padY = Math.min(6, Math.max(2, Math.round(boxH * 0.04)));
    return { padX, padY };
  }

  function computeInnerDimensions(boxW, boxH, expanded) {
    const { padX, padY } = computeBoxPadding(boxW, boxH, expanded);
    const borderTotal = expanded ? EXPANDED_BORDER_PX * 2 : 0;
    // 1px subpixel/kerning safety margin so JS never overestimates rendered CSS width.
    const safetyMargin = 1;
    const innerW = Math.max(0, boxW - padX * 2 - borderTotal - safetyMargin);
    const innerH = Math.max(0, boxH - padY * 2 - borderTotal - safetyMargin);
    return { innerW, innerH, padX, padY, borderTotal };
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

  function findBestFontSize(
    text,
    boxW,
    boxH,
    measureFn,
    ctx,
    expanded = false,
    minFontPx = MIN_FONT_PX
  ) {
    const { innerW, innerH } = computeInnerDimensions(boxW, boxH, expanded);

    if (!evaluateFit(text, minFontPx, innerW, innerH, measureFn, ctx)) {
      return null;
    }
    if (evaluateFit(text, MAX_FONT_PX, innerW, innerH, measureFn, ctx)) {
      return MAX_FONT_PX;
    }

    let lo = minFontPx;
    let hi = MAX_FONT_PX;
    let best = minFontPx;
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

  function coerceBox(raw) {
    if (!raw) return null;
    if (Array.isArray(raw) && raw.length >= 4) {
      return {
        x1: Number(raw[0]),
        y1: Number(raw[1]),
        x2: Number(raw[2]),
        y2: Number(raw[3]),
      };
    }
    if (typeof raw === "object" && ("x1" in raw || "x2" in raw)) {
      return {
        x1: Number(raw.x1 ?? 0),
        y1: Number(raw.y1 ?? 0),
        x2: Number(raw.x2 ?? 0),
        y2: Number(raw.y2 ?? 0),
      };
    }
    return null;
  }

  function normalizeBox(safeBox, imgW, imgH) {
    const box = coerceBox(safeBox) || {};
    const rawX1 = Number(box.x1 ?? 0);
    const rawY1 = Number(box.y1 ?? 0);
    const rawX2 = Number(box.x2 ?? rawX1 + 20);
    const rawY2 = Number(box.y2 ?? rawY1 + 20);
    const x1 = Math.max(0, Math.min(imgW - 8, rawX1));
    const y1 = Math.max(0, Math.min(imgH - 8, rawY1));
    const x2 = Math.min(imgW, Math.max(x1 + 8, rawX2));
    const y2 = Math.min(imgH, Math.max(y1 + 8, rawY2));
    return { x1, y1, x2, y2 };
  }

  function computeUsableBbox(baseBox, fullBbox, imgW, imgH) {
    const normBbox = normalizeBox(fullBbox, imgW, imgH);
    const bw = normBbox.x2 - normBbox.x1;
    const bh = normBbox.y2 - normBbox.y1;
    const insetX = Math.min(4, Math.max(2, Math.round(bw * 0.03)));
    const insetY = Math.min(4, Math.max(2, Math.round(bh * 0.03)));
    const ux1 = Math.min(baseBox.x1, normBbox.x1 + insetX);
    const uy1 = Math.min(baseBox.y1, normBbox.y1 + insetY);
    const ux2 = Math.max(baseBox.x2, normBbox.x2 - insetX);
    const uy2 = Math.max(baseBox.y2, normBbox.y2 - insetY);
    return {
      x1: Math.max(PAGE_MARGIN_PX, Math.min(baseBox.x1, ux1)),
      y1: Math.max(PAGE_MARGIN_PX, Math.min(baseBox.y1, uy1)),
      x2: Math.min(imgW - PAGE_MARGIN_PX, Math.max(baseBox.x2, ux2)),
      y2: Math.min(imgH - PAGE_MARGIN_PX, Math.max(baseBox.y2, uy2)),
    };
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
   * Computes optimal font size and box bounds inside `safeBox` (and optional `bbox`),
   * expanding horizontally into `bbox` or controlled margins (+25% W / +18% H) before
   * shrinking down to MIN_FONT_PX (7px) so whole words never split mid-word.
   */
  function computeFittedBox(text, safeBox, imgSize, measureFnOrOpts, maybeBbox) {
    let measureFn = null;
    let fullBbox = maybeBbox || null;
    if (typeof measureFnOrOpts === "function") {
      measureFn = measureFnOrOpts;
    } else if (measureFnOrOpts && typeof measureFnOrOpts === "object") {
      measureFn = typeof measureFnOrOpts.measureFn === "function" ? measureFnOrOpts.measureFn : null;
      fullBbox = measureFnOrOpts.bbox || fullBbox;
    }

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

    const directComfortFont = findBestFontSize(
      trimmed,
      safeW,
      safeH,
      measureFn,
      ctx,
      false,
      COMFORT_FONT_PX
    );

    let usableBbox = null;
    if (fullBbox) {
      usableBbox = computeUsableBbox(base, fullBbox, imgW, imgH);
      const usableW = usableBbox.x2 - usableBbox.x1;
      const usableH = usableBbox.y2 - usableBbox.y1;
      if (usableW > safeW + 4 && (directComfortFont === null || directComfortFont < 13)) {
        const bboxFont = findBestFontSize(
          trimmed,
          usableW,
          usableH,
          measureFn,
          ctx,
          true,
          COMFORT_FONT_PX
        );
        if (bboxFont !== null && (directComfortFont === null || bboxFont >= directComfortFont + 2)) {
          return { box: usableBbox, fontSizePx: bboxFont, expanded: true };
        }
      }
    }

    if (directComfortFont !== null) {
      return { box: base, fontSizePx: directComfortFont, expanded: false };
    }

    let expandedBox = computeControlledExpandedBox(base, imgW, imgH);
    if (usableBbox) {
      expandedBox = {
        x1: Math.min(expandedBox.x1, usableBbox.x1),
        y1: Math.min(expandedBox.y1, usableBbox.y1),
        x2: Math.max(expandedBox.x2, usableBbox.x2),
        y2: Math.max(expandedBox.y2, usableBbox.y2),
      };
    }

    const expW = expandedBox.x2 - expandedBox.x1;
    const expH = expandedBox.y2 - expandedBox.y1;
    const expFont = findBestFontSize(
      trimmed,
      expW,
      expH,
      measureFn,
      ctx,
      true,
      MIN_FONT_PX
    );

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

  function fitMountedBubbleElement(bubbleEl) {
    if (
      !bubbleEl ||
      typeof bubbleEl.clientWidth !== "number" ||
      typeof bubbleEl.scrollWidth !== "number"
    ) {
      return;
    }
    if (bubbleEl.clientWidth <= 0 || bubbleEl.clientHeight <= 0) {
      return;
    }

    const textSpan =
      typeof bubbleEl.querySelector === "function"
        ? bubbleEl.querySelector(".mlt-bubble-text")
        : null;

    const hasOverflow = () => {
      if (
        bubbleEl.scrollWidth > bubbleEl.clientWidth + 1 ||
        bubbleEl.scrollHeight > bubbleEl.clientHeight + 1
      ) {
        return true;
      }
      if (
        textSpan &&
        typeof textSpan.scrollWidth === "number" &&
        typeof textSpan.clientWidth === "number" &&
        textSpan.clientWidth > 0 &&
        (textSpan.scrollWidth > textSpan.clientWidth + 1 ||
          textSpan.scrollHeight > textSpan.clientHeight + 1)
      ) {
        return true;
      }
      return false;
    };

    let currentPx = parseFloat(bubbleEl.style?.fontSize || "");
    if (!Number.isFinite(currentPx) || currentPx <= 0) return;

    let guard = 0;
    while (hasOverflow() && currentPx > MIN_RESIZE_FONT_PX && guard < 40) {
      currentPx = Math.max(MIN_RESIZE_FONT_PX, currentPx - 0.5);
      bubbleEl.style.fontSize = `${currentPx.toFixed(2)}px`;
      guard += 1;
    }
  }

  function applyScaledBubbleStyles(bubbleEls, scale, runDomFit) {
    for (const bEl of bubbleEls) {
      const baseFont = Number(bEl.dataset?.baseFontPx || 13);
      const scaledFont = Math.max(MIN_RESIZE_FONT_PX, baseFont * scale);
      bEl.style.fontSize = `${scaledFont.toFixed(2)}px`;

      const basePadX = Number(bEl.dataset?.basePadX || 2);
      const basePadY = Number(bEl.dataset?.basePadY || 2);
      const scaledPadY = Math.max(1, basePadY * scale);
      const scaledPadX = Math.max(1, basePadX * scale);
      bEl.style.padding = `${scaledPadY.toFixed(2)}px ${scaledPadX.toFixed(2)}px`;

      if (runDomFit) {
        fitMountedBubbleElement(bEl);
      }
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
      applyScaledBubbleStyles(bubbleEls, scale, true);
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
      applyScaledBubbleStyles(bubbleEls, scale, true);
    }
  }

  function applyBubbleEntryData(entry, bubble, imgW, imgH, options = {}) {
    const translated = String(bubble?.translated ?? "").trim();
    const safeBox = coerceBox(bubble.safe_box) || coerceBox(bubble.bbox);
    if (!translated || !safeBox) return false;
    const fullBbox = coerceBox(bubble.bbox);

    const bgColor = bubble.bg_color || (bubble.is_dark ? "#111111" : "#FFFFFF");
    const textColor = bubble.text_color || (bubble.is_dark ? "#FFFFFF" : "#111111");
    const isDark = Boolean(bubble.is_dark);

    entry.maskEl.style.left = pct(safeBox.x1, imgW);
    entry.maskEl.style.top = pct(safeBox.y1, imgH);
    entry.maskEl.style.width = pct(Math.max(8, safeBox.x2 - safeBox.x1), imgW);
    entry.maskEl.style.height = pct(Math.max(8, safeBox.y2 - safeBox.y1), imgH);
    entry.maskEl.style.backgroundColor = bgColor;

    const fit = computeFittedBox(
      translated,
      safeBox,
      { w: imgW, h: imgH },
      options.measureFn,
      fullBbox
    );
    const boxW = Math.max(8, fit.box.x2 - fit.box.x1);
    const boxH = Math.max(8, fit.box.y2 - fit.box.y1);
    const pad = computeBoxPadding(boxW, boxH, fit.expanded);

    entry.bubbleEl.classList.toggle("mlt-bubble-dark", isDark);
    entry.bubbleEl.classList.toggle("mlt-bubble-expanded", Boolean(fit.expanded));

    entry.bubbleEl.style.left = pct(fit.box.x1, imgW);
    entry.bubbleEl.style.top = pct(fit.box.y1, imgH);
    entry.bubbleEl.style.width = pct(boxW, imgW);
    entry.bubbleEl.style.height = pct(boxH, imgH);
    entry.bubbleEl.style.backgroundColor = bgColor;
    entry.bubbleEl.style.color = textColor;
    entry.bubbleEl.style.fontSize = `${fit.fontSizePx}px`;
    entry.bubbleEl.style.padding = `${pad.padY}px ${pad.padX}px`;

    if (entry.bubbleEl.dataset) {
      entry.bubbleEl.dataset.bubbleId = String(bubble.id ?? "");
      entry.bubbleEl.dataset.baseFontPx = String(fit.fontSizePx);
      entry.bubbleEl.dataset.basePadX = String(pad.padX);
      entry.bubbleEl.dataset.basePadY = String(pad.padY);
      entry.bubbleEl.dataset.expanded = fit.expanded ? "true" : "false";
    }

    entry.textSpan.textContent = translated;
    if (typeof entry.hitEl.setAttribute === "function") {
      entry.hitEl.setAttribute("aria-label", translated);
    }
    entry.enEl.textContent = translated;

    const original = String(bubble.original ?? "").trim();
    if (original) {
      entry.origEl.textContent = original;
      if (entry.origEl.parentNode !== entry.tooltipEl) {
        entry.tooltipEl.appendChild(entry.origEl);
      }
    } else if (entry.origEl.parentNode === entry.tooltipEl) {
      entry.tooltipEl.removeChild(entry.origEl);
    }

    return true;
  }

  function buildBubbleElements(doc, bubble, meta, options = {}) {
    const imgW = Math.max(1, Number(meta?.width || 1000));
    const imgH = Math.max(1, Number(meta?.height || 1000));

    const maskEl = doc.createElement("div");
    maskEl.className = "mlt-mask";

    const bubbleEl = doc.createElement("div");
    bubbleEl.className = "mlt-bubble";

    const textSpan = doc.createElement("span");
    textSpan.className = "mlt-bubble-text";
    bubbleEl.appendChild(textSpan);

    const hitEl = doc.createElement("div");
    hitEl.className = "mlt-bubble-hit";
    if (typeof hitEl.setAttribute === "function") {
      hitEl.setAttribute("tabindex", "0");
    }

    const tooltipEl = doc.createElement("div");
    tooltipEl.className = "mlt-tooltip";

    const enEl = doc.createElement("div");
    enEl.className = "mlt-tooltip-en";
    tooltipEl.appendChild(enEl);

    const origEl = doc.createElement("div");
    origEl.className = "mlt-tooltip-orig";

    hitEl.appendChild(tooltipEl);
    bubbleEl.appendChild(hitEl);

    const entry = {
      maskEl,
      bubbleEl,
      textSpan,
      hitEl,
      tooltipEl,
      enEl,
      origEl,
    };

    if (!applyBubbleEntryData(entry, bubble, imgW, imgH, options)) {
      return null;
    }
    return entry;
  }

  /**
   * Creates an incremental streaming overlay session over `targetEl`.
   * Mounts container + ResizeObserver once on `meta` without painting masks upfront.
   */
  function createOverlaySession(targetEl, meta = {}, options = {}) {
    clearOverlay(targetEl);

    const doc =
      options.document ||
      (targetEl && targetEl.ownerDocument) ||
      (typeof document !== "undefined" ? document : null);
    const win =
      options.window ||
      (doc && doc.defaultView) ||
      (typeof window !== "undefined" ? window : null);

    if (!targetEl || !doc) {
      return {
        overlayEl: null,
        element: null,
        addBubble() {
          return null;
        },
        finish() {},
        sync() {},
        destroy() {},
      };
    }

    initAltPeekListeners(win, doc);

    const imgW = Math.max(
      1,
      Number(meta?.width || targetEl.naturalWidth || targetEl.width || 1000)
    );
    const imgH = Math.max(
      1,
      Number(meta?.height || targetEl.naturalHeight || targetEl.height || 1000)
    );
    const normMeta = { width: imgW, height: imgH, lang: meta?.lang || "auto" };

    const metaBubblesById = new Map();
    if (Array.isArray(meta?.bubbles)) {
      for (const b of meta.bubbles) {
        if (b && b.id !== undefined && b.id !== null) {
          metaBubblesById.set(String(b.id), b);
        }
      }
    }

    const overlayEl = doc.createElement("div");
    overlayEl.className = "mlt-overlay";
    if (overlayEl.dataset) {
      overlayEl.dataset.mltOverlay = "true";
      overlayEl.dataset.lang = String(normMeta.lang);
    }

    const mountParent = targetEl.parentElement || doc.body || doc.documentElement;
    if (mountParent && typeof mountParent.appendChild === "function") {
      mountParent.appendChild(overlayEl);
    }

    const bubbleEls = [];
    const mountedById = new Map();
    let anonCounter = 0;
    let destroyed = false;

    const sync = () => {
      if (destroyed) return;
      if (targetEl.isConnected === false) {
        destroy();
        return;
      }
      syncOverlayPosition(targetEl, overlayEl, imgW, bubbleEls, win, doc);
    };

    const destroy = () => {
      if (destroyed) return;
      destroyed = true;
      mountedById.clear();
      bubbleEls.length = 0;
      clearOverlay(targetEl);
    };

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
    sync();

    function removeMountedEntry(idKey) {
      const existing = mountedById.get(idKey);
      if (!existing) return;
      mountedById.delete(idKey);
      const idx = bubbleEls.indexOf(existing.bubbleEl);
      if (idx !== -1) bubbleEls.splice(idx, 1);
      if (existing.maskEl && existing.maskEl.parentNode === overlayEl) {
        overlayEl.removeChild(existing.maskEl);
      }
      if (existing.bubbleEl && existing.bubbleEl.parentNode === overlayEl) {
        overlayEl.removeChild(existing.bubbleEl);
      }
    }

    function addBubble(bubble, addOpts = {}) {
      if (destroyed || !bubble) return null;

      const hasId = bubble.id !== undefined && bubble.id !== null;
      const idKey = hasId ? String(bubble.id) : `__anon_${anonCounter++}`;
      const metaBase = hasId ? metaBubblesById.get(idKey) : null;
      const merged = metaBase ? { ...metaBase, ...bubble } : bubble;

      const translated = String(merged.translated ?? "").trim();
      if (!translated) {
        removeMountedEntry(idKey);
        if (!addOpts.skipSync) sync();
        return null;
      }

      const existing = mountedById.get(idKey);
      if (existing) {
        const updated = applyBubbleEntryData(existing, merged, imgW, imgH, options);
        if (!updated) {
          removeMountedEntry(idKey);
          if (!addOpts.skipSync) sync();
          return null;
        }
        if (!addOpts.skipSync) sync();
        return existing.bubbleEl;
      }

      const entry = buildBubbleElements(doc, merged, normMeta, options);
      if (!entry) return null;

      overlayEl.appendChild(entry.maskEl);
      overlayEl.appendChild(entry.bubbleEl);
      bubbleEls.push(entry.bubbleEl);
      mountedById.set(idKey, entry);

      if (!addOpts.skipSync) sync();
      return entry.bubbleEl;
    }

    function finish(finalResponse) {
      if (destroyed) return;
      if (finalResponse && Array.isArray(finalResponse.bubbles)) {
        for (const b of finalResponse.bubbles) {
          addBubble(b, { skipSync: true });
        }
      }
      sync();
    }

    return {
      overlayEl,
      element: overlayEl,
      addBubble,
      finish,
      sync,
      destroy,
    };
  }

  /**
   * Renders translated dialogue bubbles aligned pixel-accurate over `targetEl`.
   */
  function renderOverlay(targetEl, translateResponse, options = {}) {
    clearOverlay(targetEl);

    const doc =
      options.document ||
      (targetEl && targetEl.ownerDocument) ||
      (typeof document !== "undefined" ? document : null);

    if (!targetEl || !doc || !translateResponse || !Array.isArray(translateResponse.bubbles)) {
      return { element: null, sync() {}, destroy() {} };
    }

    const session = createOverlaySession(targetEl, translateResponse, options);
    for (const bubble of translateResponse.bubbles) {
      session.addBubble(bubble, { skipSync: true });
    }
    session.finish();

    return {
      element: session.overlayEl,
      sync: session.sync,
      destroy: session.destroy,
    };
  }

  // Register Alt-peek automatically when loaded inside a browser document.
  if (typeof window !== "undefined" && typeof document !== "undefined") {
    initAltPeekListeners();
  }

  const api = {
    MIN_FONT_PX,
    MIN_RESIZE_FONT_PX,
    computeBoxPadding,
    computeInnerDimensions,
    measureWordWidth,
    fitMountedBubbleElement,
    computeFittedBox,
    buildBubbleElements,
    createOverlaySession,
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
