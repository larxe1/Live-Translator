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
  const tooltipPortalsByDoc = new WeakMap();
  let altListenersRegistered = false;
  let sharedMeasureCtx = null;

  function toTranslucentRgba(color, alpha) {
    const raw = String(color || "").trim();
    const a = Math.max(0, Math.min(1, Number(alpha ?? 0.86)));
    const hex6 = raw.match(/^#([0-9a-f]{6})$/i);
    if (hex6) {
      const n = parseInt(hex6[1], 16);
      return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
    }
    const hex3 = raw.match(/^#([0-9a-f]{3})$/i);
    if (hex3) {
      const h = hex3[1];
      const r = parseInt(h[0] + h[0], 16);
      const g = parseInt(h[1] + h[1], 16);
      const b = parseInt(h[2] + h[2], 16);
      return `rgba(${r}, ${g}, ${b}, ${a})`;
    }
    const rgb = raw.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i);
    if (rgb) {
      return `rgba(${rgb[1]}, ${rgb[2]}, ${rgb[3]}, ${a})`;
    }
    return `rgba(255, 255, 255, ${a})`;
  }

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

  function boxesOverlap(a, b, gap = 0) {
    if (!a || !b) return false;
    return (
      a.x1 < b.x2 + gap &&
      a.x2 > b.x1 - gap &&
      a.y1 < b.y2 + gap &&
      a.y2 > b.y1 - gap
    );
  }

  /**
   * Clamps `candidateBox` against `neighborBoxes` so neither expansion nor slightly
   * overlapping original safeBoxes eat into a neighbor's text area.
   */
  function clampBoxAgainstNeighbors(candidateBox, baseBox, neighborBoxes, imgW, imgH, gap = 2) {
    if (!Array.isArray(neighborBoxes) || neighborBoxes.length === 0) {
      return {
        x1: candidateBox.x1,
        y1: candidateBox.y1,
        x2: candidateBox.x2,
        y2: candidateBox.y2,
      };
    }

    const minGap = Math.max(2, Number(gap) || 2);
    const halfGap = minGap / 2;
    const minBoxDim = 8;

    let x1 = candidateBox.x1;
    let y1 = candidateBox.y1;
    let x2 = candidateBox.x2;
    let y2 = candidateBox.y2;

    const ax1 = baseBox.x1;
    const ay1 = baseBox.y1;
    const ax2 = baseBox.x2;
    const ay2 = baseBox.y2;
    const acx = (ax1 + ax2) / 2;
    const acy = (ay1 + ay2) / 2;

    for (const rawNeighbor of neighborBoxes) {
      if (!rawNeighbor) continue;
      const rawBase =
        rawNeighbor.safeBox ||
        rawNeighbor.safe_box ||
        (rawNeighbor.box && !("x1" in rawNeighbor) ? rawNeighbor.box : rawNeighbor);
      const coercedBase = coerceBox(rawBase);
      if (!coercedBase) continue;
      const other = normalizeBox(coercedBase, imgW, imgH);

      const bx1 = other.x1;
      const by1 = other.y1;
      const bx2 = other.x2;
      const by2 = other.y2;

      if (
        Math.abs(bx1 - ax1) < 0.5 &&
        Math.abs(by1 - ay1) < 0.5 &&
        Math.abs(bx2 - ax2) < 0.5 &&
        Math.abs(by2 - ay2) < 0.5
      ) {
        continue;
      }

      const rawCurrent = rawNeighbor.box ? coerceBox(rawNeighbor.box) : null;
      const otherCurrent = rawCurrent ? normalizeBox(rawCurrent, imgW, imgH) : other;

      const bcx = (bx1 + bx2) / 2;
      const bcy = (by1 + by2) / 2;

      const overlapX = Math.min(ax2, bx2) - Math.max(ax1, bx1);
      const overlapY = Math.min(ay2, by2) - Math.max(ay1, by1);
      const sepX = -overlapX;
      const sepY = -overlapY;

      const isLeft = acx < bcx || (acx === bcx && (ax1 < bx1 || (ax1 === bx1 && ay1 <= by1)));
      const isTop = acy < bcy || (acy === bcy && (ay1 < by1 || (ay1 === by1 && ax1 <= bx1)));
      const midX = isLeft ? (ax2 + bx1) / 2 : (bx2 + ax1) / 2;
      const midY = isTop ? (ay2 + by1) / 2 : (by2 + ay1) / 2;

      const clampX = () => {
        if (isLeft) {
          const limit = Math.max(x1 + minBoxDim, Math.floor(midX - halfGap));
          if (x2 > limit) x2 = limit;
        } else {
          const limit = Math.min(x2 - minBoxDim, Math.ceil(midX + halfGap));
          if (x1 < limit) x1 = limit;
        }
      };

      const clampY = () => {
        if (isTop) {
          const limit = Math.max(y1 + minBoxDim, Math.floor(midY - halfGap));
          if (y2 > limit) y2 = limit;
        } else {
          const limit = Math.min(y2 - minBoxDim, Math.ceil(midY + halfGap));
          if (y1 < limit) y1 = limit;
        }
      };

      // Side-by-side: safeBoxes overlap vertically and are horizontally separated (or less overlapping on X).
      if (overlapY > 0 && (sepX >= 0 || overlapX <= overlapY)) {
        if (y1 < Math.max(by2, otherCurrent.y2) + minGap && y2 > Math.min(by1, otherCurrent.y1) - minGap) {
          clampX();
        }
        continue;
      }

      // Stacked: safeBoxes overlap horizontally and are vertically separated (or less overlapping on Y).
      if (overlapX > 0 && (sepY >= 0 || overlapY < overlapX)) {
        if (x1 < Math.max(bx2, otherCurrent.x2) + minGap && x2 > Math.min(bx1, otherCurrent.x1) - minGap) {
          clampY();
        }
        continue;
      }

      // Diagonal separation: clamp if candidate enters neighbor's half-space on both axes
      // or intersects neighbor's current rendered box.
      const crossesX = isLeft ? x2 > midX - halfGap : x1 < midX + halfGap;
      const crossesY = isTop ? y2 > midY - halfGap : y1 < midY + halfGap;
      const overlapsCurrent = boxesOverlap({ x1, y1, x2, y2 }, otherCurrent, minGap);

      if ((crossesX && crossesY) || overlapsCurrent) {
        if (crossesX && (sepX >= sepY || !crossesY)) {
          clampX();
        } else if (crossesY) {
          clampY();
        } else if (overlapsCurrent) {
          if (isLeft && x2 > otherCurrent.x1 - minGap) {
            x2 = Math.max(x1 + minBoxDim, Math.floor(otherCurrent.x1 - minGap));
          } else if (!isLeft && x1 < otherCurrent.x2 + minGap) {
            x1 = Math.min(x2 - minBoxDim, Math.ceil(otherCurrent.x2 + minGap));
          }
        }
      }
    }

    return {
      x1: Math.max(0, Math.min(imgW - minBoxDim, x1)),
      y1: Math.max(0, Math.min(imgH - minBoxDim, y1)),
      x2: Math.min(imgW, Math.max(x1 + minBoxDim, x2)),
      y2: Math.min(imgH, Math.max(y1 + minBoxDim, y2)),
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
    let neighborBoxes = null;
    let collisionGap = 2;
    if (typeof measureFnOrOpts === "function") {
      measureFn = measureFnOrOpts;
    } else if (measureFnOrOpts && typeof measureFnOrOpts === "object") {
      measureFn = typeof measureFnOrOpts.measureFn === "function" ? measureFnOrOpts.measureFn : null;
      fullBbox = measureFnOrOpts.bbox || fullBbox;
      if (Array.isArray(measureFnOrOpts.neighborBoxes)) {
        neighborBoxes = measureFnOrOpts.neighborBoxes;
      } else if (Array.isArray(measureFnOrOpts.neighbors)) {
        neighborBoxes = measureFnOrOpts.neighbors;
      }
      if (Number.isFinite(Number(measureFnOrOpts.collisionGap))) {
        collisionGap = Number(measureFnOrOpts.collisionGap);
      }
    }

    const imgW = Math.max(16, Number(imgSize?.w ?? imgSize?.width ?? 1000));
    const imgH = Math.max(16, Number(imgSize?.h ?? imgSize?.height ?? 1000));
    const base = normalizeBox(safeBox, imgW, imgH);
    const hasNeighbors = Boolean(neighborBoxes && neighborBoxes.length > 0);
    const clampedBase = hasNeighbors
      ? clampBoxAgainstNeighbors(base, base, neighborBoxes, imgW, imgH, collisionGap)
      : base;
    const trimmed = String(text || "").trim();

    if (!trimmed) {
      return { box: clampedBase, fontSizePx: 14, expanded: false };
    }

    const ctx = measureFn ? null : getMeasureContext();
    const safeW = clampedBase.x2 - clampedBase.x1;
    const safeH = clampedBase.y2 - clampedBase.y1;

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
      usableBbox = computeUsableBbox(clampedBase, fullBbox, imgW, imgH);
      if (hasNeighbors) {
        usableBbox = clampBoxAgainstNeighbors(usableBbox, base, neighborBoxes, imgW, imgH, collisionGap);
      }
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
      return { box: clampedBase, fontSizePx: directComfortFont, expanded: false };
    }

    let expandedBox = computeControlledExpandedBox(clampedBase, imgW, imgH);
    if (usableBbox) {
      expandedBox = {
        x1: Math.min(expandedBox.x1, usableBbox.x1),
        y1: Math.min(expandedBox.y1, usableBbox.y1),
        x2: Math.max(expandedBox.x2, usableBbox.x2),
        y2: Math.max(expandedBox.y2, usableBbox.y2),
      };
    }
    if (hasNeighbors) {
      expandedBox = clampBoxAgainstNeighbors(expandedBox, base, neighborBoxes, imgW, imgH, collisionGap);
    }

    const didExpand =
      expandedBox.x1 < clampedBase.x1 - 0.5 ||
      expandedBox.y1 < clampedBase.y1 - 0.5 ||
      expandedBox.x2 > clampedBase.x2 + 0.5 ||
      expandedBox.y2 > clampedBase.y2 + 0.5;

    const expW = expandedBox.x2 - expandedBox.x1;
    const expH = expandedBox.y2 - expandedBox.y1;
    const expFont = findBestFontSize(
      trimmed,
      expW,
      expH,
      measureFn,
      ctx,
      didExpand,
      MIN_FONT_PX
    );

    return {
      box: expandedBox,
      fontSizePx: expFont !== null ? expFont : MIN_FONT_PX,
      expanded: didExpand,
    };
  }

  function pct(value, total) {
    return `${((value / Math.max(1, total)) * 100).toFixed(4)}%`;
  }

  function isDescendantOf(node, ancestor) {
    let cur = node;
    while (cur) {
      if (cur === ancestor) return true;
      cur = cur.parentNode || cur.parentElement || null;
    }
    return false;
  }

  function getOrCreateTooltipPortal(doc, win) {
    if (!doc || !doc.body || typeof doc.createElement !== "function") return null;
    let state = tooltipPortalsByDoc.get(doc);
    if (state && state.portalEl && state.portalEl.parentNode === doc.body) {
      return state;
    }

    const portalEl = doc.createElement("div");
    portalEl.className = "mlt-tooltip-portal";
    if (typeof portalEl.setAttribute === "function") {
      portalEl.setAttribute("role", "tooltip");
    }
    portalEl.style.display = "none";

    const enEl = doc.createElement("div");
    enEl.className = "mlt-tooltip-portal-en";
    portalEl.appendChild(enEl);

    const origEl = doc.createElement("div");
    origEl.className = "mlt-tooltip-portal-orig";

    doc.body.appendChild(portalEl);

    state = {
      portalEl,
      enEl,
      origEl,
      activeHitEl: null,
      listenersAttached: false,
    };

    const w = win || doc.defaultView || (typeof window !== "undefined" ? window : null);
    if (w && typeof w.addEventListener === "function") {
      const onGlobalHide = () => hideTooltipPortal(doc);
      w.addEventListener("scroll", onGlobalHide, true);
      w.addEventListener("blur", onGlobalHide);
      w.addEventListener("resize", onGlobalHide);
      state.listenersAttached = true;
    }

    tooltipPortalsByDoc.set(doc, state);
    return state;
  }

  function positionTooltipPortal(state, hitEl, win, doc) {
    const portalEl = state.portalEl;
    if (!portalEl || !hitEl || typeof hitEl.getBoundingClientRect !== "function") return;

    const rect = hitEl.getBoundingClientRect();
    const rectLeft = Number(rect?.left ?? 0);
    const rectTop = Number(rect?.top ?? 0);
    const rectWidth = Number(
      rect?.width ?? (rect?.right !== undefined ? rect.right - rectLeft : 0)
    );
    const rectHeight = Number(
      rect?.height ?? (rect?.bottom !== undefined ? rect.bottom - rectTop : 0)
    );
    const rectBottom = Number(rect?.bottom ?? rectTop + rectHeight);

    const vw = Number(win?.innerWidth || doc?.documentElement?.clientWidth || 1024);
    const vh = Number(win?.innerHeight || doc?.documentElement?.clientHeight || 768);

    const tipRect =
      typeof portalEl.getBoundingClientRect === "function"
        ? portalEl.getBoundingClientRect()
        : null;
    const tipW = Number(portalEl.offsetWidth || tipRect?.width || 240);
    const tipH = Number(portalEl.offsetHeight || tipRect?.height || 56);

    const gap = 8;
    const margin = 8;
    let top = rectTop - tipH - gap;
    let placement = "top";
    if (top < margin && (vh - rectBottom >= rectTop || vh - rectBottom >= tipH + gap + margin)) {
      top = rectBottom + gap;
      placement = "bottom";
    }
    top = Math.max(margin, Math.min(Math.max(margin, vh - tipH - margin), top));

    const centerX = rectLeft + rectWidth / 2;
    let left = centerX - tipW / 2;
    left = Math.max(margin, Math.min(Math.max(margin, vw - tipW - margin), left));

    portalEl.style.left = `${Math.round(left)}px`;
    portalEl.style.top = `${Math.round(top)}px`;
    if (portalEl.dataset) {
      portalEl.dataset.placement = placement;
    }
  }

  function showTooltipPortal(doc, win, hitEl, translated, original) {
    const state = getOrCreateTooltipPortal(doc, win);
    if (!state) return;

    const enText = String(translated || "").trim();
    const origText = String(original || "").trim();
    if (!enText && !origText) {
      hideTooltipPortal(doc);
      return;
    }

    state.activeHitEl = hitEl;
    state.enEl.textContent = enText;
    if (origText) {
      state.origEl.textContent = origText;
      if (state.origEl.parentNode !== state.portalEl) {
        state.portalEl.appendChild(state.origEl);
      }
    } else if (state.origEl.parentNode === state.portalEl) {
      state.portalEl.removeChild(state.origEl);
    }

    state.portalEl.style.display = "block";
    positionTooltipPortal(state, hitEl, win, doc);
  }

  function hideTooltipPortal(doc, onlyIfHitEl = null, onlyIfInsideOverlayEl = null) {
    if (!doc) return;
    const state = tooltipPortalsByDoc.get(doc);
    if (!state || !state.portalEl) return;
    if (onlyIfHitEl && state.activeHitEl !== onlyIfHitEl) return;
    if (
      onlyIfInsideOverlayEl &&
      state.activeHitEl &&
      !isDescendantOf(state.activeHitEl, onlyIfInsideOverlayEl)
    ) {
      return;
    }
    state.activeHitEl = null;
    state.portalEl.style.display = "none";
  }

  function setPeekOriginal(enabled, doc) {
    const d = doc || (typeof document !== "undefined" ? document : null);
    if (d && d.documentElement && d.documentElement.classList) {
      d.documentElement.classList.toggle("mlt-peek-original", Boolean(enabled));
    }
    if (enabled && d) {
      hideTooltipPortal(d);
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

    if (entry.doc) {
      hideTooltipPortal(entry.doc, null, entry.overlayEl);
    }
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
    const rawSafeBox = coerceBox(bubble.safe_box) || coerceBox(bubble.bbox);
    if (!translated || !rawSafeBox) return false;
    const fullBbox = coerceBox(bubble.bbox);
    const safeBox = normalizeBox(rawSafeBox, imgW, imgH);
    const neighborBoxes = Array.isArray(options.neighborBoxes) ? options.neighborBoxes : null;
    const maskBox =
      neighborBoxes && neighborBoxes.length > 0
        ? clampBoxAgainstNeighbors(safeBox, safeBox, neighborBoxes, imgW, imgH)
        : safeBox;

    const bgColor = bubble.bg_color || (bubble.is_dark ? "#111111" : "#FFFFFF");
    const textColor = bubble.text_color || (bubble.is_dark ? "#FFFFFF" : "#111111");
    const isDark = Boolean(bubble.is_dark);
    const isFloating = bubble.region_type === "floating";

    entry.maskEl.classList.toggle("mlt-mask-floating", isFloating);
    entry.maskEl.style.display = isFloating ? "none" : "";
    entry.maskEl.style.left = pct(maskBox.x1, imgW);
    entry.maskEl.style.top = pct(maskBox.y1, imgH);
    entry.maskEl.style.width = pct(Math.max(8, maskBox.x2 - maskBox.x1), imgW);
    entry.maskEl.style.height = pct(Math.max(8, maskBox.y2 - maskBox.y1), imgH);
    entry.maskEl.style.backgroundColor = isFloating ? "transparent" : bgColor;

    const fit = computeFittedBox(
      translated,
      safeBox,
      { w: imgW, h: imgH },
      {
        measureFn: options.measureFn,
        bbox: fullBbox,
        neighborBoxes,
      }
    );
    const boxW = Math.max(8, fit.box.x2 - fit.box.x1);
    const boxH = Math.max(8, fit.box.y2 - fit.box.y1);
    const pad = computeBoxPadding(boxW, boxH, fit.expanded);

    entry.bubbleEl.classList.toggle("mlt-bubble-dark", isDark);
    entry.bubbleEl.classList.toggle("mlt-bubble-expanded", Boolean(fit.expanded));
    entry.bubbleEl.classList.toggle("mlt-bubble-floating", isFloating);

    entry.bubbleEl.style.left = pct(fit.box.x1, imgW);
    entry.bubbleEl.style.top = pct(fit.box.y1, imgH);
    entry.bubbleEl.style.width = pct(boxW, imgW);
    entry.bubbleEl.style.height = pct(boxH, imgH);
    entry.bubbleEl.style.backgroundColor = isFloating
      ? toTranslucentRgba(bgColor, isDark ? 0.84 : 0.86)
      : bgColor;
    entry.bubbleEl.style.color = textColor;
    entry.bubbleEl.style.fontSize = `${fit.fontSizePx}px`;
    entry.bubbleEl.style.padding = `${pad.padY}px ${pad.padX}px`;

    if (entry.bubbleEl.dataset) {
      entry.bubbleEl.dataset.bubbleId = String(bubble.id ?? "");
      entry.bubbleEl.dataset.baseFontPx = String(fit.fontSizePx);
      entry.bubbleEl.dataset.basePadX = String(pad.padX);
      entry.bubbleEl.dataset.basePadY = String(pad.padY);
      entry.bubbleEl.dataset.expanded = fit.expanded ? "true" : "false";
      entry.bubbleEl.dataset.regionType = String(bubble.region_type || "bubble");
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

    entry.bubble = bubble;
    entry.safeBox = safeBox;
    entry.fit = fit;
    entry.translated = translated;
    entry.original = original;

    if (entry.doc) {
      const portalState = tooltipPortalsByDoc.get(entry.doc);
      if (portalState && portalState.activeHitEl === entry.hitEl) {
        showTooltipPortal(entry.doc, entry.win, entry.hitEl, translated, original);
      }
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

    const win =
      options.window ||
      doc.defaultView ||
      (typeof window !== "undefined" ? window : null);

    const entry = {
      doc,
      win,
      maskEl,
      bubbleEl,
      textSpan,
      hitEl,
      tooltipEl,
      enEl,
      origEl,
      bubble: null,
      safeBox: null,
      fit: null,
      translated: "",
      original: "",
    };

    if (typeof hitEl.addEventListener === "function") {
      const onShow = () => {
        showTooltipPortal(doc, win, hitEl, entry.translated, entry.original);
      };
      const onHide = () => {
        hideTooltipPortal(doc, hitEl);
      };
      hitEl.addEventListener("mouseenter", onShow);
      hitEl.addEventListener("focusin", onShow);
      hitEl.addEventListener("mouseleave", onHide);
      hitEl.addEventListener("focusout", onHide);
    }

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
    const skippedIds = new Set();
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
      skippedIds.clear();
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

    overlaysByTarget.set(targetEl, { overlayEl, resizeObserver, sync, win, doc });
    sync();

    function collectNeighborBoxes(selfIdKey, mountedOnly = false) {
      const neighbors = [];
      for (const [idKey, mEntry] of mountedById.entries()) {
        if (idKey === selfIdKey || !mEntry.safeBox) continue;
        neighbors.push({
          safeBox: mEntry.safeBox,
          box: mEntry.fit ? mEntry.fit.box : mEntry.safeBox,
        });
      }
      if (!mountedOnly) {
        for (const [idKey, mBubble] of metaBubblesById.entries()) {
          if (idKey === selfIdKey || mountedById.has(idKey) || skippedIds.has(idKey)) continue;
          const mb = coerceBox(mBubble.safe_box) || coerceBox(mBubble.bbox);
          if (mb) neighbors.push(mb);
        }
      }
      return neighbors;
    }

    function resolveOverlayCollisions(mountedOnly = false, onlyIds = null) {
      if (mountedById.size <= 1) return;
      for (const [idKey, mEntry] of mountedById.entries()) {
        if (onlyIds && !onlyIds.has(idKey)) continue;
        if (!mEntry.bubble) continue;
        const neighborBoxes = collectNeighborBoxes(idKey, mountedOnly);
        applyBubbleEntryData(mEntry, mEntry.bubble, imgW, imgH, {
          ...options,
          document: doc,
          window: win,
          neighborBoxes,
        });
      }
    }

    function removeMountedEntry(idKey) {
      const existing = mountedById.get(idKey);
      if (!existing) return;
      mountedById.delete(idKey);
      const idx = bubbleEls.indexOf(existing.bubbleEl);
      if (idx !== -1) bubbleEls.splice(idx, 1);
      const portalState = tooltipPortalsByDoc.get(doc);
      if (portalState && portalState.activeHitEl === existing.hitEl) {
        hideTooltipPortal(doc, existing.hitEl);
      }
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
        if (hasId) skippedIds.add(idKey);
        removeMountedEntry(idKey);
        if (!addOpts.skipSync) sync();
        return null;
      }
      if (hasId) skippedIds.delete(idKey);

      const neighborBoxes = collectNeighborBoxes(idKey, false);
      const entryOpts = {
        ...options,
        document: doc,
        window: win,
        neighborBoxes,
      };

      const existing = mountedById.get(idKey);
      if (existing) {
        const updated = applyBubbleEntryData(existing, merged, imgW, imgH, entryOpts);
        if (!updated) {
          removeMountedEntry(idKey);
          if (!addOpts.skipSync) sync();
          return null;
        }
        if (!addOpts.skipSync) {
          const colliding = new Set();
          for (const [otherId, otherEntry] of mountedById.entries()) {
            if (otherId === idKey) continue;
            if (
              boxesOverlap(otherEntry.fit?.box, existing.fit?.box, 2) ||
              boxesOverlap(otherEntry.fit?.box, existing.safeBox, 2)
            ) {
              colliding.add(otherId);
            }
          }
          if (colliding.size > 0) resolveOverlayCollisions(false, colliding);
          sync();
        }
        return existing.bubbleEl;
      }

      const entry = buildBubbleElements(doc, merged, normMeta, entryOpts);
      if (!entry) return null;

      overlayEl.appendChild(entry.maskEl);
      overlayEl.appendChild(entry.bubbleEl);
      bubbleEls.push(entry.bubbleEl);
      mountedById.set(idKey, entry);

      if (!addOpts.skipSync) {
        const colliding = new Set();
        for (const [otherId, otherEntry] of mountedById.entries()) {
          if (otherId === idKey) continue;
          if (
            !metaBase ||
            boxesOverlap(otherEntry.fit?.box, entry.fit?.box, 2) ||
            boxesOverlap(otherEntry.fit?.box, entry.safeBox, 2)
          ) {
            colliding.add(otherId);
          }
        }
        if (colliding.size > 0) resolveOverlayCollisions(false, colliding);
        sync();
      }
      return entry.bubbleEl;
    }

    function finish(finalResponse) {
      if (destroyed) return;
      if (finalResponse && Array.isArray(finalResponse.bubbles)) {
        for (const b of finalResponse.bubbles) {
          addBubble(b, { skipSync: true });
        }
      }
      resolveOverlayCollisions(true);
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
    clampBoxAgainstNeighbors,
    computeFittedBox,
    buildBubbleElements,
    createOverlaySession,
    renderOverlay,
    clearOverlay,
    hideTooltipPortal,
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
