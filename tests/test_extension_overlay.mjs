import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const overlay = require("../extension/overlay.js");
const content = require("../extension/content.js");
const background = require("../extension/background.js");

function createMockClassList(el) {
  const set = new Set();
  return {
    add(...names) {
      for (const n of names) set.add(n);
      el.className = [...set].join(" ");
    },
    remove(...names) {
      for (const n of names) set.delete(n);
      el.className = [...set].join(" ");
    },
    toggle(name, force) {
      const next = force !== undefined ? Boolean(force) : !set.has(name);
      if (next) set.add(name);
      else set.delete(name);
      el.className = [...set].join(" ");
      return next;
    },
    contains(name) {
      return set.has(name);
    },
    _syncFromString(str) {
      set.clear();
      for (const part of String(str || "").split(/\s+/).filter(Boolean)) {
        set.add(part);
      }
    },
  };
}

function createMockDocument() {
  const listeners = new Map();

  function makeElement(tagName) {
    const el = {
      tagName: String(tagName).toUpperCase(),
      children: [],
      parentNode: null,
      parentElement: null,
      style: {},
      dataset: {},
      attributes: new Map(),
      _textContent: "",
      _className: "",
      ownerDocument: null,
      listeners: new Map(),
      complete: true,
      naturalWidth: 0,
      naturalHeight: 0,
      width: 0,
      height: 0,
      src: "",
      currentSrc: "",
      isConnected: true,
    };
    const cl = createMockClassList(el);
    Object.defineProperty(el, "className", {
      get() {
        return el._className;
      },
      set(v) {
        el._className = String(v || "");
        cl._syncFromString(el._className);
      },
    });
    el.classList = cl;
    Object.defineProperty(el, "textContent", {
      get() {
        if (el.children.length === 0) return el._textContent;
        return el._textContent + el.children.map((c) => c.textContent).join("");
      },
      set(v) {
        el._textContent = String(v ?? "");
        if (el._textContent === "") el.children = [];
      },
    });
    el.appendChild = (child) => {
      if (child.parentNode && child.parentNode !== el) {
        const idx = child.parentNode.children.indexOf(child);
        if (idx !== -1) child.parentNode.children.splice(idx, 1);
      }
      child.parentNode = el;
      child.parentElement = el;
      child.ownerDocument = el.ownerDocument;
      el.children.push(child);
      return child;
    };
    el.removeChild = (child) => {
      const idx = el.children.indexOf(child);
      if (idx !== -1) {
        el.children.splice(idx, 1);
        child.parentNode = null;
        child.parentElement = null;
      }
      return child;
    };
    el.setAttribute = (k, v) => el.attributes.set(k, String(v));
    el.getAttribute = (k) => (el.attributes.has(k) ? el.attributes.get(k) : "");
    el.addEventListener = (type, fn) => {
      if (!el.listeners.has(type)) el.listeners.set(type, []);
      el.listeners.get(type).push(fn);
    };
    el.removeEventListener = (type, fn) => {
      const arr = el.listeners.get(type) || [];
      el.listeners.set(
        type,
        arr.filter((f) => f !== fn)
      );
    };
    el.dispatchEvent = (evt) => {
      for (const fn of el.listeners.get(evt.type) || []) fn(evt);
    };
    el.closest = (selector) => {
      const cls = selector.startsWith(".") ? selector.slice(1) : selector;
      let cur = el;
      while (cur) {
        if (cur.classList && cur.classList.contains(cls)) return cur;
        cur = cur.parentElement;
      }
      return null;
    };
    el.querySelectorAll = (selector) => {
      const selectors = selector.split(",").map((s) => s.trim());
      const out = [];
      function walk(node) {
        for (const child of node.children || []) {
          const matches = selectors.some((sel) => {
            if (sel.startsWith(".")) return child.classList?.contains(sel.slice(1));
            return child.tagName === sel.toUpperCase();
          });
          if (matches) out.push(child);
          walk(child);
        }
      }
      walk(el);
      return out;
    };
    el.querySelector = (selector) => el.querySelectorAll(selector)[0] || null;
    el.getBoundingClientRect = () => ({
      left: 0,
      top: 0,
      width: el.width || el.naturalWidth || 0,
      height: el.height || el.naturalHeight || 0,
    });
    return el;
  }

  const doc = {
    hidden: false,
    documentElement: makeElement("html"),
    body: makeElement("body"),
    createElement(tag) {
      const el = makeElement(tag);
      el.ownerDocument = doc;
      return el;
    },
    querySelectorAll(selector) {
      return doc.body.querySelectorAll(selector);
    },
    querySelector(selector) {
      return doc.body.querySelector(selector);
    },
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(fn);
    },
    removeEventListener(type, fn) {
      const arr = listeners.get(type) || [];
      listeners.set(
        type,
        arr.filter((f) => f !== fn)
      );
    },
    dispatchEvent(evt) {
      for (const fn of listeners.get(evt.type) || []) fn(evt);
    },
  };
  doc.documentElement.ownerDocument = doc;
  doc.body.ownerDocument = doc;
  doc.documentElement.appendChild(doc.body);
  return doc;
}

function createMockWindow(doc, href = "https://mangadex.org/chapter/1") {
  const listeners = new Map();
  const url = new URL(href);
  return {
    location: {
      href,
      hostname: url.hostname,
      pathname: url.pathname,
    },
    document: doc,
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(fn);
    },
    removeEventListener(type, fn) {
      const arr = listeners.get(type) || [];
      listeners.set(
        type,
        arr.filter((f) => f !== fn)
      );
    },
    dispatchEvent(evt) {
      for (const fn of listeners.get(evt.type) || []) fn(evt);
    },
    getComputedStyle() {
      return { position: "static" };
    },
  };
}

test("1. narrow vertical bubble triggers controlled expansion within +25% width / +18% height", () => {
  const safeBox = { x1: 120, y1: 80, x2: 175, y2: 340 }; // 55 x 260
  const imgSize = { w: 900, h: 1300 };
  const text = "A giant centipede! Run!";

  const fit = overlay.computeFittedBox(text, safeBox, imgSize);
  assert.ok(fit.fontSizePx >= 11, `Expected fontSizePx >= 11, got ${fit.fontSizePx}`);
  assert.equal(fit.expanded, true);

  const expandedW = fit.box.x2 - fit.box.x1;
  const expandedH = fit.box.y2 - fit.box.y1;
  assert.ok(expandedW <= 55 * 1.25 + 0.001, `Expanded width ${expandedW} exceeded +25%`);
  assert.ok(expandedH <= 260 * 1.18 + 0.001, `Expanded height ${expandedH} exceeded +18%`);
  assert.ok(fit.box.x1 >= 2 && fit.box.x2 <= 898);
  assert.ok(fit.box.y1 >= 2 && fit.box.y2 <= 1298);

  const doc = createMockDocument();
  const win = createMockWindow(doc);
  const img = doc.createElement("img");
  img.width = 900;
  img.height = 1300;
  doc.body.appendChild(img);

  const handle = overlay.renderOverlay(
    img,
    {
      lang: "ja",
      width: 900,
      height: 1300,
      bubbles: [
        {
          id: 0,
          bbox: { x1: 112, y1: 70, x2: 183, y2: 350 },
          safe_box: safeBox,
          bg_color: "#FFFFFF",
          text_color: "#111111",
          is_dark: false,
          original: "誰か助けてくれ！",
          translated: text,
        },
      ],
    },
    { document: doc, window: win }
  );

  const bubbleEl = handle.element.querySelector(".mlt-bubble");
  assert.ok(bubbleEl.classList.contains("mlt-bubble-expanded"));
  handle.destroy();
});

test("2. normal roomy bubble fits cleanly inside safe_box without expansion", () => {
  const safeBox = { x1: 200, y1: 150, x2: 460, y2: 330 }; // 260 x 180
  const fit = overlay.computeFittedBox("Hey, watch out!", safeBox, { w: 900, h: 1300 });
  assert.equal(fit.expanded, false);
  assert.ok(fit.fontSizePx >= 14 && fit.fontSizePx <= 26);
  assert.deepEqual(fit.box, safeBox);
});

test("3. dark bubble uses high-contrast white ink and hover tooltip shows English + original OCR", () => {
  const doc = createMockDocument();
  const win = createMockWindow(doc);
  const img = doc.createElement("img");
  img.width = 800;
  img.height = 1200;
  doc.body.appendChild(img);

  const handle = overlay.renderOverlay(
    img,
    {
      lang: "ja",
      width: 800,
      height: 1200,
      bubbles: [
        {
          id: 0,
          bbox: { x1: 40, y1: 40, x2: 260, y2: 240 },
          safe_box: { x1: 55, y1: 55, x2: 245, y2: 225 },
          bg_color: "#111111",
          text_color: "#FFFFFF",
          is_dark: true,
          original: "ぎゃああああっ",
          translated: "GYAAAAAH!",
        },
      ],
    },
    { document: doc, window: win }
  );

  const maskEl = handle.element.querySelector(".mlt-mask");
  const bubbleEl = handle.element.querySelector(".mlt-bubble");
  const enTip = handle.element.querySelector(".mlt-tooltip-en");
  const origTip = handle.element.querySelector(".mlt-tooltip-orig");

  assert.equal(maskEl.style.backgroundColor, "#111111");
  assert.ok(bubbleEl.classList.contains("mlt-bubble-dark"));
  assert.equal(bubbleEl.style.color, "#FFFFFF");
  assert.equal(bubbleEl.style.backgroundColor, "#111111");
  assert.equal(enTip.textContent, "GYAAAAAH!");
  assert.equal(origTip.textContent, "ぎゃああああっ");
  handle.destroy();
});

test("4. Alt peek hides overlays via class toggle and restores cleanly on keyup or blur", () => {
  const doc = createMockDocument();
  const win = createMockWindow(doc);
  const img = doc.createElement("img");
  img.width = 800;
  img.height = 1200;
  doc.body.appendChild(img);

  overlay.renderOverlay(
    img,
    {
      lang: "ja",
      width: 800,
      height: 1200,
      bubbles: [
        {
          id: 0,
          bbox: { x1: 100, y1: 100, x2: 300, y2: 240 },
          safe_box: { x1: 110, y1: 110, x2: 290, y2: 230 },
          bg_color: "#FFFFFF",
          text_color: "#111111",
          is_dark: false,
          original: "今！",
          translated: "Now!",
        },
      ],
    },
    { document: doc, window: win }
  );

  const cleanup = overlay.initAltPeekListeners(win, doc);
  win.dispatchEvent({ type: "keydown", key: "Alt" });
  assert.equal(doc.documentElement.classList.contains("mlt-peek-original"), true);
  assert.ok(doc.querySelector(".mlt-overlay"), "Overlay DOM must stay mounted while peeking");

  win.dispatchEvent({ type: "keyup", key: "Alt" });
  assert.equal(doc.documentElement.classList.contains("mlt-peek-original"), false);

  // Also ensure blur resets peek so Alt never sticks across Alt+Tab.
  win.dispatchEvent({ type: "keydown", key: "Alt" });
  assert.equal(doc.documentElement.classList.contains("mlt-peek-original"), true);
  win.dispatchEvent({ type: "blur" });
  assert.equal(doc.documentElement.classList.contains("mlt-peek-original"), false);
  cleanup();
});

test("5. opt-in dormant behavior on disabled hostname mounts zero pills and sends zero requests", async () => {
  const doc = createMockDocument();
  const win = createMockWindow(doc, "https://www.youtube.com/watch?v=123");
  const img = doc.createElement("img");
  img.src = "https://i.ytimg.com/vi/123/maxresdefault.jpg";
  img.width = 800;
  img.height = 1000;
  img.naturalWidth = 800;
  img.naturalHeight = 1000;
  doc.body.appendChild(img);

  let observerConstructed = 0;
  class SpyObserver {
    constructor() {
      observerConstructed += 1;
    }
    observe() {}
    disconnect() {}
  }

  const messages = [];
  const controller = content.createPageController({
    document: doc,
    window: win,
    hostname: "www.youtube.com",
    storage: {
      async get() {
        return { enabledDomains: { "mangadex.org": true }, langMode: "auto", glossary: {} };
      },
    },
    sendMessage: async (msg) => {
      messages.push(msg);
      return { ok: true };
    },
    overlayApi: overlay,
    MutationObserver: SpyObserver,
    IntersectionObserver: SpyObserver,
  });

  const status = await controller.init();
  assert.equal(status.enabled, false);
  assert.equal(controller.isEnabled(), false);
  assert.equal(observerConstructed, 0);
  assert.equal(messages.length, 0);
  assert.equal(doc.querySelector(".mlt-pill"), null);
});

test("6. paginated src swap clears stale overlay and discards late in-flight response", async () => {
  const doc = createMockDocument();
  const win = createMockWindow(doc, "https://mangadex.org/chapter/10");
  const img = doc.createElement("img");
  img.src = "https://cdn.example.org/page1.jpg";
  img.width = 800;
  img.height = 1200;
  img.naturalWidth = 800;
  img.naturalHeight = 1200;
  doc.body.appendChild(img);

  let resolvePage1;
  let resolvePage2;
  const sent = [];

  const controller = content.createPageController({
    document: doc,
    window: win,
    hostname: "mangadex.org",
    storage: {
      async get() {
        return { enabledDomains: { "mangadex.org": true }, langMode: "ja", glossary: {} };
      },
    },
    sendMessage: (msg) => {
      sent.push(msg);
      if (msg.type === "CANCEL_TRANSLATE") return Promise.resolve({ ok: true });
      if (msg.imageUrl?.endsWith("page1.jpg")) {
        return new Promise((r) => {
          resolvePage1 = r;
        });
      }
      return new Promise((r) => {
        resolvePage2 = r;
      });
    },
    overlayApi: overlay,
    MutationObserver: null,
    IntersectionObserver: null,
  });

  await controller.init();
  controller.setVisibility(img, true);
  const p1Promise = controller.refreshVisibleAndLookahead();
  assert.equal(sent.some((m) => m.imageUrl?.endsWith("page1.jpg")), true);

  // User clicks "Next Page" while page 1 is still translating.
  img.src = "https://cdn.example.org/page2.jpg";
  controller.handleSourceMutation(img);

  assert.equal(sent.some((m) => m.type === "CANCEL_TRANSLATE"), true);
  assert.equal(doc.querySelector(".mlt-overlay"), null);

  resolvePage2({
    ok: true,
    data: {
      lang: "ja",
      width: 800,
      height: 1200,
      bubbles: [
        {
          id: 0,
          bbox: { x1: 100, y1: 100, x2: 320, y2: 260 },
          safe_box: { x1: 110, y1: 110, x2: 310, y2: 250 },
          bg_color: "#FFFFFF",
          text_color: "#111111",
          is_dark: false,
          original: "二ページ目",
          translated: "Page Two Dialogue",
        },
      ],
    },
  });
  await new Promise((r) => setTimeout(r, 0));

  // Late response from page 1 arrives after page 2 already rendered.
  resolvePage1({
    ok: true,
    data: {
      lang: "ja",
      width: 800,
      height: 1200,
      bubbles: [
        {
          id: 0,
          bbox: { x1: 100, y1: 100, x2: 320, y2: 260 },
          safe_box: { x1: 110, y1: 110, x2: 310, y2: 250 },
          bg_color: "#FFFFFF",
          text_color: "#111111",
          is_dark: false,
          original: "一ページ目",
          translated: "STALE PAGE ONE DIALOGUE",
        },
      ],
    },
  });
  await p1Promise;

  const renderedText = doc.querySelector(".mlt-bubble-text")?.textContent;
  assert.equal(renderedText, "Page Two Dialogue");
});

test("7. look-ahead priority ordering sends visible=0, next 1=1, and defers distant pages", async () => {
  const doc = createMockDocument();
  const win = createMockWindow(doc, "https://webtoons.example/ep1");
  const pages = [];
  for (let i = 0; i < 6; i += 1) {
    const img = doc.createElement("img");
    img.src = `https://cdn.example.org/p${i}.jpg`;
    img.width = 800;
    img.height = 1400;
    img.naturalWidth = 800;
    img.naturalHeight = 1400;
    doc.body.appendChild(img);
    pages.push(img);
  }

  const calls = [];
  const controller = content.createPageController({
    document: doc,
    window: win,
    hostname: "webtoons.example",
    storage: {
      async get() {
        return { enabledDomains: { "webtoons.example": true }, langMode: "ko", glossary: {} };
      },
    },
    sendMessage: async (msg) => {
      if (msg.type === "TRANSLATE_IMAGE") {
        calls.push({ url: msg.imageUrl, priority: msg.priority });
        return { ok: true, data: { lang: "ko", width: 800, height: 1400, bubbles: [] } };
      }
      return { ok: true };
    },
    overlayApi: overlay,
    MutationObserver: null,
    IntersectionObserver: null,
  });

  await controller.init();
  controller.setVisibility(pages[0], true);
  await controller.refreshVisibleAndLookahead();

  assert.deepEqual(
    calls.map((c) => [c.url.split("/").pop(), c.priority]),
    [
      ["p0.jpg", 0],
      ["p1.jpg", 1],
    ]
  );

  // Scroll closer to p3 -> p3 (priority 0) and p4 (priority 1) are queued; p5 remains deferred.
  calls.length = 0;
  controller.setVisibility(pages[0], false);
  controller.setVisibility(pages[3], true);
  await controller.refreshVisibleAndLookahead();

  assert.deepEqual(
    calls.map((c) => [c.url.split("/").pop(), c.priority]),
    [
      ["p3.jpg", 0],
      ["p4.jpg", 1],
    ]
  );
});

test("8. background Referer session rule is added before fetch and removed in finally", async () => {
  const events = [];
  let activeRuleId = null;
  const sampleBytes = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]);

  const fakeDnr = {
    async updateSessionRules({ addRules, removeRuleIds }) {
      if (addRules) {
        activeRuleId = addRules[0].id;
        events.push({ kind: "add", rule: addRules[0] });
      }
      if (removeRuleIds) {
        events.push({ kind: "remove", ids: removeRuleIds });
        activeRuleId = null;
      }
    },
  };

  const fakeFetch = async (url) => {
    events.push({ kind: "fetch", url, ruleWasActive: activeRuleId !== null });
    return {
      ok: true,
      status: 200,
      async arrayBuffer() {
        return sampleBytes.buffer;
      },
    };
  };

  const b64 = await background.fetchImageAsBase64(
    "https://img-cdn.example/ch1/001.png",
    "https://reader.example/manga/ch1",
    { fetch: fakeFetch, declarativeNetRequest: fakeDnr }
  );

  assert.equal(b64, Buffer.from(sampleBytes).toString("base64"));
  assert.equal(events[0].kind, "add");
  assert.equal(
    events[0].rule.action.requestHeaders[0].value,
    "https://reader.example/manga/ch1"
  );
  assert.equal(events[1].kind, "fetch");
  assert.equal(events[1].ruleWasActive, true);
  assert.equal(events[2].kind, "remove");
  assert.deepEqual(events[2].ids, [events[0].rule.id]);
  assert.equal(activeRuleId, null);

  // Verify cleanup still runs when fetch rejects or returns HTTP 403.
  events.length = 0;
  await assert.rejects(
    background.fetchImageAsBase64(
      "https://img-cdn.example/ch1/002.png",
      "https://reader.example/manga/ch1",
      {
        fetch: async () => ({ ok: false, status: 403 }),
        declarativeNetRequest: fakeDnr,
      }
    ),
    /403/
  );
  assert.equal(events.length, 2);
  assert.equal(events[0].kind, "add");
  assert.equal(events[1].kind, "remove");
  assert.equal(activeRuleId, null);
});

test("9. whole-word fitting prevents mid-word chopping for narrow vertical bubbles down to MIN_FONT_PX", () => {
  const cases = [
    { text: "Phantom Troupe", box: { x1: 120, y1: 60, x2: 175, y2: 320 } }, // 55 x 260
    { text: "That's for sure!!", box: { x1: 120, y1: 60, x2: 175, y2: 320 } }, // 55 x 260
    {
      text: "because of the mistakes that were made, but still mostly unchanged",
      box: { x1: 120, y1: 60, x2: 175, y2: 320 }, // 55 x 260
    },
    {
      text: "because of the mistakes that were made, but still mostly unchanged",
      box: { x1: 100, y1: 80, x2: 170, y2: 280 }, // 70 x 200
    },
    // Super-narrow box that MUST drop below 11px down to 7..10px so "unchanged" fits whole
    {
      text: "unchanged mistakes",
      box: { x1: 100, y1: 80, x2: 142, y2: 280 }, // 42 x 200
    },
  ];

  for (const { text, box } of cases) {
    const fit = overlay.computeFittedBox(text, box, { w: 900, h: 1300 });
    const boxW = fit.box.x2 - fit.box.x1;
    const boxH = fit.box.y2 - fit.box.y1;
    const { innerW } = overlay.computeInnerDimensions(boxW, boxH, fit.expanded);
    const words = text.trim().split(/\s+/);
    const maxWordW = Math.max(...words.map((w) => overlay.measureWordWidth(w, fit.fontSizePx, null)));
    assert.ok(
      maxWordW <= innerW,
      `Longest word in "${text}" (${maxWordW.toFixed(2)}px at ${fit.fontSizePx}px) exceeded innerW=${innerW}`
    );
    assert.ok(fit.fontSizePx >= overlay.MIN_FONT_PX);
  }

  // Verify super-narrow case actually shrunk below the old 11px floor
  const narrowFit = overlay.computeFittedBox(
    "unchanged mistakes",
    { x1: 100, y1: 80, x2: 142, y2: 280 },
    { w: 900, h: 1300 }
  );
  assert.ok(narrowFit.fontSizePx < 11, `Expected < 11px, got ${narrowFit.fontSizePx}`);
  assert.ok(narrowFit.fontSizePx >= 7, `Expected >= 7px, got ${narrowFit.fontSizePx}`);
});

test("10. bbox geometry utilization reclaims wasted horizontal space in tall narrow bubbles", () => {
  const safeBox = { x1: 130, y1: 70, x2: 185, y2: 330 }; // 55 x 260
  const fullBbox = { x1: 105, y1: 55, x2: 210, y2: 345 }; // 105 x 290
  const fit = overlay.computeFittedBox(
    "because of the mistakes that were made, but still mostly unchanged",
    safeBox,
    { w: 900, h: 1300 },
    { bbox: fullBbox }
  );

  const fittedW = fit.box.x2 - fit.box.x1;
  assert.ok(fittedW >= (fullBbox.x2 - fullBbox.x1) * 0.85, `Expected box width >= 85% of bbox, got ${fittedW}`);
  assert.ok(fit.fontSizePx >= 11, `Expected roomy font inside bbox, got ${fit.fontSizePx}`);
});

test("11. styles.css disables mid-word breaking on .mlt-bubble and removes conflicting px padding", () => {
  const cssPath = path.resolve(__dirname, "../extension/styles.css");
  const css = fs.readFileSync(cssPath, "utf8");
  const bubbleMatch = css.match(/\.mlt-bubble\s*\{([^}]*)\}/);
  assert.ok(bubbleMatch, ".mlt-bubble rule must exist in styles.css");
  const bubbleRule = bubbleMatch[1];

  assert.equal(/overflow-wrap\s*:\s*break-word/i.test(bubbleRule), false);
  assert.equal(/word-break\s*:\s*break-all/i.test(bubbleRule), false);
  assert.match(bubbleRule, /overflow-wrap\s*:\s*normal/i);
  assert.match(bubbleRule, /word-break\s*:\s*normal/i);
  assert.match(bubbleRule, /hyphens\s*:\s*none/i);
  assert.match(bubbleRule, /text-wrap\s*:\s*balance/i);

  const expandedMatch = css.match(/\.mlt-bubble-expanded\s*\{([^}]*)\}/);
  assert.ok(expandedMatch);
  assert.equal(/padding\s*:/i.test(expandedMatch[1]), false);
});

test("12. createOverlaySession mounts incrementally, skips empty text, updates duplicate IDs, and rescales", () => {
  const doc = createMockDocument();
  const win = createMockWindow(doc);
  const img = doc.createElement("img");
  img.width = 900;
  img.height = 1300;
  img.naturalWidth = 900;
  img.naturalHeight = 1300;
  doc.body.appendChild(img);

  const session = overlay.createOverlaySession(
    img,
    {
      lang: "ja",
      width: 900,
      height: 1300,
      bubbles: [
        {
          id: 0,
          bbox: [100, 80, 220, 260],
          safe_box: [108, 90, 212, 250],
          bg_color: "#FFFFFF",
          text_color: "#111111",
          is_dark: false,
          original: "ファントム旅団",
        },
        {
          id: 1,
          bbox: [300, 100, 460, 220],
          safe_box: [310, 110, 450, 210],
          bg_color: "#FFFFFF",
          text_color: "#111111",
          is_dark: false,
          original: "！？",
        },
        {
          id: 2,
          bbox: [500, 150, 680, 320],
          safe_box: [510, 160, 670, 310],
          bg_color: "#111111",
          text_color: "#FFFFFF",
          is_dark: true,
          original: "やめろ",
        },
      ],
    },
    { document: doc, window: win }
  );

  // Upfront on meta: zero masks or bubbles painted yet.
  assert.ok(session.overlayEl);
  assert.equal(session.overlayEl.querySelectorAll(".mlt-mask").length, 0);
  assert.equal(session.overlayEl.querySelectorAll(".mlt-bubble").length, 0);

  // Stream bubble 0.
  session.addBubble({ id: 0, translated: "Phantom Troupe!", final: true });
  assert.equal(session.overlayEl.querySelectorAll(".mlt-mask").length, 1);
  assert.equal(session.overlayEl.querySelectorAll(".mlt-bubble").length, 1);
  assert.equal(session.overlayEl.querySelector(".mlt-bubble-text").textContent, "Phantom Troupe!");

  // Empty / whitespace translation skips painting mask & bubble so original art stays clean.
  session.addBubble({ id: 1, translated: "   ", final: true });
  assert.equal(session.overlayEl.querySelectorAll(".mlt-mask").length, 1);
  assert.equal(session.overlayEl.querySelectorAll(".mlt-bubble").length, 1);

  // Stream bubble 2, then send a late retry update for duplicate id=2.
  session.addBubble({ id: 2, translated: "Stop!", final: false });
  assert.equal(session.overlayEl.querySelectorAll(".mlt-bubble").length, 2);

  session.addBubble({ id: 2, translated: "Stop right there!!", final: true });
  assert.equal(session.overlayEl.querySelectorAll(".mlt-mask").length, 2);
  assert.equal(session.overlayEl.querySelectorAll(".mlt-bubble").length, 2);
  const b2 = session.overlayEl
    .querySelectorAll(".mlt-bubble")
    .find((el) => el.dataset.bubbleId === "2");
  assert.equal(b2.querySelector(".mlt-bubble-text").textContent, "Stop right there!!");
  assert.equal(b2.querySelector(".mlt-tooltip-en").textContent, "Stop right there!!");

  // Proportional resize scaling (0.5x display width) scales fontSize down to floor 6px and padding proportionally.
  img.width = 450;
  img.height = 650;
  session.finish();
  for (const bEl of session.overlayEl.querySelectorAll(".mlt-bubble")) {
    const baseFont = Number(bEl.dataset.baseFontPx);
    const expectedFont = Math.max(6, baseFont * 0.5).toFixed(2);
    assert.equal(bEl.style.fontSize, `${expectedFont}px`);
    assert.match(bEl.style.padding, /^\d+(\.\d+)?px \d+(\.\d+)?px$/);
  }

  session.destroy();
  assert.equal(doc.querySelector(".mlt-overlay"), null);
});

test("13. fitMountedBubbleElement shrinks font down to 6px when DOM scrollWidth/scrollHeight overflows", () => {
  const bubbleEl = {
    style: { fontSize: "12.00px" },
    clientWidth: 44,
    clientHeight: 120,
    get scrollWidth() {
      const fs = parseFloat(this.style.fontSize);
      return fs > 8.5 ? 54 : 42;
    },
    get scrollHeight() {
      return 60;
    },
    querySelector() {
      return null;
    },
  };

  overlay.fitMountedBubbleElement(bubbleEl);
  assert.equal(parseFloat(bubbleEl.style.fontSize), 8.5);
});

test("14. neighbor-aware collision clamping prevents side-by-side and overlapping boxes from eating neighbor text", () => {
  const imgSize = { w: 900, h: 1300 };
  const leftSafe = { x1: 100, y1: 100, x2: 158, y2: 340 }; // 58 x 240
  const rightSafe = { x1: 166, y1: 100, x2: 224, y2: 340 }; // 58 x 240 (8px gap)
  const longText = "Wait, why are you running straight toward that monster right now?!";

  // Without neighbors, left bubble would expand rightward past x2=158 into rightSafe.
  const leftFit = overlay.computeFittedBox(longText, leftSafe, imgSize, {
    bbox: { x1: 88, y1: 90, x2: 190, y2: 350 },
    neighborBoxes: [rightSafe],
  });
  const rightFit = overlay.computeFittedBox(longText, rightSafe, imgSize, {
    bbox: { x1: 140, y1: 90, x2: 240, y2: 350 },
    neighborBoxes: [leftSafe],
  });

  assert.ok(
    leftFit.box.x2 <= rightFit.box.x1 - 2,
    `Expected left.x2 (${leftFit.box.x2}) <= right.x1 - 2 (${rightFit.box.x1 - 2})`
  );

  // Verify word still fits inside the clamped box down to MIN_FONT_PX.
  const leftInner = overlay.computeInnerDimensions(
    leftFit.box.x2 - leftFit.box.x1,
    leftFit.box.y2 - leftFit.box.y1,
    leftFit.expanded
  );
  assert.ok(
    overlay.measureWordWidth("monster", leftFit.fontSizePx, null) <= leftInner.innerW,
    "Text should shrink to fit cleanly inside clamped box"
  );

  // Also test renderOverlay with two adjacent side-by-side bubbles & slightly overlapping safeBoxes.
  const doc = createMockDocument();
  const win = createMockWindow(doc);
  const img = doc.createElement("img");
  img.width = 900;
  img.height = 1300;
  doc.body.appendChild(img);

  const handle = overlay.renderOverlay(
    img,
    {
      lang: "ja",
      width: 900,
      height: 1300,
      bubbles: [
        {
          id: 0,
          bbox: { x1: 90, y1: 90, x2: 185, y2: 350 },
          safe_box: { x1: 100, y1: 100, x2: 164, y2: 340 },
          bg_color: "#FFFFFF",
          text_color: "#111111",
          is_dark: false,
          original: "ちょっと待ってよ！",
          translated: longText,
        },
        {
          id: 1,
          bbox: { x1: 145, y1: 90, x2: 235, y2: 350 },
          safe_box: { x1: 160, y1: 100, x2: 224, y2: 340 }, // 4px overlap in original safe_box
          bg_color: "#FFFFFF",
          text_color: "#111111",
          is_dark: false,
          original: "危険すぎるだろ！",
          translated: longText,
        },
      ],
    },
    { document: doc, window: win }
  );

  const bubbles = handle.element.querySelectorAll(".mlt-bubble");
  assert.equal(bubbles.length, 2);
  const b0Left = (parseFloat(bubbles[0].style.left) / 100) * 900;
  const b0Width = (parseFloat(bubbles[0].style.width) / 100) * 900;
  const b1Left = (parseFloat(bubbles[1].style.left) / 100) * 900;
  assert.ok(
    b0Left + b0Width <= b1Left + 0.01,
    `Rendered bubbles overlapped: b0.x2=${(b0Left + b0Width).toFixed(2)} > b1.x1=${b1Left.toFixed(2)}`
  );
  handle.destroy();
});

test("15. body-level hover tooltip portal opens on mouseenter, flips near top edge, and closes cleanly", () => {
  const doc = createMockDocument();
  const win = createMockWindow(doc);
  win.innerWidth = 1024;
  win.innerHeight = 768;

  const img = doc.createElement("img");
  img.width = 800;
  img.height = 1200;
  doc.body.appendChild(img);

  const handle = overlay.renderOverlay(
    img,
    {
      lang: "ja",
      width: 800,
      height: 1200,
      bubbles: [
        {
          id: 0,
          bbox: { x1: 100, y1: 100, x2: 300, y2: 240 },
          safe_box: { x1: 110, y1: 110, x2: 290, y2: 230 },
          bg_color: "#FFFFFF",
          text_color: "#111111",
          is_dark: false,
          original: "どうしたんだ！？",
          translated: "What happened?!",
        },
      ],
    },
    { document: doc, window: win }
  );

  const hitEl = handle.element.querySelector(".mlt-bubble-hit");
  let mockRect = { left: 300, top: 200, width: 120, height: 80 };
  hitEl.getBoundingClientRect = () => mockRect;

  hitEl.dispatchEvent({ type: "mouseenter" });
  const portal = doc.body.querySelector(".mlt-tooltip-portal");
  assert.ok(portal, "Expected .mlt-tooltip-portal mounted on doc.body");
  assert.equal(portal.style.display, "block");
  assert.equal(portal.querySelector(".mlt-tooltip-portal-en").textContent, "What happened?!");
  assert.equal(portal.querySelector(".mlt-tooltip-portal-orig").textContent, "どうしたんだ！？");
  assert.equal(portal.dataset.placement, "top");
  assert.ok(parseFloat(portal.style.top) < mockRect.top, "Portal should sit above bubble");

  // Near top edge (top < 60): portal must flip below the bubble.
  mockRect = { left: 10, top: 24, width: 100, height: 40 };
  hitEl.dispatchEvent({ type: "mouseenter" });
  assert.equal(portal.dataset.placement, "bottom");
  assert.ok(
    parseFloat(portal.style.top) >= mockRect.top + mockRect.height,
    `Portal should flip below bubble when top < 60 (got top=${portal.style.top})`
  );
  assert.ok(parseFloat(portal.style.left) >= 8, "Portal left must clamp inside [8, vw - 8]");

  // Hides on mouseleave.
  hitEl.dispatchEvent({ type: "mouseleave" });
  assert.equal(portal.style.display, "none");

  // Opens on focusin and hides on Alt peek.
  hitEl.dispatchEvent({ type: "focusin" });
  assert.equal(portal.style.display, "block");
  overlay.setPeekOriginal(true, doc);
  assert.equal(portal.style.display, "none");
  overlay.setPeekOriginal(false, doc);

  // Opens again, then hides on window scroll.
  hitEl.dispatchEvent({ type: "mouseenter" });
  assert.equal(portal.style.display, "block");
  win.dispatchEvent({ type: "scroll" });
  assert.equal(portal.style.display, "none");

  // Opens again, then hides on clearOverlay.
  hitEl.dispatchEvent({ type: "mouseenter" });
  assert.equal(portal.style.display, "block");
  overlay.clearOverlay(img);
  assert.equal(portal.style.display, "none");
});

test("16. floating dialogue & SFX render borderless translucent pill without opaque white mask", () => {
  const doc = createMockDocument();
  const win = createMockWindow(doc);
  const img = doc.createElement("img");
  img.width = 800;
  img.height = 1200;
  doc.body.appendChild(img);

  const handle = overlay.renderOverlay(
    img,
    {
      lang: "ja",
      width: 800,
      height: 1200,
      bubbles: [
        {
          id: 0,
          region_type: "floating",
          bbox: { x1: 80, y1: 120, x2: 260, y2: 200 },
          safe_box: { x1: 85, y1: 125, x2: 255, y2: 195 },
          bg_color: "#FFFFFF",
          text_color: "#111111",
          is_dark: false,
          original: "ザワ…",
          translated: "*RUSTLE*",
        },
        {
          id: 1,
          region_type: "floating",
          bbox: { x1: 380, y1: 220, x2: 560, y2: 300 },
          safe_box: { x1: 385, y1: 225, x2: 555, y2: 295 },
          bg_color: "#111111",
          text_color: "#FFFFFF",
          is_dark: true,
          original: "ドクン",
          translated: "*THUMP*",
        },
      ],
    },
    { document: doc, window: win }
  );

  const masks = handle.element.querySelectorAll(".mlt-mask");
  const bubbles = handle.element.querySelectorAll(".mlt-bubble");
  assert.equal(masks.length, 2);
  assert.equal(bubbles.length, 2);

  for (const m of masks) {
    assert.ok(m.classList.contains("mlt-mask-floating"));
    assert.equal(m.style.display, "none");
    assert.equal(m.style.backgroundColor, "transparent");
  }

  assert.ok(bubbles[0].classList.contains("mlt-bubble-floating"));
  assert.match(bubbles[0].style.backgroundColor, /^rgba\(255, 255, 255, 0\.86\)$/);

  assert.ok(bubbles[1].classList.contains("mlt-bubble-floating"));
  assert.ok(bubbles[1].classList.contains("mlt-bubble-dark"));
  assert.match(bubbles[1].style.backgroundColor, /^rgba\(17, 17, 17, 0\.84\)$/);

  const cssPath = path.resolve(__dirname, "../extension/styles.css");
  const css = fs.readFileSync(cssPath, "utf8");
  assert.match(css, /\.mlt-mask-floating\s*\{[^}]*display\s*:\s*none/i);
  assert.match(css, /\.mlt-bubble-floating\s*\{[^}]*backdrop-filter\s*:/i);
  assert.match(css, /\.mlt-tooltip-portal\s*\{[^}]*position\s*:\s*fixed/i);
  assert.match(css, /\.mlt-tooltip-portal\s*\{[^}]*z-index\s*:\s*2147483647/i);
  assert.match(css, /\.mlt-bubble:hover,\s*\.mlt-bubble:focus-within\s*\{[^}]*z-index\s*:\s*100/i);

  handle.destroy();
});

test("17. streaming addBubble without upfront meta re-clamps previously expanded neighbors and handles null body", () => {
  const doc = createMockDocument();
  const win = createMockWindow(doc);
  const img = doc.createElement("img");
  img.width = 900;
  img.height = 1300;
  doc.body.appendChild(img);

  const session = overlay.createOverlaySession(
    img,
    { lang: "ja", width: 900, height: 1300 },
    { document: doc, window: win }
  );

  // Bubble 0 streams first with no upfront meta and expands freely.
  const b0 = session.addBubble({
    id: 0,
    bbox: [90, 90, 190, 350],
    safe_box: [100, 100, 160, 340],
    translated: "Wait, why are you running straight toward that monster?!",
  });
  const initialB0Width = (parseFloat(b0.style.width) / 100) * 900;
  assert.ok(initialB0Width > 60);

  // Bubble 1 streams immediately to the right -> b0 must be re-clamped so boxes do not overlap.
  const b1 = session.addBubble({
    id: 1,
    bbox: [150, 90, 240, 350],
    safe_box: [168, 100, 228, 340],
    translated: "Because there is no time left to hesitate!",
  });

  const b0x1 = (parseFloat(b0.style.left) / 100) * 900;
  const b0x2 = b0x1 + (parseFloat(b0.style.width) / 100) * 900;
  const b1x1 = (parseFloat(b1.style.left) / 100) * 900;
  assert.ok(
    b0x2 <= b1x1 - 1.5,
    `Streaming collision not resolved: b0x2=${b0x2.toFixed(2)}, b1x1=${b1x1.toFixed(2)}`
  );

  // Minimal DOM without doc.body: tooltip hover must be null-safe and never throw.
  const minDoc = createMockDocument();
  minDoc.body = null;
  const hitEl = b0.querySelector(".mlt-bubble-hit");
  assert.doesNotThrow(() => {
    overlay.setPeekOriginal(true, minDoc);
    overlay.hideTooltipPortal(minDoc);
    hitEl.dispatchEvent({ type: "mouseenter" });
    hitEl.dispatchEvent({ type: "mouseleave" });
  });

  session.destroy();
});
