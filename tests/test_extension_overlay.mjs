import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

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

test("7. look-ahead priority ordering sends visible=0, next 3=1, and defers distant pages", async () => {
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
      ["p2.jpg", 1],
      ["p3.jpg", 1],
    ]
  );

  // Scroll closer to p3 -> p4 and p5 are now queued.
  calls.length = 0;
  controller.setVisibility(pages[0], false);
  controller.setVisibility(pages[3], true);
  await controller.refreshVisibleAndLookahead();

  const urlsAfterScroll = calls.map((c) => c.url.split("/").pop());
  assert.ok(urlsAfterScroll.includes("p4.jpg"));
  assert.ok(urlsAfterScroll.includes("p5.jpg"));
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
