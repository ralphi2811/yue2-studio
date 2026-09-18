/* YuE2 Studio — comportements côté client (vanilla). HTMX gère les échanges serveur. */
(function () {
  "use strict";

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const TOKENS_PER_SECOND = 25;   // ≈ tokens sémantiques par seconde d'audio (observé : 1652 tokens → 66 s)
  const BARS_PER_LINE = 2;        // calibré sur les partitions planifiées : 4 lignes → 17 mesures, 8 lignes → 24 mesures
  const OVERHEAD_BARS = 9;        // intro + outro instrumentaux ajoutés par le planificateur
  const EMPTY_SECTION_BARS = 4;   // [Intro]/[Interlude]/[Outro] sans paroles
  const DEFAULT_BPM = 90;

  const fmtDur = s => { s = Math.max(0, Math.round(s)); const m = Math.floor(s / 60), r = s % 60; return m ? `${m} min ${String(r).padStart(2, "0")} s` : `${r} s`; };

  function parseBpm(style) {
    const m = /(\d{2,3})\s*(?:bpm|BPM)/.exec(style || "");
    const v = m ? parseInt(m[1], 10) : NaN;
    return v >= 30 && v <= 300 ? v : null;
  }

  // Durée depuis les paroles : mesures estimées × temps par mesure (4/4 supposé).
  function estimateFromLyrics(lyrics, style) {
    const lines = (lyrics || "").split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const tags = lines.filter(l => /^\[.+\]$/.test(l));
    const sung = lines.length - tags.length;
    let empty = 0;
    for (let i = 0; i < lines.length; i++) {
      if (/^\[.+\]$/.test(lines[i]) && (i + 1 >= lines.length || /^\[.+\]$/.test(lines[i + 1]))) empty++;
    }
    const bpm = parseBpm(style);
    const bars = OVERHEAD_BARS + BARS_PER_LINE * sung + EMPTY_SECTION_BARS * empty;
    return { seconds: bars * 4 * 60 / (bpm || DEFAULT_BPM), bars, bpm, sung, sections: tags.length, empty };
  }

  // Durée exacte depuis une partition ABC native : Σ mesures × (M / unité de Q) × 60 / BPM, sur la voix Vocal.
  function estimateFromAbc(text) {
    let meter = [4, 4], beat = [1, 4], bpm = null, voice = null;
    const bars = {};
    for (const raw of (text || "").split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith("%")) continue;
      const f = /^([A-Za-z]):\s*(.*)$/.exec(line);
      if (f) {
        const [, k, v] = f;
        if (k === "M") { const m = /(\d+)\s*\/\s*(\d+)/.exec(v); if (m) meter = [+m[1], +m[2]]; }
        else if (k === "Q") { const q = /(?:(\d+)\s*\/\s*(\d+)\s*=\s*)?(\d+)/.exec(v); if (q) { bpm = +q[3]; beat = q[1] ? [+q[1], +q[2]] : [1, 4]; } }
        else if (k === "V") { voice = v.split(/\s+/)[0]; }
        continue;
      }
      const secPerBar = (meter[0] / meter[1]) / (beat[0] / beat[1]) * 60 / (bpm || DEFAULT_BPM);
      for (const seg of line.split("|")) {
        const sgm = seg.replace(/"[^"]*"/g, "").trim();
        if (!sgm) continue;
        const z = /^Z(\d*)$/.exec(sgm);
        const n = z ? parseInt(z[1] || "1", 10) : 1;
        const b = bars[voice || "_"] || (bars[voice || "_"] = { bars: 0, seconds: 0 });
        b.bars += n; b.seconds += n * secPerBar;
      }
    }
    const key = bars.Vocal ? "Vocal" : Object.keys(bars)[0];
    if (!key) return null;
    return { seconds: bars[key].seconds, bars: bars[key].bars, bpm, meter: `${meter[0]}/${meter[1]}`, voice: key };
  }

  function updateDurationBadge(form) {
    const badge = $("#duration-badge", form);
    if (!badge) return;
    const cot = $('select[name="cot"]', form)?.value;
    const abcText = $('textarea[name="abc"]', form)?.value.trim();
    const max = parseInt($('input[name="semantic.max_tokens"]', form)?.value || "0", 10);
    const cap = max ? max / TOKENS_PER_SECOND : Infinity;
    let est = null, label = "", title = "";
    if (abcText && cot !== "off") {
      est = estimateFromAbc(abcText);
      if (est) { label = `Durée ≈ ${fmtDur(est.seconds)} (partition : ${est.bars} mesures)`; title = "Calcul exact depuis la partition fournie : mesures × temps par mesure ÷ BPM. L'audio réel finit sur la dernière note, souvent 2 à 4 s plus tôt."; }
    } else {
      est = estimateFromLyrics($('textarea[name="lyrics"]', form)?.value, $('textarea[name="style"]', form)?.value);
      label = `Durée estimée ≈ ${fmtDur(est.seconds)}`;
      title = `Estimation grossière : ${est.sung} lignes chantées × ${BARS_PER_LINE} mesures + ${OVERHEAD_BARS} mesures d'intro/outro` + (est.empty ? ` + ${est.empty} section(s) instrumentale(s) × ${EMPTY_SECTION_BARS} mesures` : "") + ` = ${est.bars} mesures en 4/4 à ${est.bpm || DEFAULT_BPM + " (BPM non précisé dans le style)"} BPM. Le modèle reste libre : comptez ±25 %.`;
    }
    badge.classList.remove("warn");
    const maxDur = parseInt($('input[name="max_duration"]', form)?.value || "0", 10);
    if (maxDur) { label += ` · limite ${fmtDur(maxDur)}`; title += ` Durée maximale : les tokens sémantiques sont bornés à ${maxDur} s ; si la partition planifiée dépasse ${fmtDur(maxDur * 1.3)}, elle est raccourcie ou le job s'arrête selon « Partition trop longue ».`; }
    if (est && maxDur && est.seconds > maxDur * 1.3) { badge.classList.add("warn"); title += " L'estimation dépasse déjà la limite : la partition sera probablement raccourcie ou le job annulé après la planification."; }
    if (est && est.seconds > cap) { label += ` · plafonné à ${fmtDur(cap)} par max_tokens`; badge.classList.add("warn"); title += ` Le plafond max_tokens (${max}) coupera la génération avant la fin.`; }
    badge.textContent = label; badge.title = title;
  }

  // ------------------------------------------------------------------ dialogues stylés (remplacent confirm/prompt/alert)
  function openDialog(id, { title, message, okLabel = "OK", cancelLabel = "Annuler", danger = false, input = null }) {
    return new Promise(resolve => {
      const dlg = document.getElementById(id);
      if (!dlg) { resolve(input !== null ? window.prompt(message, input) : window.confirm(message)); return; }
      const q = r => dlg.querySelector(`[data-role="${r}"]`);
      q("title").textContent = title;
      q("message").textContent = message || "";
      q("message").hidden = !message;
      q("ok").textContent = okLabel;
      q("ok").classList.toggle("danger", danger);
      q("ok").classList.toggle("primary", !danger);
      const cancel = q("cancel");
      cancel.hidden = cancelLabel === null;
      if (cancelLabel) cancel.textContent = cancelLabel;
      const field = q("input");
      if (field) { field.value = input || ""; field.placeholder = dlg.dataset.placeholder || ""; }
      const done = value => { cleanup(); if (dlg.open) dlg.close(); resolve(value); };
      const onOk = () => done(field ? field.value.trim() : true);
      const onCancel = () => done(field ? null : false);
      const onKey = ev => { if (ev.key === "Enter" && field && !ev.shiftKey) { ev.preventDefault(); onOk(); } if (ev.key === "Enter" && !field) { ev.preventDefault(); onOk(); } };
      const onClose = () => done(field ? null : false);
      const onBackdrop = ev => { if (ev.target === dlg) onCancel(); };
      function cleanup() {
        q("ok").removeEventListener("click", onOk); cancel.removeEventListener("click", onCancel);
        dlg.removeEventListener("keydown", onKey); dlg.removeEventListener("close", onClose); dlg.removeEventListener("click", onBackdrop);
      }
      q("ok").addEventListener("click", onOk); cancel.addEventListener("click", onCancel);
      dlg.addEventListener("keydown", onKey); dlg.addEventListener("close", onClose); dlg.addEventListener("click", onBackdrop);
      dlg.showModal();
      (field || q("ok")).focus();
    });
  }
  const ui = {
    confirm: (message, opts = {}) => openDialog("confirm-dialog", { title: opts.title || "Confirmer", message, okLabel: opts.okLabel || "Confirmer", danger: !!opts.danger }),
    prompt: (message, opts = {}) => { const d = document.getElementById("prompt-dialog"); if (d) d.dataset.placeholder = opts.placeholder || ""; return openDialog("prompt-dialog", { title: opts.title || "Instruction", message, okLabel: opts.okLabel || "Valider", input: opts.value || "" }); },
    alert: (message, opts = {}) => openDialog("confirm-dialog", { title: opts.title || "Information", message, okLabel: "Fermer", cancelLabel: null }),
  };

  // hx-confirm → dialogue stylé. Les suppressions sont marquées en rouge.
  document.addEventListener("htmx:confirm", ev => {
    if (!ev.detail.question) return;
    ev.preventDefault();
    const elt = ev.detail.elt;
    const danger = elt.hasAttribute("hx-delete") || /supprimer/i.test(ev.detail.question);
    const okLabel = danger ? "Supprimer" : (elt.getAttribute("data-confirm-ok") || "Confirmer");
    ui.confirm(ev.detail.question, { title: danger ? "Suppression" : "Confirmer", okLabel, danger }).then(ok => { if (ok) ev.detail.issueRequest(true); });
  });

  // ------------------------------------------------------------------ onglets
  function activateTab(btn) {
    const nav = btn.closest(".tabs");
    let scope = nav.parentElement;
    while (scope && !scope.querySelector(":scope > .tab-pane")) scope = scope.parentElement;
    if (!scope) return;
    $$(".tab", nav).forEach(t => t.classList.toggle("active", t === btn));
    $$(":scope > .tab-pane", scope).forEach(p => p.classList.toggle("active", p.dataset.pane === btn.dataset.tab));
    if (scope.id === "form-panel" || scope.closest("#form-panel")) {
      try { localStorage.setItem("yue2.formTab", btn.dataset.tab); } catch (_) {}
    }
  }

  function restoreFormTab(root) {
    let saved = null;
    try { saved = localStorage.getItem("yue2.formTab"); } catch (_) {}
    const btn = saved && $(`#job-form .tab[data-tab="${saved}"]`, root || document);
    if (btn) activateTab(btn);
  }

  // ------------------------------------------------------------------ aide
  function applyGlobalHelp() {
    const on = $("#toggle-help")?.checked;
    document.body.classList.toggle("show-all-help", !!on);
    try { localStorage.setItem("yue2.help", on ? "1" : "0"); } catch (_) {}
  }

  // ------------------------------------------------------------------ formulaire
  function syncModeRules(form) {
    if (!form) return;
    const cot = $('select[name="cot"]', form)?.value;
    const abc = $('textarea[name="abc"]', form);
    const abcGroup = $("#group-abc-sampling", form);
    const planBtn = $("#btn-plan", form);
    const hasAbc = !!(abc && abc.value.trim());
    if (abc) {
      abc.disabled = cot === "off";
      const status = $("#abc-status", form);
      if (status) {
        let txt = cot === "off" ? "Désactivé en mode off." : (hasAbc ? "Partition fournie : la planification est court-circuitée." : "");
        if (hasAbc && cot !== "off") { const e = estimateFromAbc(abc.value); if (e) txt += ` ${e.bars} mesures en ${e.meter} à ${e.bpm || DEFAULT_BPM} BPM → ≈ ${fmtDur(e.seconds)}.`; }
        status.textContent = txt;
      }
    }
    updateDurationBadge(form);
    if (abcGroup) abcGroup.classList.toggle("disabled", cot === "off" || hasAbc);
    if (planBtn) planBtn.disabled = cot === "off" || hasAbc;
    const scoreTab = $('.tab[data-tab="score"]', form);
    if (scoreTab) scoreTab.textContent = hasAbc ? "Partition ●" : "Partition";
  }

  // Miroir simplifié de studio/lora.py::instrumental_lyrics, pour prévisualiser la structure envoyée à la LoRA.
  const LORA_TAGS = ["intro", "verse", "pre-chorus", "chorus", "bridge", "outro"];
  const LORA_ALIASES = { prechorus: "pre-chorus", "pre chorus": "pre-chorus", refrain: "chorus", hook: "chorus", drop: "chorus",
    interlude: "bridge", solo: "bridge", break: "bridge", instrumental: "bridge", end: "outro", ending: "outro", coda: "outro", couplet: "verse", pont: "bridge", final: "outro" };
  function instrumentalPlan(lyrics) {
    const out = [];
    for (const line of lyrics.split(/\r?\n/)) {
      const m = /^\s*\[\s*([^\]]+?)\s*(?:(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2}))?\s*\]\s*$/.exec(line);
      if (!m) continue;
      let name = m[1].trim().toLowerCase().replace(/[\s_]+/g, " ").replace(/\s*\d+\s*$/, "");
      let tag = LORA_TAGS.includes(name) ? name : LORA_ALIASES[name];
      if (!tag) { const first = name.split(" ")[0]; tag = LORA_TAGS.includes(first) ? first : LORA_ALIASES[first]; }
      if (!tag) continue;
      out.push(m[2] && m[3] ? `[${tag} ${m[2]}-${m[3]}]` : `[${tag}]`);
    }
    if (out.length && out.every(t => t === "[bridge]") && /instrumental/i.test(lyrics)) return "[instrumental]";
    return out.length ? out.join("\n") : "[instrumental]";
  }
  function updateInstrumentalNote(form) {
    const box = $('input[name="instrumental"]', form), note = $("#instrumental-note", form);
    if (!box || !note) return;
    if (!box.checked) { note.hidden = true; return; }
    const cot = $('select[name="cot"]', form);
    if (cot && cot.value !== "full") cot.value = "full";
    const plan = instrumentalPlan($('textarea[name="lyrics"]', form)?.value || "");
    note.innerHTML = `<strong>Mode instrumental (LoRA, expérimental).</strong> Mode de composition « full » imposé. Les paroles ne servent que de structure : le modèle recevra <code class="mono">${escapeHtml(plan).replace(/\n/g, " ")}</code>. Un horodatage « [verse 0:15-0:45] » guide les proportions. La voix peut encore apparaître ; licence de la LoRA : CC BY-NC (non commercial).`;
    note.hidden = false;
  }

  function updateLyricsStats(form) {
    const ta = $('textarea[name="lyrics"]', form);
    const out = $("#lyrics-stats", form);
    updateInstrumentalNote(form);
    if (!ta || !out) return;
    const lines = ta.value.split(/\r?\n/).filter(l => l.trim());
    const sections = lines.filter(l => /^\s*\[.+\]\s*$/.test(l)).length;
    const sung = lines.length - sections;
    const est = estimateFromLyrics(ta.value, $('textarea[name="style"]', form)?.value);
    out.textContent = `${sections} section${sections > 1 ? "s" : ""} · ${sung} ligne${sung > 1 ? "s" : ""} · ≈ ${fmtDur(est.seconds)}` + (est.bpm ? ` à ${est.bpm} BPM` : " (BPM non précisé)");
    const warn = $("#lyrics-warning", form);
    if (warn) {
      const maxDur = $('input[name="max_duration"]', form)?.value;
      const instrumental = sung === 0 && lines.length > 0;
      warn.hidden = !instrumental;
      const overflow = $('select[name="plan_overflow"]', form)?.value || "auto";
      const overflowText = overflow === "stop" ? "annulera le job (partition conservée)" : "raccourcira la partition planifiée à des sections entières";
      if (instrumental) warn.innerHTML = maxDur
        ? `<strong>Morceau instrumental</strong> : aucune ligne chantée, la durée n'est ancrée par rien. La durée maximale de ${fmtDur(parseInt(maxDur, 10))} bornera la génération et ${overflowText} si la partition planifiée la dépasse de plus de 30 %.`
        : `<strong>Morceau instrumental</strong> : aucune ligne chantée, la durée n'est ancrée par rien et le modèle peut dériver pendant des minutes. Renseignez une <em>durée maximale</em> ci-dessous, gardez 2 ou 3 balises, ou fournissez une partition pour fixer exactement la longueur.`;
    }
    updateDurationBadge(form);
  }

  function updateDurationEstimate(form) {
    const max = parseInt($('input[name="semantic.max_tokens"]', form)?.value || "0", 10);
    const min = parseInt($('input[name="semantic.min_tokens"]', form)?.value || "0", 10);
    const out = $("#duration-estimate", form);
    if (!out) return;
    out.textContent = max ? `≈ ${fmtDur(min / TOKENS_PER_SECOND)} à ${fmtDur(max / TOKENS_PER_SECOND)} d'audio` : "";
    updateDurationBadge(form);
  }

  function insertAtCursor(ta, text) {
    const start = ta.selectionStart ?? ta.value.length, end = ta.selectionEnd ?? start;
    const before = ta.value.slice(0, start), after = ta.value.slice(end);
    const prefix = before && !before.endsWith("\n") ? (before.endsWith("\n\n") ? "" : "\n") : "";
    const insert = `${prefix}${text}\n`;
    ta.value = before + insert + after;
    ta.selectionStart = ta.selectionEnd = (before + insert).length;
    ta.focus();
    ta.dispatchEvent(new Event("input", { bubbles: true }));
  }

  // ------------------------------------------------------------------ rendu de partitions (abcjs)
  // Deux modes : "real" = taille réelle avec défilement horizontal ; "fit" = ajusté à la largeur du conteneur.
  const REAL_STAFF_WIDTH = 1000;
  function abcZoomMode() { try { return localStorage.getItem("yue2.abcZoom") || "real"; } catch (_) { return "real"; } }
  function setAbcZoomMode(mode) { try { localStorage.setItem("yue2.abcZoom", mode); } catch (_) {} }

  function renderAbcInto(el, text, mode) {
    if (!el || !window.ABCJS) return;
    el.removeAttribute("style");        // abcjs laisse un style inline en mode responsive
    el.innerHTML = "";
    text = (text || "").trim();
    if (!text) return;
    try {
      const opts = mode === "fit"
        ? { responsive: "resize", add_classes: true }
        : { staffwidth: REAL_STAFF_WIDTH, add_classes: true };
      ABCJS.renderAbc(el, text, opts);
      if (!el.querySelector("svg")) el.innerHTML = '<p class="abc-error">Aperçu impossible pour ce texte.</p>';
      if (mode !== "fit") { el.style.overflow = "auto"; el.style.height = ""; }   // abcjs force overflow:hidden + hauteur fixe
    } catch (e) {
      el.innerHTML = `<p class="abc-error">Rendu abcjs impossible : ${e.message || e}</p>`;
    }
    el.dataset.mode = mode;
  }

  function syncZoomButtons(scope) {
    const mode = abcZoomMode();
    $$("[data-action='abc-zoom']", scope || document).forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
  }

  let abcTimer = null;
  function renderAbcPreview(form) {
    const ta = $('textarea[name="abc"]', form);
    const box = $("#abc-preview", form);
    if (!ta || !box) return;
    clearTimeout(abcTimer);
    abcTimer = setTimeout(() => { renderAbcInto(box, ta.value, abcZoomMode()); syncZoomButtons(form); }, 300);
  }

  function renderAbcBlocks(root) {
    $$(".abc-render[data-abc]", root || document).forEach(el => renderAbcInto(el, el.dataset.abc, abcZoomMode()));
    syncZoomButtons(root);
  }

  function rerenderAllAbc() {
    const mode = abcZoomMode();
    $$(".abc-render[data-abc]").forEach(el => renderAbcInto(el, el.dataset.abc, mode));
    const form = $("#job-form");
    if (form) renderAbcInto($("#abc-preview", form), $('textarea[name="abc"]', form)?.value, mode);
    syncZoomButtons(document);
  }

  function resetAdvanced(form) {
    $$(".field-advanced input, .field-advanced select", form).forEach(el => {
      const def = el.closest(".field")?.querySelector(".default")?.textContent.replace("déf. ", "");
      if (def !== undefined && el.type === "number") { el.value = def; el.dispatchEvent(new Event("input", { bubbles: true })); }
    });
  }

  function initForm(root) {
    const form = $("#job-form", root || document);
    if (!form) return;
    restoreFormTab(form.parentElement);
    if (form.dataset.openTab) { const b = $(`.tab[data-tab="${form.dataset.openTab}"]`, form); if (b) activateTab(b); }
    syncModeRules(form);
    updateLyricsStats(form);
    updateDurationEstimate(form);
    renderAbcPreview(form);
    $$(".slider", form).forEach(sl => {
      const input = document.getElementById(sl.dataset.for);
      if (!input) return;
      sl.addEventListener("input", () => { input.value = sl.value; input.dispatchEvent(new Event("change", { bubbles: true })); });
      input.addEventListener("input", () => { sl.value = input.value; });
    });
  }

  // ------------------------------------------------------------------ outils de partition / lot
  async function stripChords(form) {
    const ta = $('textarea[name="abc"]', form);
    const check = $("#abc-check", form);
    if (!ta || !ta.value.trim()) { if (check) check.innerHTML = '<div class="alert error small">Aucune partition à traiter.</div>'; return; }
    const body = new FormData();
    body.append("abc", ta.value);
    body.append("keep_voice", $('select[name="keep_voice"]', form)?.value || "both");
    try {
      const res = await fetch("/abc/strip", { method: "POST", body });
      const data = await res.json();
      if (!data.ok) { check.innerHTML = `<div class="alert error small"><strong>Retrait refusé</strong> : ${escapeHtml(data.error)}</div>`; return; }
      ta.value = data.abc;
      ta.dispatchEvent(new Event("input", { bubbles: true }));
      const cot = $('select[name="cot"]', form);
      let hint = "";
      if (cot && cot.value === "full") { cot.value = "melody"; cot.dispatchEvent(new Event("change", { bubbles: true })); hint = " Mode passé en <em>melody</em> (partition sans accords)."; }
      check.innerHTML = `<div class="alert ok small"><strong>Accords retirés</strong>, mélodie vérifiée identique.${hint}</div>`;
    } catch (e) {
      check.innerHTML = `<div class="alert error small">Erreur réseau : ${escapeHtml(String(e))}</div>`;
    }
  }

  async function submitBatch(form) {
    const file = $("#batch-file", form)?.files[0];
    const out = $("#batch-result", form);
    if (!file) return;
    const body = new FormData(form);          // hérite du formulaire courant (style, échantillonnage…)
    body.delete("kind");
    body.append("file", file, file.name);
    out.innerHTML = '<p class="muted small">Import en cours…</p>';
    try {
      const res = await fetch("/jobs/batch", { method: "POST", body });
      out.innerHTML = await res.text();
      if (res.ok) { document.body.dispatchEvent(new Event("queue-changed")); $("#batch-file", form).value = ""; $("#batch-filename", form).textContent = ""; $("#batch-submit", form).disabled = true; }
    } catch (e) {
      out.innerHTML = `<div class="alert error small">Erreur réseau : ${escapeHtml(String(e))}</div>`;
    }
  }

  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

  // ------------------------------------------------------------------ modes Composer / Écouter
  function setMode(mode, persist = true) {
    const layout = $("#layout");
    if (!layout || (mode !== "compose" && mode !== "browse")) return;
    layout.dataset.mode = mode;
    $$(".mode-tab").forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
    if (persist) { try { localStorage.setItem("yue2.mode", mode); } catch (_) {} }
  }

  // ------------------------------------------------------------------ assistant LLM
  // Fait suivre à l'assistant le morceau ouvert (job_id) ou le projet du formulaire (project_id),
  // sans écraser un message en cours de frappe.
  function syncAssistant(query, want) {
    const body = $("#assistant-body");
    if (!body || !window.htmx) return;
    const cur = $(".assistant", body);
    if (cur && want.project && cur.dataset.project === want.project) return;
    if (cur && !want.project && want.job && cur.dataset.job === want.job) return;
    const ta = $("textarea", body);
    if (ta && ta.value.trim()) return;
    htmx.ajax("GET", `/assistant/panel?${query}`, { target: "#assistant-body", swap: "innerHTML" });
  }

  function setAssistantOpen(open) {
    const drawer = $("#assistant-drawer");
    if (!drawer) return;
    drawer.hidden = !open;
    document.body.classList.toggle("assistant-open", open);
    try { localStorage.setItem("yue2.assistant", open ? "1" : "0"); } catch (_) {}
    if (open) { const ta = $("#assistant-body textarea"); if (ta && !ta.disabled) ta.focus(); }
  }
  function scrollChat() { const chat = $('#assistant-body [data-role="chat"]'); if (chat) chat.scrollTop = chat.scrollHeight; }

  async function quickAction(btn) {
    const form = btn.closest("form");
    const kind = btn.dataset.kind;
    const field = $(`textarea[name="${kind}"]`, form);
    const why = $("#quick-why", form);
    let instruction = "";
    if (kind === "lyrics") {
      instruction = await ui.prompt("Instruction facultative pour le LLM. Exemples : « refrain plus court », « plus d'images », « rimes en -ir ». Laissez vide pour une simple amélioration.",
                                    { title: "Retravailler les paroles", okLabel: "Retravailler", placeholder: "ex. refrain plus court" });
      if (instruction === null) return;
    }
    const body = new FormData();
    body.append("action", kind);
    body.append("style", $('textarea[name="style"]', form).value);
    body.append("lyrics", $('textarea[name="lyrics"]', form).value);
    body.append("instruction", instruction || "");
    const previous = field.value;
    btn.disabled = true; btn.classList.add("busy"); const label = btn.textContent; btn.textContent = "✨ …";
    try {
      const res = await fetch("/assistant/quick", { method: "POST", body });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || res.statusText);
      field.value = data.value;
      field.dispatchEvent(new Event("input", { bubbles: true }));
      why.hidden = false;
      why.innerHTML = `<div class="alert ok small"><strong>${kind === "style" ? "Style amélioré" : "Paroles retravaillées"}</strong>${data.why ? " · " + escapeHtml(data.why) : ""} <button type="button" class="btn ghost tiny" data-action="quick-undo">annuler</button></div>`;
      why.dataset.field = kind; why.dataset.previous = previous;
    } catch (e) {
      why.hidden = false;
      why.innerHTML = `<div class="alert error small">Assistant : ${escapeHtml(e.message || e)}</div>`;
    } finally {
      btn.disabled = false; btn.classList.remove("busy"); btn.textContent = label;
    }
  }

  // ------------------------------------------------------------------ renommage en ligne
  function startRename(titleEl) {
    const wrap = titleEl.closest(".card-title");
    const form = wrap && wrap.querySelector("form.rename");
    if (!form) return;
    titleEl.hidden = true; form.hidden = false;
    const input = form.querySelector("input");
    input.focus(); input.select();
  }
  function cancelRename(btn) {
    const wrap = btn.closest(".card-title");
    const form = wrap.querySelector("form.rename"), title = wrap.querySelector("h2, h3");
    form.hidden = true; title.hidden = false;
    form.querySelector("input").value = title.textContent.replace("✎", "").trim();
  }

  // ------------------------------------------------------------------ lecteur global (page Morceaux)
  const gp = { wave: null, card: null };
  function gpEl(role) { return $(`#global-player [data-player="${role}"]`); }
  function gpCards() { return $$(".card[data-audio], .version[data-audio]"); }
  function gpFmt(s) { return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }

  function gpLoad(card, autoplay = true) {
    const player = $("#global-player");
    if (!player || !card || !window.WaveSurfer) return;
    player.hidden = false;
    document.body.classList.add("has-player");
    gpCards().forEach(c => c.classList.toggle("playing", c === card));
    gp.card = card;
    gpEl("title").textContent = card.dataset.name;
    gpEl("sub").textContent = card.dataset.sub || "";
    gpEl("open").href = `/studio#job=${card.dataset.job}`;
    gpEl("download").href = card.dataset.audio;
    gpEl("current").textContent = "0:00"; gpEl("total").textContent = "…";
    if (gp.wave) { try { gp.wave.destroy(); } catch (_) {} }
    const css = getComputedStyle(document.documentElement);
    gp.wave = WaveSurfer.create({
      container: gpEl("wave"), url: card.dataset.audio, height: 40, normalize: true, barWidth: 2, barGap: 1, barRadius: 2,
      waveColor: css.getPropertyValue("--fg-faint").trim(), progressColor: css.getPropertyValue("--accent").trim(),
      cursorColor: css.getPropertyValue("--fg").trim(), autoplay,
    });
    gp.wave.on("ready", d => { gpEl("total").textContent = gpFmt(d); });
    gp.wave.on("timeupdate", t => { gpEl("current").textContent = gpFmt(t); });
    gp.wave.on("play", () => { gpEl("toggle").textContent = "❚❚"; card.classList.add("is-playing"); });
    gp.wave.on("pause", () => { gpEl("toggle").textContent = "▶"; card.classList.remove("is-playing"); });
    gp.wave.on("finish", () => { gpStep(1); });
  }
  function gpStep(delta) {
    const cards = gpCards();
    if (!cards.length) return;
    const index = Math.max(0, cards.indexOf(gp.card));
    const next = cards[(index + delta + cards.length) % cards.length];
    gpLoad(next, true);
  }
  function gpToggle() {
    if (gp.wave) gp.wave.playPause();
    else if (gpCards().length) gpLoad(gpCards()[0], true);
  }

  // ------------------------------------------------------------------ écoute A/B (page projet)
  const ab = { waves: {}, active: "A", ready: 0 };
  function abInit(root) {
    const box = $("[data-ab]", root || document);
    if (!box || !window.WaveSurfer) return;
    Object.values(ab.waves).forEach(w => { try { w.destroy(); } catch (_) {} });
    ab.waves = {}; ab.active = "A";
    const css = getComputedStyle(document.documentElement);
    for (const side of ["A", "B"]) {
      const sel = $(`[data-ab-select="${side}"]`, box);
      const wave = WaveSurfer.create({
        container: $(`[data-ab-wave="${side}"]`, box), url: sel.value, height: 44, normalize: true, barWidth: 2, barGap: 1, barRadius: 2,
        waveColor: css.getPropertyValue("--fg-faint").trim(), progressColor: css.getPropertyValue("--accent").trim(),
        cursorColor: css.getPropertyValue("--fg").trim(),
      });
      wave.setVolume(side === "A" ? 1 : 0);
      wave.on("timeupdate", t => { if (side === ab.active) $("[data-ab-time]", box).textContent = gpFmt(t); });
      wave.on("play", () => { if (side === ab.active) $("[data-ab-play]", box).textContent = "❚❚"; });
      wave.on("pause", () => { if (side === ab.active) $("[data-ab-play]", box).textContent = "▶"; });
      wave.on("interaction", t => { const other = ab.waves[side === "A" ? "B" : "A"]; if (other) other.setTime(t); });
      sel.addEventListener("change", () => { wave.load(sel.value); });
      ab.waves[side] = wave;
    }
    $("[data-ab-play]", box).addEventListener("click", () => {
      const playing = ab.waves[ab.active].isPlaying();
      Object.values(ab.waves).forEach(w => playing ? w.pause() : w.play());
    });
    $("[data-ab-toggle]", box).addEventListener("click", () => abSwitch());
  }
  function abSwitch(side) {
    const box = $("[data-ab]");
    if (!box || !ab.waves.A) return;
    side = side || (ab.active === "A" ? "B" : "A");
    const from = ab.waves[ab.active], to = ab.waves[side];
    if (side !== ab.active) { to.setTime(from.getCurrentTime()); from.setVolume(0); to.setVolume(1); ab.active = side; }
    $("[data-ab-current]", box).textContent = side;
    $$(".ab-wave", box).forEach(w => w.classList.toggle("active", w.dataset.abWave === side));
    $("[data-ab-time]", box).textContent = gpFmt(to.getCurrentTime());
  }

  // ------------------------------------------------------------------ lecteur audio
  let wave = null;
  function destroyPlayer() { if (wave) { try { wave.destroy(); } catch (_) {} wave = null; } }

  function initPlayer(root) {
    destroyPlayer();
    const player = $(".player[data-src]", root || document);
    if (!player || !window.WaveSurfer) return;
    const css = getComputedStyle(document.documentElement);
    const playBtn = $('[data-action="play"]', player);
    const cur = $('[data-role="current"]', player);
    const tot = $('[data-role="total"]', player);
    const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
    wave = WaveSurfer.create({
      container: $('[data-role="wave"]', player), url: player.dataset.src, height: 72, normalize: true,
      waveColor: css.getPropertyValue("--fg-faint").trim() || "#5d6575",
      progressColor: css.getPropertyValue("--accent").trim() || "#f0b429",
      cursorColor: css.getPropertyValue("--fg").trim(), barWidth: 2, barGap: 1, barRadius: 2,
    });
    wave.on("ready", d => { tot.textContent = fmt(d); });
    wave.on("timeupdate", t => { cur.textContent = fmt(t); });
    wave.on("play", () => { playBtn.textContent = "❚❚"; });
    wave.on("pause", () => { playBtn.textContent = "▶"; });
    wave.on("finish", () => { playBtn.textContent = "▶"; });
    playBtn.addEventListener("click", () => wave.playPause());
  }

  // ------------------------------------------------------------------ délégation d'événements
  document.addEventListener("click", ev => {
    const p = ev.target.closest("[data-player]");
    if (p) {
      const role = p.dataset.player;
      if (role === "play-card") { const card = p.closest(".card, .version"); if (gp.card === card && gp.wave) gp.wave.playPause(); else gpLoad(card, true); return; }
      if (role === "toggle") { gpToggle(); return; }
      if (role === "prev") { gpStep(-1); return; }
      if (role === "next") { gpStep(1); return; }
    }
    const t = ev.target.closest("[data-action], .tab, .help-toggle, .track");
    if (!t) return;
    if (t.classList.contains("tab")) { activateTab(t); return; }
    if (t.classList.contains("help-toggle")) { t.closest(".field").classList.toggle("show-help"); return; }
    if (t.classList.contains("track")) { $$(".track.selected").forEach(x => x.classList.remove("selected")); t.classList.add("selected"); return; }
    const form = t.closest("form");
    switch (t.dataset.action) {
      case "random-seed": {
        const inp = $('input[name="seed"]', form);
        inp.value = Math.floor(Math.random() * 2 ** 31);
        inp.dispatchEvent(new Event("input", { bubbles: true }));
        break;
      }
      case "preset": {
        const ta = $('textarea[name="style"]', form);
        ta.value = t.dataset.prompt; ta.focus();
        ta.dispatchEvent(new Event("input", { bubbles: true }));
        break;
      }
      case "tag": insertAtCursor($('textarea[name="lyrics"]', form), t.dataset.tag); break;
      case "clear-abc": {
        const ta = $('textarea[name="abc"]', form);
        ta.value = ""; ta.dispatchEvent(new Event("input", { bubbles: true }));
        break;
      }
      case "reset-advanced": resetAdvanced(form); break;
      case "abc-zoom": setAbcZoomMode(t.dataset.mode); rerenderAllAbc(); break;
      case "rename": startRename(t); break;
      case "rename-cancel": cancelRename(t); break;
      case "toggle-assistant": setAssistantOpen($("#assistant-drawer").hidden); break;
      case "mode": setMode(t.dataset.mode); break;
      case "detach-project": {
        const f = $("#job-form");
        $('[name="project_id"]', f).value = "";
        t.closest(".project-badge").remove();
        const kind = $('[name="origin_kind"]', f)?.value, name = $('[name="origin_name"]', f)?.value;
        const labels = { reuse: "Réglages repris de", variation: "Variation de", "edit-score": "Partition retouchée de" };
        const title = $('[data-role="form-title"]', f);
        if (title) title.textContent = kind === "assistant" ? "Proposition de l'assistant" : (labels[kind] && name ? `${labels[kind]} « ${name} »` : "Nouveau morceau");
        break;
      }
      case "close-detail": {
        const d = $("#detail");
        if (d) { d.innerHTML = "<p>Sélectionnez un morceau dans la bibliothèque pour l'écouter, lire sa partition et ses réglages.</p>"; d.classList.add("detail-empty"); }
        $$(".track.selected").forEach(x => x.classList.remove("selected"));
        break;
      }
      case "close-assistant": setAssistantOpen(false); break;
      case "starter": {
        const ta = $("#assistant-body textarea");
        if (ta && !ta.disabled) { ta.value = t.textContent.trim(); ta.closest("form").requestSubmit(); }
        break;
      }
      case "quick": quickAction(t); break;
      case "quick-undo": {
        const why = t.closest("#quick-why"); const field = $(`textarea[name="${why.dataset.field}"]`, form);
        field.value = why.dataset.previous; field.dispatchEvent(new Event("input", { bubbles: true })); why.hidden = true; why.innerHTML = "";
        break;
      }
      case "llm-preset": {
        const f = t.closest("form");
        const url = $('[name="base_url"]', f), model = $('[name="model"]', f);
        if (url && !url.disabled) url.value = t.dataset.url;
        if (model && !model.disabled && t.dataset.model !== undefined) model.value = t.dataset.model;
        break;
      }
      case "strip-chords": stripChords(form); break;
      case "batch-submit": submitBatch(form); break;
      case "open-settings": {
        const dlg = $("#settings-dialog");
        document.body.dispatchEvent(new Event("settings-open"));
        if (!dlg.open) dlg.showModal();
        break;
      }
      case "close-settings": $("#settings-dialog").close(); break;
      case "play": break; // géré par initPlayer
    }
  });

  document.addEventListener("keydown", ev => {
    const t = ev.target;
    if (ev.key === "Enter" && !ev.shiftKey && t.matches && t.matches(".composer textarea")) { ev.preventDefault(); t.closest("form").requestSubmit(); return; }
    if (ev.key === "Escape" && t.closest && t.closest("form.rename")) { cancelRename(t); return; }
    if (["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(t.tagName) === false && $("[data-ab]") && ab.waves.A) {
      if (ev.key === "a" || ev.key === "A") { abSwitch("A"); return; }
      if (ev.key === "b" || ev.key === "B") { abSwitch("B"); return; }
      if (ev.key === " ") { ev.preventDefault(); $("[data-ab-play]").click(); return; }
    }
    if (ev.key === " " && $("#global-player") && !["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(t.tagName)) { ev.preventDefault(); gpToggle(); return; }
    if (t.dataset && t.dataset.player === "play-card" && (ev.key === "Enter" || ev.key === " ")) { ev.preventDefault(); t.click(); return; }
    if (t.classList && t.classList.contains("track") && (ev.key === "Enter" || ev.key === " ")) { ev.preventDefault(); t.click(); }
  });

  document.addEventListener("input", ev => {
    const form = ev.target.closest("#job-form");
    if (!form) return;
    const name = ev.target.name;
    if (name === "lyrics" || name === "style") updateLyricsStats(form);
    if (name === "abc") { syncModeRules(form); renderAbcPreview(form); }
    if (name === "semantic.max_tokens" || name === "semantic.min_tokens") updateDurationEstimate(form);
    if (name === "max_duration" || name === "plan_overflow" || name === "instrumental") updateLyricsStats(form);
    if (name === "cot" && $('input[name="instrumental"]', form)?.checked && ev.target.value !== "full") {
      ev.target.value = "full"; updateInstrumentalNote(form);
    }
    if (ev.target.type === "checkbox" && ev.target.closest(".switch")) {
      const lbl = ev.target.closest(".switch").querySelector(".switch-label");
      if (lbl) lbl.textContent = ev.target.checked ? "activé" : "désactivé";
    }
  });
  document.addEventListener("change", ev => {
    const form = ev.target.closest("#job-form");
    if (form && ev.target.name === "cot") syncModeRules(form);
    if (ev.target.id === "toggle-help") applyGlobalHelp();
    if (ev.target.id === "batch-file") {
      const f = ev.target.files[0];
      const form = ev.target.closest("form");
      $("#batch-filename", form).textContent = f ? `${f.name} · ${(f.size / 1024).toFixed(1)} Ko` : "";
      $("#batch-submit", form).disabled = !f;
    }
    if (ev.target.dataset && ev.target.dataset.action === "import-abc") {
      const file = ev.target.files[0];
      if (!file) return;
      file.text().then(text => {
        const ta = $('textarea[name="abc"]', form);
        ta.value = text; ta.dispatchEvent(new Event("input", { bubbles: true }));
        ev.target.value = "";
      });
    }
  });

  // ------------------------------------------------------------------ hooks HTMX
  document.addEventListener("htmx:afterSwap", ev => {
    const target = (ev.detail && ev.detail.target) || ev.target;
    if (!target || !target.id) return;
    if (target.id === "form-panel") {
      setMode("compose"); initForm(target); target.scrollIntoView({ behavior: "smooth", block: "start" });
      const pid = $('[name="project_id"]', target)?.value;
      if (pid) syncAssistant(`project_id=${encodeURIComponent(pid)}`, { project: pid });
    }
    if (target.id === "detail") {
      initPlayer(target); renderAbcBlocks(target);
      const d = $(".detail", target), id = d?.dataset.job;
      if (id) {
        setMode("browse"); $$(".track").forEach(x => x.classList.toggle("selected", x.dataset.job === id));
        syncAssistant(`job_id=${encodeURIComponent(id)}`, { project: d.dataset.project || "", job: id });
      }
      if (!$(".detail", target)) target.innerHTML = '<p>Sélectionnez un morceau dans la bibliothèque.</p>';
      target.classList.toggle("detail-empty", !$(".detail", target));
    }
    if (target.id === "queue") {
      const live = $('[data-role="live-abc"]', target);
      if (live) live.scrollTop = live.scrollHeight;
    }
    if (target.id === "library") {
      const count = $('[data-role="library-count"]'); if (count) count.textContent = $$(".track", target).length;
      const id = $(".detail")?.dataset.job;
      if (id) $$(".track", target).forEach(x => x.classList.toggle("selected", x.dataset.job === id));
    }
    if (target.id === "project") { abInit(target); renderAbcBlocks(target); }
    if (target.id === "assistant-body") { scrollChat(); const ta = $("textarea", target); if (ta && !ta.disabled && !$("#assistant-drawer").hidden) ta.focus(); }
    if (target.id === "gallery" && gp.card) {
      const again = $(`.card[data-job="${gp.card.dataset.job}"]`, target);
      if (again) { gp.card = again; again.classList.add("playing"); if (gp.wave && gp.wave.isPlaying()) again.classList.add("is-playing"); }
    }
  });
  document.addEventListener("htmx:beforeRequest", ev => {
    if (ev.detail.elt && ev.detail.elt.matches && ev.detail.elt.matches(".composer")) {
      const pending = $('#assistant-body [data-role="pending"]'); if (pending) { pending.hidden = false; scrollChat(); }
      const ta = $("textarea", ev.detail.elt); if (ta) ta.readOnly = true;
    }
  });
  document.addEventListener("htmx:responseError", ev => {
    if (ev.detail.xhr.status === 422) return; // le serveur renvoie le formulaire avec l'erreur
    ui.alert(`${ev.detail.xhr.responseText.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim().slice(0, 300)}`, { title: `Erreur serveur (${ev.detail.xhr.status})` });
  });
  document.addEventListener("htmx:beforeSwap", ev => {
    if (ev.detail.xhr.status === 422) ev.detail.shouldSwap = true;
  });
  document.addEventListener("assistant-open", () => setAssistantOpen(true));
  document.addEventListener("htmx:sseMessage", ev => {
    // Quand un job se termine et qu'il est affiché dans le détail, rafraîchir le détail.
    if (!ev.detail || ev.detail.type !== "library") return;
    const id = $(".detail")?.dataset.job;
    if (id && window.htmx) htmx.ajax("GET", `/jobs/${id}`, { target: "#detail", swap: "innerHTML" });
  });

  // ------------------------------------------------------------------ démarrage
  document.addEventListener("DOMContentLoaded", () => {
    try { if (localStorage.getItem("yue2.help") === "1") { $("#toggle-help").checked = true; } } catch (_) {}
    applyGlobalHelp();
    initForm(document);
    try { if (localStorage.getItem("yue2.assistant") === "1") setAssistantOpen(true); } catch (_) {}
    try { const saved = localStorage.getItem("yue2.mode"); if (saved && !location.hash) setMode(saved, false); } catch (_) {}
    abInit(document);
    const pm = /#project=([A-Za-z0-9_.-]+)/.exec(location.hash);
    if (pm && $("#assistant-body") && window.htmx) {
      htmx.ajax("GET", `/assistant/panel?project_id=${pm[1]}`, { target: "#assistant-body", swap: "innerHTML" }).then(() => setAssistantOpen(true));
      history.replaceState(null, "", location.pathname);
    }
    const m = /#job=([A-Za-z0-9_.-]+)/.exec(location.hash);
    if (m && $("#detail") && window.htmx) {
      htmx.ajax("GET", `/jobs/${m[1]}`, { target: "#detail", swap: "innerHTML" }).then(() => $("#detail")?.scrollIntoView({ behavior: "smooth", block: "start" }));
      history.replaceState(null, "", location.pathname);
    }
    $("#settings-dialog")?.addEventListener("click", ev => { if (ev.target === ev.currentTarget) ev.currentTarget.close(); });
  });
})();
