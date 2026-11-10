(() => {
  const $ = (id) => document.getElementById(id);
  const SERVER = !!(window.APP && window.APP.server);
  const el = {
    form: $("form"), path: $("path"), file: $("file"), language: $("language"), model: $("model"), modelHint: $("modelHint"),
    start: $("btnStart"), cancel: $("btnCancel"), clear: $("btnClear"), refresh: $("btnRefresh"), browse: $("btnBrowse"),
    box: $("progressBox"), bar: $("bar"), stage: $("stage"), percent: $("percent"), alert: $("alert"),
  };
  // estados da tela: idle (formulário livre) | busy (enviando/na fila/processando) | final (concluído/erro/cancelado)
  let state = "idle", jobId = null, timer = null, setupReady = false, lastDownloadName = "legenda.srt";

  const HEADERS = { "X-Requested-With": "fetch" };
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  async function api(url, opts = {}) {
    const r = await fetch(url, { ...opts, headers: { ...HEADERS, ...(opts.headers || {}) } });
    if (r.status === 401 && SERVER) { location.href = "/login"; throw new Error("sessão expirada"); }
    let body = {};
    try { body = await r.json(); } catch { /* corpo vazio */ }
    return { status: r.status, ok: r.ok, body };
  }
  const getJSON = async (url) => (await api(url)).body;
  const post = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

  function showAlert(kind, html) {
    el.alert.className = `alert alert-${kind} mt-4 mb-0`;
    el.alert.innerHTML = html;
  }
  const hideAlert = () => el.alert.classList.add("d-none");

  function setProgress(p, stage, active = true) {
    el.box.classList.remove("d-none");
    el.bar.style.width = `${p}%`;
    el.bar.parentElement.setAttribute("aria-valuenow", Math.round(p));
    el.bar.classList.toggle("progress-bar-animated", active);
    el.bar.classList.toggle("progress-bar-striped", active);
    el.percent.textContent = `${Math.round(p)}%`;
    el.stage.textContent = stage;
  }

  // ---------- estado dos controles ----------
  function applyState() {
    const idle = state === "idle", busy = state === "busy", ready = setupReady;
    [el.path, el.language, el.model, el.refresh, el.browse].forEach((e) => e && (e.disabled = !(idle && ready)));
    // no estado final o arquivo pode ser trocado (isso zera a tela); o botão Gerar só volta ao zerar
    if (el.file) el.file.disabled = busy || !ready;
    el.start.disabled = !(idle && ready);
    el.clear.disabled = busy;
    el.cancel.classList.toggle("d-none", !busy);
    el.start.textContent = busy ? "Processando…" : "Gerar legenda";
  }
  function setState(s) { state = s; applyState(); }

  // volta a tela ao estado inicial (sem mexer no que já estiver escolhido, se keepInputs)
  function resetUI(keepInputs = false) {
    clearInterval(timer);
    jobId = null;
    el.box.classList.add("d-none");
    el.bar.classList.remove("bg-success", "bg-danger");
    hideAlert();
    if (!keepInputs) {
      if (el.path) el.path.value = "";
      if (el.file) el.file.value = "";
    }
    setState("idle");
  }

  // "Limpar" / escolher outro arquivo: no servidor, apaga legenda e job guardados
  async function discardJob() {
    if (SERVER && jobId) { try { await api(`/api/jobs/${jobId}`, { method: "DELETE" }); } catch { /* já não existe */ } }
  }

  // ---------- idiomas e modelos ----------
  async function loadLanguages() {
    const langs = await getJSON("/api/languages");
    el.language.innerHTML = langs.map((l) => `<option value="${l.code}">${esc(l.name)}</option>`).join("");
    const saved = localStorage.getItem("language");
    if (saved) el.language.value = saved;
  }

  async function loadModels() {
    el.model.innerHTML = "<option>Carregando…</option>";
    el.modelHint.textContent = "";
    el.modelHint.className = "form-text";
    let data;
    try { data = await getJSON("/api/models"); } catch { data = { ok: false, error: "Falha ao consultar o servidor." }; }
    if (!data.ok) {
      el.model.innerHTML = "";
      el.modelHint.className = "form-text text-danger";
      el.modelHint.textContent = data.error;
      return;
    }
    if (!data.models.length) {
      el.model.innerHTML = "";
      el.modelHint.className = "form-text text-danger";
      el.modelHint.textContent = "Nenhum modelo com suporte a áudio encontrado no Ollama.";
      return;
    }
    const opt = (m) => {
      const sz = m.installed ? `${(m.size / 1e9).toFixed(1)} GB` : (m.approx_size ? `~${m.approx_size}` : "tamanho n/d");
      const star = m.recommended ? "" : " (fora do catálogo)";
      return `<option value="${esc(m.name)}" data-installed="${m.installed}" data-size="${esc(m.approx_size)}" data-note="${esc(m.note)}">${esc(m.name)} — ${esc(sz)}${star}</option>`;
    };
    const inst = data.models.filter((m) => m.installed), other = data.models.filter((m) => !m.installed);
    el.model.innerHTML =
      (inst.length ? `<optgroup label="✔ Instalados nesta máquina">${inst.map(opt).join("")}</optgroup>` : "") +
      (other.length ? `<optgroup label="Recomendados — serão baixados ao gerar">${other.map(opt).join("")}</optgroup>` : "");
    const saved = localStorage.getItem("model");
    if (saved && data.models.some((m) => m.name === saved)) el.model.value = saved;
    else el.model.selectedIndex = 0; // primeiro instalado do catálogo (família Gemma tem prioridade)
    showModelHint(inst.length);
  }

  function showModelHint(installedCount) {
    const o = el.model.selectedOptions[0];
    if (!o) return;
    el.modelHint.className = "form-text";
    if (o.dataset.installed === "false") {
      el.modelHint.className = "form-text text-warning-emphasis";
      el.modelHint.textContent = `Não instalado: será baixado antes de gerar${o.dataset.size ? ` (~${o.dataset.size})` : ""}. ${o.dataset.note}`;
    } else {
      el.modelHint.textContent = `${o.dataset.note || "Instalado."}` + (installedCount != null ? ` · ${installedCount} instalado(s).` : "");
    }
  }
  el.model.addEventListener("change", () => showModelHint());

  // ---------- acompanhamento do job ----------
  function stageText(s) {
    if (s.status === "queued") {
      return s.queue_position > 1 ? `Na fila — posição ${s.queue_position}` : "Na fila — você é o próximo";
    }
    return s.message ? `${s.stage} — ${s.message}` : s.stage;
  }

  function renderJob(s) {
    const warn = s.warnings && s.warnings.length ? `<hr><small>${s.warnings.map(esc).join("<br>")}</small>` : "";
    if (s.status === "queued" || s.status === "running") {
      if (state !== "busy") setState("busy");
      setProgress(s.progress, stageText(s));
      return;
    }
    clearInterval(timer);
    setState("final");
    if (s.status === "done") {
      setProgress(100, "Concluído", false);
      el.bar.classList.add("bg-success");
      lastDownloadName = s.download_name || "legenda.srt";
      if (SERVER) {
        showAlert("success",
          `<strong>Legenda pronta!</strong> ${esc(s.message)}<br>
           <small class="text-break">${esc(lastDownloadName)} — será apagada do servidor após o download.</small><br>
           <button type="button" id="btnDownload" class="btn btn-success btn-sm mt-2">Baixar .srt</button>${warn}`);
        $("btnDownload").addEventListener("click", downloadServer);
      } else {
        showAlert("success",
          `<strong>Legenda gerada!</strong> ${esc(s.message)}<br>
           <small class="text-break">Salva em: ${esc(s.output_path)}</small><br>
           <a class="btn btn-success btn-sm mt-2" href="/api/jobs/${jobId}/download">Baixar .srt</a>${warn}`);
      }
    } else if (s.status === "cancelled") {
      setProgress(s.progress, "Cancelado", false);
      showAlert("warning", "Processamento cancelado.");
    } else {
      setProgress(s.progress, "Erro", false);
      el.bar.classList.add("bg-danger");
      showAlert("danger", `<strong>Erro:</strong> ${esc(s.error)}${warn}`);
    }
  }

  async function poll() {
    let r;
    try { r = await api(`/api/jobs/${jobId}`); } catch { return; }
    if (r.status === 404) {  // expirou (TTL) ou foi removido
      resetUI(true);
      showAlert("warning", "Este processamento expirou e foi apagado do servidor.");
      return;
    }
    if (r.ok) renderJob(r.body);
  }

  function startPolling() {
    clearInterval(timer);
    timer = setInterval(poll, 800);
    poll();
  }

  // ---------- baixar (servidor): baixa, a tela volta ao início ----------
  async function downloadServer() {
    const btn = $("btnDownload");
    btn.disabled = true;
    const r = await fetch(`/api/jobs/${jobId}/download`, { headers: HEADERS });
    if (!r.ok) {
      btn.disabled = false;
      let msg = "Não foi possível baixar a legenda.";
      try { msg = (await r.json()).error || msg; } catch { /* sem corpo */ }
      showAlert("danger", esc(msg));
      return;
    }
    const url = URL.createObjectURL(await r.blob());
    const a = document.createElement("a");
    a.href = url; a.download = lastDownloadName;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    resetUI();  // o servidor já apagou a legenda: tela zerada
  }

  // ---------- enviar / iniciar ----------
  function uploadFile(file, language, model) {
    return new Promise((resolve) => {
      const xhr = new XMLHttpRequest();
      const fd = new FormData();
      fd.append("file", file); fd.append("language", language); fd.append("model", model);
      xhr.open("POST", "/api/jobs");
      xhr.setRequestHeader("X-Requested-With", "fetch");
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) setProgress(100 * e.loaded / e.total, "Enviando arquivo…"); };
      xhr.onload = () => {
        if (xhr.status === 401) { location.href = "/login"; return; }
        let body = {};
        try { body = JSON.parse(xhr.responseText); } catch { /* resposta inesperada */ }
        resolve({ ok: xhr.status >= 200 && xhr.status < 300, body });
      };
      xhr.onerror = () => resolve({ ok: false, body: { error: "Falha de rede ao enviar o arquivo." } });
      xhr.send(fd);
    });
  }

  el.form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (state !== "idle") return;
    hideAlert();
    el.bar.classList.remove("bg-success", "bg-danger");
    localStorage.setItem("language", el.language.value);
    localStorage.setItem("model", el.model.value);
    let r;
    if (SERVER) {
      const f = el.file.files[0];
      if (!f) { showAlert("danger", "Selecione um arquivo de áudio ou vídeo."); return; }
      setState("busy");
      setProgress(0, "Enviando arquivo…");
      r = await uploadFile(f, el.language.value, el.model.value);
    } else {
      const res = await post("/api/jobs", { path: el.path.value, language: el.language.value, model: el.model.value })
        .catch(() => ({ body: { ok: false, error: "Falha ao contatar o servidor." } }));
      r = { ok: res.body.ok, body: res.body };
      if (r.ok) { setState("busy"); setProgress(0, "Iniciando…"); }
    }
    if (!r.ok || !r.body.ok) {
      if (SERVER) { el.box.classList.add("d-none"); setState("idle"); }
      showAlert("danger", esc(r.body.error || "Não foi possível iniciar o processamento."));
      return;
    }
    jobId = r.body.id;
    startPolling();
  });

  el.cancel.addEventListener("click", async () => {
    el.cancel.disabled = true;
    await post(`/api/jobs/${jobId}/cancel`);
    el.cancel.disabled = false;
  });

  el.clear.addEventListener("click", async () => {
    await discardJob();
    resetUI();
  });

  // escolher outro arquivo depois do estado final zera a tela (e apaga o que estava guardado no servidor)
  if (el.file) {
    el.file.addEventListener("change", async () => {
      if (state === "final") { await discardJob(); resetUI(true); }
    });
  }

  el.refresh.addEventListener("click", loadModels);

  // ---------- preparo do ambiente (Ollama / FFmpeg / modelo base) ----------
  const sc = {
    card: $("setupCard"), title: $("setupTitle"), msg: $("setupMsg"), prog: $("setupProgress"), bar: $("setupBar"),
    detail: $("setupDetail"), actions: $("setupActions"),
  };
  let modelsLoaded = false, setupBusy = false;

  function btn(label, cls, fn) {
    const b = document.createElement("button");
    b.type = "button"; b.className = `btn btn-sm ${cls}`; b.textContent = label; b.onclick = fn;
    sc.actions.appendChild(b);
  }

  function renderSetup(s) {
    setupReady = s.status === "ready";
    sc.card.classList.toggle("d-none", setupReady);
    applyState();
    if (setupReady) {
      if (!modelsLoaded) { modelsLoaded = true; loadModels(); }
      return;
    }
    const titles = {
      checking: "Preparando o ambiente", starting: "Iniciando o Ollama", pulling: "Baixando o modelo base",
      installing: "Instalando o Ollama", installing_ffmpeg: "Baixando o FFmpeg", ollama_missing: "Ollama não encontrado",
      ffmpeg_missing: "FFmpeg não encontrado", error: "Não foi possível preparar o ambiente",
    };
    sc.title.textContent = titles[s.status] || "Preparando";
    sc.msg.textContent = s.message;
    sc.detail.textContent = s.detail || "";
    const showBar = ["pulling", "installing", "installing_ffmpeg"].includes(s.status);
    sc.prog.classList.toggle("d-none", !showBar);
    sc.bar.style.width = `${s.progress || 0}%`;
    sc.card.classList.toggle("border-danger", ["error", "ffmpeg_missing"].includes(s.status));
    sc.actions.innerHTML = "";
    if (s.status === "ollama_missing") {
      sc.msg.innerHTML = esc(s.message) + "<br>O <strong>Ollama é um requisito</strong> do Mini Transcritor: ele precisa estar instalado para executar os modelos.";
      if (s.can_auto_install) btn("Baixar e instalar o Ollama", "btn-primary", async () => renderSetup((await post("/api/setup/install-ollama")).body));
      btn("Abrir página de download", "btn-outline-primary", () => window.open(s.download_page, "_blank"));
    }
    if (["ollama_missing", "ffmpeg_missing", "error"].includes(s.status)) {
      btn("Verificar novamente", "btn-outline-secondary", async () => renderSetup((await post("/api/setup/run")).body));
      if (!SERVER) btn("Configurações", "btn-outline-secondary", openSettings);
    }
  }

  async function pollSetup() {
    if (setupBusy) return;
    setupBusy = true;
    try { renderSetup(await getJSON("/api/setup")); } catch { /* servidor reiniciando */ }
    setupBusy = false;
  }

  // ---------- configurações (somente modo local) ----------
  let openSettings = () => {};
  if (!SERVER) {
    const settingsModal = new bootstrap.Modal($("settingsModal"));
    const modal = new bootstrap.Modal($("browseModal"));
    const bPath = $("browsePath"), bList = $("browseList");

    openSettings = async () => {
      $("settingsAlert").classList.add("d-none");
      const s = await getJSON("/api/settings");
      $("setHost").value = s.ollama_host; $("setBase").value = s.base_model;
      $("setFfmpeg").value = s.ffmpeg_path || "";
      $("ffmpegInfo").textContent = (s.ffmpeg_active ? `Em uso: ${s.ffmpeg_active}. ` : "") + `Pasta local: ${s.ffmpeg_local_dir}`;
      $("setAuto").checked = s.auto_pull; $("setChunk").value = s.chunk_seconds;
      settingsModal.show();
    };
    $("btnSettings").addEventListener("click", openSettings);

    $("settingsForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const r = (await post("/api/settings", {
        ollama_host: $("setHost").value, ffmpeg_path: $("setFfmpeg").value, base_model: $("setBase").value,
        auto_pull: $("setAuto").checked, chunk_seconds: $("setChunk").value,
      })).body;
      if (!r.ok) { const a = $("settingsAlert"); a.textContent = r.error; a.classList.remove("d-none"); return; }
      settingsModal.hide();
      modelsLoaded = false;
      pollSetup();
    });

    async function openDir(path) {
      const d = await getJSON(`/api/browse?path=${encodeURIComponent(path || "")}`);
      if (!d.ok) { bList.innerHTML = `<div class="p-3 text-danger">${esc(d.error)}</div>`; return; }
      bPath.textContent = d.path || "Este computador";
      const rows = [];
      if (d.parent !== null) rows.push(`<a class="list-group-item" data-dir="${esc(d.parent)}">⬆ ..</a>`);
      d.dirs.forEach((x) => rows.push(`<a class="list-group-item" data-dir="${esc(x.path)}">📁 ${esc(x.name)}</a>`));
      d.files.forEach((x) => rows.push(`<a class="list-group-item" data-file="${esc(x.path)}">${x.kind === "video" ? "🎬" : "🎵"} ${esc(x.name)}</a>`));
      if (!rows.length) rows.push('<div class="p-3 text-body-secondary">Pasta vazia.</div>');
      bList.innerHTML = rows.join("");
    }
    bList.addEventListener("click", (e) => {
      const a = e.target.closest("a");
      if (!a) return;
      if (a.dataset.dir !== undefined) openDir(a.dataset.dir);
      else if (a.dataset.file) { el.path.value = a.dataset.file; modal.hide(); }
    });
    el.browse.addEventListener("click", () => {
      const cur = el.path.value.trim();
      modal.show();
      openDir(cur ? cur.replace(/[\\/][^\\/]*$/, "") : "");
    });
  } else {
    $("btnLogout").addEventListener("click", async () => {
      await post("/logout").catch(() => {});
      location.href = "/login";
    });
  }

  // ---------- inicialização ----------
  async function restoreServerJob() {  // recarregou a página no meio do processo: retoma de onde estava
    const cur = (await getJSON("/api/jobs/current")).job;
    if (!cur) return;
    jobId = cur.id;
    renderJob(cur);
    if (cur.status === "queued" || cur.status === "running") startPolling();
  }

  loadLanguages();
  applyState();
  pollSetup().then(() => { if (SERVER) restoreServerJob(); });
  setInterval(pollSetup, 1000);
})();
