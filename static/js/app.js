(() => {
  const $ = (id) => document.getElementById(id);
  const SERVER = !!(window.APP && window.APP.server);
  const el = {
    form: $("form"), path: $("path"), file: $("file"), language: $("language"), model: $("model"), modelHint: $("modelHint"),
    engine: $("engine"), engineHint: $("engineHint"), enginePlan: $("enginePlan"), modelLabel: $("modelLabel"), diarizeBox: $("diarizeBox"),
    diarize: $("diarize"), diarizeHint: $("diarizeHint"), context: $("context"),
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
  var fixedModels = false;  // modos de qualidade têm os modelos fixos (gerador e revisor): a lista de modelos fica travada
  function applyState() {
    const idle = state === "idle", busy = state === "busy", ready = setupReady;
    [el.path, el.language, el.engine, el.model, el.diarize, el.context, el.refresh, el.browse].forEach((e) => e && (e.disabled = !(idle && ready)));
    if (fixedModels) el.model.disabled = true;
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
      if (el.context) el.context.value = "";
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

  let engines = [], diarizationReady = false;
  const OLLAMA = "ollama";
  // nome de exibição dos modelos dos motores alternativos (o valor enviado continua sendo o identificador): palavras separadas por espaço,
  // só a 1ª letra maiúscula; siglas e tokens com número (ASR, NVIDIA, V3, Q5_0, 30B, A4B) em caixa alta
  const ACRONYMS = /^(asr|nvidia|nvfp4|gguf|vllm)$/i;
  const modelTitle = (name, eid) => (eid === OLLAMA || !name ? name : String(name).split("-").map((w) => (ACRONYMS.test(w) || /\d/.test(w) ? w.toUpperCase() : w.charAt(0).toUpperCase() + w.slice(1))).join(" "));
  // ordem oficial dos modelos: do menor ao maior (tiny < base < small < medium < turbo < large; e2b < e4b < 12b < 30b);
  // a versão quantizada vem antes da completa. Estável: o que não for reconhecido mantém a ordem do catálogo.
  const WHISPER_RANK = { tiny: 1, base: 2, small: 3, medium: 4, turbo: 5, large: 6 };
  const sizeRank = (name) => {
    const n = String(name || "").toLowerCase();
    const w = n.match(/(tiny|base|small|medium|turbo|large)/);
    if (w) return n.includes("turbo") ? WHISPER_RANK.turbo : WHISPER_RANK[w[1]];
    const p = n.match(/(\d+(?:\.\d+)?)b(?![a-z])/);
    return p ? parseFloat(p[1]) : Infinity;
  };
  const gb = (m) => parseFloat(String(m.approx_size || "").replace(/[^0-9.]/g, "")) || 0;
  const bySize = (a, b) => (sizeRank(a.name) - sizeRank(b.name) || 0) || (/q\d/i.test(b.name) - /q\d/i.test(a.name)) || (gb(a) - gb(b)) || String(a.name).localeCompare(String(b.name), "en", { numeric: true });
  const engineId = () => el.engine.value || OLLAMA;
  const engineInfo = () => engines.find((e) => e.id === engineId()) || { id: OLLAMA, supports_diarization: false };

  async function loadEngines() {
    let data;
    try { data = await getJSON("/api/engines"); } catch { return; }
    if (!data.ok) return;
    engines = data.engines; diarizationReady = data.diarization_ready;
    const saved = localStorage.getItem("engine");
    el.engine.innerHTML = engines.map((e) => `<option value="${esc(e.id)}">${esc(e.label)}${e.available ? "" : " — não instalado"}</option>`).join("");
    if (saved && engines.some((e) => e.id === saved)) el.engine.value = saved;
    onEngineChange(false);
  }

  function onEngineChange(reload = true) {
    const e = engineInfo();
    el.engineHint.className = e.available ? "form-text" : "form-text text-danger";
    el.engineHint.textContent = e.available ? e.description : `${e.description} ${e.hint}`;
    el.modelLabel.textContent = e.id === OLLAMA ? "Modelo Ollama (com áudio)" : `Modelo ${e.label}`;
    el.diarizeBox.classList.toggle("d-none", !e.supports_diarization);
    if (!e.supports_diarization) el.diarize.checked = false;
    el.diarizeHint.className = diarizationReady ? "form-text" : "form-text text-warning-emphasis";
    el.diarizeHint.textContent = diarizationReady
      ? "Marca na legenda quando muda o falante ([Falante 1], [Falante 2]…)."
      : "Exige um token do Hugging Face: informe em Configurações.";
    showDevices(e);
    if (reload) loadModels();
  }

  // Hardware separado da descrição do motor: onde cada ETAPA roda nesta máquina + o que foi detectado (só motores Whisper)
  async function showDevices(e) {
    el.enginePlan.classList.add("d-none");
    if (e.id === OLLAMA || e.fixed_models || !e.available) return;
    let info;
    try { info = await getJSON(`/api/devices?engine=${encodeURIComponent(e.id)}`); } catch { return; }
    if (engineId() !== e.id || !info.ok || !(info.plan || []).length) return;
    const found = info.devices.filter((d) => d.available).map((d) => d.label).join(" · ");
    const chosen = { auto: "Automático" }[info.selected] || (info.devices.find((d) => d.id === info.selected) || {}).label || info.selected;
    const rows = info.plan.map((p) => {
      const gpu = p.device && p.device !== "cpu";
      return `<div class="mb-1"><div class="d-flex flex-wrap align-items-baseline gap-2">
        <span class="text-body-secondary">${esc(p.stage)}:</span>
        <span class="badge ${p.device ? (gpu ? "text-bg-success" : "text-bg-secondary") : "text-bg-danger"}">${esc(p.label)}</span></div>
        ${p.why ? `<div class="text-body-secondary ps-3">↳ ${esc(p.why)}</div>` : ""}</div>`;
    }).join("");
    el.enginePlan.innerHTML = `<div class="fw-semibold">Onde vai rodar nesta máquina</div>${rows}
      <div class="text-body-secondary mt-1">Hardware detectado: ${esc(found)}. Dispositivo escolhido: ${esc(chosen)} (mude em Configurações → Dispositivo dos motores Whisper).</div>`;
    el.enginePlan.classList.remove("d-none");
  }
  el.engine.addEventListener("change", () => { localStorage.setItem("engine", engineId()); onEngineChange(); });

  async function loadModels() {
    const fx = engineInfo().fixed_models;
    fixedModels = !!fx;
    if (fx) {  // modelos fixos do modo: nada a escolher (o servidor baixa o que faltar)
      el.model.innerHTML = `<option value="${esc(fx.generator)}">${esc(fx.generator)} (gerador) + ${esc(fx.reviewer)} (revisor) — fixos</option>`;
      el.model.disabled = true;
      el.modelHint.className = "form-text";
      el.modelHint.textContent = "Modelos fixos deste modo; os que faltarem serão baixados ao gerar.";
      return;
    }
    el.model.disabled = false;
    el.model.innerHTML = "<option>Carregando…</option>";
    el.modelHint.textContent = "";
    el.modelHint.className = "form-text";
    const eid = engineId();
    let data;
    try { data = await getJSON(`/api/models?engine=${encodeURIComponent(eid)}`); } catch { data = { ok: false, error: "Falha ao consultar o servidor." }; }
    if (eid !== engineId()) return;  // o usuário trocou de motor durante a consulta
    if (!data.ok) {
      el.model.innerHTML = "";
      el.modelHint.className = "form-text text-danger";
      el.modelHint.textContent = data.error;
      return;
    }
    if (!data.models.length) {
      el.model.innerHTML = "";
      el.modelHint.className = "form-text text-danger";
      el.modelHint.textContent = eid === OLLAMA ? "Nenhum modelo com suporte a áudio encontrado no Ollama."
        : "Nenhum modelo deste motor está disponível" + (SERVER ? " neste servidor." : ".");
      return;
    }
    const opt = (m) => {
      const sz = m.installed && m.size ? `${(m.size / 1e9).toFixed(1)} GB` : (m.approx_size ? m.approx_size.replace(/^~?/, "~") : "tamanho n/d");
      const star = m.recommended ? "" : " (fora do catálogo)";
      return `<option value="${esc(m.name)}" data-installed="${m.installed}" data-size="${esc(m.approx_size)}" data-note="${esc(m.note)}">${esc(modelTitle(m.name, eid))} — ${esc(sz)}${star}</option>`;
    };
    const inst = data.models.filter((m) => m.installed), other = data.models.filter((m) => !m.installed);
    const first = inst[0] || other[0]; // o padrão segue a prioridade do catálogo; só a EXIBIÇÃO usa a ordem oficial (menor → maior)
    inst.sort(bySize); other.sort(bySize);
    el.model.innerHTML =
      (inst.length ? `<optgroup label="✔ ${eid === OLLAMA ? "Instalados" : "Baixados"} nesta máquina">${inst.map(opt).join("")}</optgroup>` : "") +
      (other.length ? `<optgroup label="${eid === OLLAMA ? "Recomendados" : "Disponíveis"} — serão baixados ao gerar">${other.map(opt).join("")}</optgroup>` : "");
    const saved = localStorage.getItem("model:" + eid) || (eid === OLLAMA ? localStorage.getItem("model") : null);
    if (saved && data.models.some((m) => m.name === saved)) el.model.value = saved;
    else if (first) el.model.value = first.name; // primeiro instalado do catálogo (família Gemma tem prioridade)
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
      if (s.warnings && s.warnings.length) console.info("Avisos do processamento (não exibidos na tela):", s.warnings);  // ruído para o usuário; fica no console e na API
      el.bar.classList.add("bg-success");
      lastDownloadName = s.download_name || "legenda.srt";
      if (SERVER) {
        showAlert("success",
          `<strong>Legenda pronta!</strong> ${esc(s.message)}<br>
           <small class="text-break">${esc(lastDownloadName)} — será apagada do servidor após o download.</small><br>
           <button type="button" id="btnDownload" class="btn btn-success btn-sm mt-2">Baixar .srt</button>`);
        $("btnDownload").addEventListener("click", downloadServer);
      } else {
        showAlert("success",
          `<strong>Legenda gerada!</strong> ${esc(s.message)}<br>
           <small class="text-break">Salva em: ${esc(s.output_path)}</small><br>
           <a class="btn btn-success btn-sm mt-2" href="/api/jobs/${jobId}/download">Baixar .srt</a>`);
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
  function uploadFile(file, language, model, extra) {
    return new Promise((resolve) => {
      const xhr = new XMLHttpRequest();
      const fd = new FormData();
      fd.append("file", file); fd.append("language", language); fd.append("model", model);
      fd.append("engine", extra.engine); fd.append("diarize", extra.diarize ? "1" : "0"); fd.append("context", extra.context);
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
    localStorage.setItem("model:" + engineId(), el.model.value);
    const extra = { engine: engineId(), diarize: el.diarize.checked && engineInfo().supports_diarization, context: el.context.value.trim() };
    let r;
    if (SERVER) {
      const f = el.file.files[0];
      if (!f) { showAlert("danger", "Selecione um arquivo de áudio ou vídeo."); return; }
      setState("busy");
      setProgress(0, "Enviando arquivo…");
      r = await uploadFile(f, el.language.value, el.model.value, extra);
    } else {
      const res = await post("/api/jobs", { path: el.path.value, language: el.language.value, model: el.model.value, ...extra })
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

  // Cancelar é imediato: o servidor marca o job como cancelado, corta as chamadas em andamento e a tela volta limpa, pronta para outra atividade
  el.cancel.addEventListener("click", async () => {
    el.cancel.disabled = true;
    setProgress(parseFloat(el.bar.style.width) || 0, "Cancelando…", false);
    clearInterval(timer);
    const id = jobId;
    try { await post(`/api/jobs/${id}/cancel`); } catch { /* o servidor já não tem o job: a tela limpa do mesmo jeito */ }
    await discardJob();
    resetUI();
    el.cancel.disabled = false;
    showAlert("warning", "Processamento cancelado. Tudo pronto para uma nova atividade.");
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
      if (!modelsLoaded) { modelsLoaded = true; loadEngines().then(loadModels); }
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
      $("setAuto").checked = s.auto_pull; $("setChunk").value = s.chunk_seconds; $("setDevice").value = s.device || "auto"; $("setWcpp").value = s.whispercpp_path || ""; $("setWcppCuda").value = s.whispercpp_cuda_path || ""; $("setNemoCpp").value = s.nemocpp_path || "";
      $("setHf").value = ""; $("setHf").placeholder = s.hf_token_set ? "(token salvo — deixe em branco para manter)" : "hf_…";
      settingsModal.show();
    };
    $("btnSettings").addEventListener("click", openSettings);

    $("settingsForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const r = (await post("/api/settings", {
        ollama_host: $("setHost").value, ffmpeg_path: $("setFfmpeg").value, base_model: $("setBase").value,
        auto_pull: $("setAuto").checked, chunk_seconds: $("setChunk").value, device: $("setDevice").value, whispercpp_path: $("setWcpp").value, whispercpp_cuda_path: $("setWcppCuda").value, nemocpp_path: $("setNemoCpp").value,
        ...($("setHf").value.trim() ? { hf_token: $("setHf").value.trim() } : {}),  // em branco = manter o token salvo
      })).body;
      if (!r.ok) { const a = $("settingsAlert"); a.textContent = r.error; a.classList.remove("d-none"); return; }
      settingsModal.hide();
      modelsLoaded = false;
      pollSetup();
    });

    async function openDir(path, start = false) {  // start: abertura do diálogo (o servidor cai na última pasta usada)
      const d = await getJSON(`/api/browse?path=${encodeURIComponent(path || "")}${start ? "&start=1" : ""}`);
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
      openDir(cur ? cur.replace(/[\\/][^\\/]*$/, "") : "", true);
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
