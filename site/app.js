// aminx ProteinMPNN, in the browser. Plain ES modules, no bundler.
//
// Wiring: this file owns the DOM + the worker protocol; all RunSpec-shaped
// parsing and scoring math lives in design_utils.mjs, and all structure
// parsing lives in pdb_parse.mjs, so both stay unit-testable without a DOM.
import { MODEL_BASE } from "./config.js";
import {
  MPNN_ALPHABET,
  buildStructure,
  listChains,
  parseMmcif,
  parsePdb,
  writePdbWithBfactors,
} from "./pdb_parse.mjs";
import {
  designedMask,
  parseBias,
  parseFixedPositions,
  parseTiedPositions,
  recovery,
  scoreDesign,
  toFasta,
} from "./design_utils.mjs";

const PY2DMOL_URL = "https://py2dmol.solab.org/py2Dmol/resources/bundles/py2Dmol.embed.min.js";

const el = (id) => document.getElementById(id);
const structureIdInput = el("structure-id");
const fetchBtn = el("fetch-btn");
const structureFileInput = el("structure-file");
const chainChecklistEl = el("chain-checklist");
const numDesignsEl = el("num-designs");
const temperatureEl = el("temperature");
const baseSeedEl = el("base-seed");
const omitAaEl = el("omit-aa");
const biasAaEl = el("bias-aa");
const fixedPositionsEl = el("fixed-positions");
const tiedPositionsEl = el("tied-positions");
const threadsNoteEl = el("threads-note");
const runBtn = el("run-btn");
const statusLine = el("status-line");
const progressFill = el("progress-fill");
const viewerEl = el("viewer");
const downloadFastaBtn = el("download-fasta");
const downloadJsonBtn = el("download-json");
const resultsBody = el("results-body");

const state = {
  parsed: null, // {format, residues} from pdb_parse
  pdbText: null, // raw text of the last-loaded structure, for the viewer
  structure: null, // built once per run, over ALL designable chains
  chainsToDesign: new Set(),
  designedMaskCache: null,
  designs: [],
  worker: null,
  currentJobId: null,
  viewer: null,
};

// ---------------------------------------------------------------------
// Status / progress
// ---------------------------------------------------------------------

function setStatus(text, kind) {
  statusLine.textContent = text;
  statusLine.classList.remove("status-error", "status-ok");
  if (kind === "error") statusLine.classList.add("status-error");
  if (kind === "ok") statusLine.classList.add("status-ok");
}

function updateProgress(done, total, elapsedMs) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  progressFill.style.width = `${pct}%`;
  if (elapsedMs !== undefined) {
    setStatus(`${done}/${total} design(s) done (${(elapsedMs / 1000).toFixed(1)}s elapsed)`, "");
  }
}

function computeNumThreads() {
  const cores = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
  return self.crossOriginIsolated ? Math.min(4, cores) : 1;
}

function refreshThreadsNote() {
  const isolated = Boolean(self.crossOriginIsolated);
  threadsNoteEl.textContent = `threads: ${computeNumThreads()} (cross-origin isolated: ${isolated})`;
}

// ---------------------------------------------------------------------
// py2Dmol loading -- optional. A failed load must not block sampling.
// ---------------------------------------------------------------------

let py2DmolPromise = null;

function loadPy2Dmol() {
  if (py2DmolPromise) return py2DmolPromise;
  py2DmolPromise = new Promise((resolve) => {
    if (window.py2Dmol) {
      resolve(window.py2Dmol);
      return;
    }
    const script = document.createElement("script");
    script.src = PY2DMOL_URL;
    script.onload = () => resolve(window.py2Dmol || null);
    script.onerror = () => resolve(null);
    document.head.appendChild(script);
  });
  return py2DmolPromise;
}

async function ensureViewer() {
  if (state.viewer) return state.viewer;
  const py2Dmol = await loadPy2Dmol();
  if (!py2Dmol) {
    viewerEl.textContent = "Structure viewer unavailable: py2Dmol failed to load. "
      + "Design still works; use the download buttons below.";
    return null;
  }
  viewerEl.textContent = "";
  state.viewer = py2Dmol.show(viewerEl, state.pdbText || "", {
    width: viewerEl.clientWidth || 400,
    height: viewerEl.clientHeight || 360,
    controls: true,
  });
  state.viewer.setStyle("cartoon");
  state.viewer.setColor("chain");
  return state.viewer;
}

// ---------------------------------------------------------------------
// Structure loading
// ---------------------------------------------------------------------

function looksLikeMmcif(text) {
  const head = text.slice(0, 400);
  return /^data_/m.test(head) || head.includes("_atom_site.") || head.includes("_entry.id");
}

async function fetchStructureText(rawId) {
  const id = rawId.trim();
  if (/^[0-9A-Za-z]{4}$/.test(id)) {
    const res = await fetch(`https://files.rcsb.org/download/${id.toUpperCase()}.pdb`);
    if (res.ok) return { text: await res.text(), format: "pdb" };
  }
  const py2Dmol = await loadPy2Dmol();
  if (py2Dmol && typeof py2Dmol.fetch === "function") {
    const text = await py2Dmol.fetch(id);
    return { text, format: looksLikeMmcif(text) ? "mmcif" : "pdb" };
  }
  throw new Error(
    `could not fetch "${id}": not a 4-character RCSB id, and py2Dmol.fetch is unavailable`,
  );
}

function renderChainChecklist(chains) {
  chainChecklistEl.innerHTML = "";
  for (const chain of chains) {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.value = chain;
    checkbox.className = "chain-checkbox";
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(` Chain ${chain}`));
    chainChecklistEl.appendChild(label);
  }
}

function selectedChains() {
  return Array.from(chainChecklistEl.querySelectorAll(".chain-checkbox:checked"))
    .map((box) => box.value);
}

function resetResults() {
  state.designs = [];
  resultsBody.innerHTML = "";
  downloadFastaBtn.disabled = true;
  downloadJsonBtn.disabled = true;
  progressFill.style.width = "0%";
}

function onStructureLoaded(text, format) {
  const parsed = format === "mmcif" ? parseMmcif(text) : parsePdb(text);
  const chains = listChains(parsed);
  if (chains.length === 0) {
    setStatus("No chain with a complete backbone (N, CA, C, O) was found in this file.", "error");
    runBtn.disabled = true;
    return;
  }
  state.parsed = parsed;
  state.pdbText = text;
  renderChainChecklist(chains);
  resetResults();
  runBtn.disabled = false;
  setStatus(`Loaded ${chains.length} chain(s): ${chains.join(", ")}. Ready to run.`, "ok");
  ensureViewer().then((viewer) => {
    if (viewer) viewer.load(text, "structure");
  });
}

fetchBtn.addEventListener("click", async () => {
  const id = structureIdInput.value.trim();
  if (!id) {
    setStatus("Enter a PDB id (e.g. 1BC8) or a UniProt/AFDB id first.", "error");
    return;
  }
  setStatus(`Fetching "${id}"...`, "");
  try {
    const { text, format } = await fetchStructureText(id);
    onStructureLoaded(text, format);
  } catch (error) {
    setStatus(error.message, "error");
  }
});

structureFileInput.addEventListener("change", async () => {
  const file = structureFileInput.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const format = /\.(cif|mmcif)$/i.test(file.name) ? "mmcif" : "pdb";
    onStructureLoaded(text, format);
  } catch (error) {
    setStatus(error.message, "error");
  }
});

// ---------------------------------------------------------------------
// Worker lifecycle
// ---------------------------------------------------------------------

function ensureWorker() {
  if (state.worker) return state.worker;
  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = onWorkerMessage;
  worker.onerror = (event) => {
    setStatus(`Worker error: ${event.message || "unknown"}`, "error");
    runBtn.disabled = false;
  };
  state.worker = worker;
  return worker;
}

window.addEventListener("beforeunload", () => {
  if (state.worker) state.worker.postMessage({ type: "dispose" });
});

// ---------------------------------------------------------------------
// Running designs
// ---------------------------------------------------------------------

function clampInt(rawValue, min, max, fallback) {
  const value = parseInt(rawValue, 10);
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

function buildSequencePerDesignedChain(structure, tokens, chainsToDesign) {
  const byChain = new Map();
  for (let i = 0; i < structure.chain_ids.length; i += 1) {
    const chain = structure.chain_ids[i];
    if (!chainsToDesign.has(chain)) continue;
    if (!byChain.has(chain)) byChain.set(chain, []);
    byChain.get(chain).push(MPNN_ALPHABET[tokens[i]] ?? "X");
  }
  return Array.from(byChain.values()).map((letters) => letters.join("")).join("/");
}

async function onRun() {
  if (!state.parsed) {
    setStatus("Load a structure first.", "error");
    return;
  }
  const chains = selectedChains();
  if (chains.length === 0) {
    setStatus("Select at least one chain to design.", "error");
    return;
  }

  const structure = buildStructure(state.parsed);
  const numDesigns = clampInt(numDesignsEl.value, 1, 64, 4);
  const baseSeed = clampInt(baseSeedEl.value, 0, Number.MAX_SAFE_INTEGER, 42);
  const temperature = Number(temperatureEl.value);
  const finiteTemperature = Number.isFinite(temperature) && temperature > 0 ? temperature : 0.1;

  let fixedPositions;
  let bias;
  let tied;
  try {
    fixedPositions = parseFixedPositions(fixedPositionsEl.value, structure);
    bias = parseBias(biasAaEl.value);
    tied = parseTiedPositions(tiedPositionsEl.value, structure);
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }

  const omitAa = omitAaEl.value.trim().toUpperCase();
  const maskRunspec = { fixed_positions: fixedPositions, chains_to_design: chains };
  state.structure = structure;
  state.chainsToDesign = new Set(chains);
  state.designedMaskCache = designedMask(structure, maskRunspec);
  resetResults();

  const runspecs = [];
  for (let i = 0; i < numDesigns; i += 1) {
    runspecs.push({
      seed: baseSeed + i,
      temperature: finiteTemperature,
      omit_AA: omitAa || undefined,
      bias_AA: bias,
      fixed_positions: fixedPositions,
      tied_positions: tied.length ? tied : undefined,
      chains_to_design: chains,
      decoding_order: "random",
    });
  }

  const worker = ensureWorker();
  refreshThreadsNote();
  const numThreads = computeNumThreads();
  const jobId = `job-${Date.now()}`;
  state.currentJobId = jobId;
  runBtn.disabled = true;
  setStatus(`Running ${numDesigns} design(s)...`, "");
  updateProgress(0, numDesigns);

  worker.postMessage({
    type: "run",
    id: jobId,
    structure,
    runspecs,
    modelBase: MODEL_BASE,
    numThreads,
  });
}

runBtn.addEventListener("click", () => {
  onRun().catch((error) => {
    setStatus(error.message, "error");
    runBtn.disabled = false;
  });
});

// ---------------------------------------------------------------------
// Worker message handling + results rendering
// ---------------------------------------------------------------------

function onWorkerMessage(event) {
  const message = event.data;
  if (message.id && message.id !== state.currentJobId) return; // a stale/previous run
  if (message.type === "design") {
    handleDesignMessage(message);
  } else if (message.type === "done") {
    handleDoneMessage(message);
  } else if (message.type === "error") {
    setStatus(`Sampling failed: ${message.message}`, "error");
    runBtn.disabled = false;
  }
}

function handleDesignMessage(message) {
  const structure = state.structure;
  const mask = state.designedMaskCache;
  const score = scoreDesign(message.tokens, message.logProbs, mask, message.nReal);
  const designRecovery = recovery(message.tokens, structure.native_tokens, mask);
  const sequence = buildSequencePerDesignedChain(structure, message.tokens, state.chainsToDesign);

  const design = {
    index: message.index,
    seed: message.seed,
    temperature: message.temperature,
    score,
    recovery: designRecovery,
    sequence,
    tokens: message.tokens,
    logProbs: message.logProbs,
    nReal: message.nReal,
  };
  state.designs.push(design);
  appendResultsRow(design);
  updateProgress(message.done, message.total, message.elapsedMs);
}

function handleDoneMessage(message) {
  runBtn.disabled = false;
  const hasDesigns = state.designs.length > 0;
  downloadFastaBtn.disabled = !hasDesigns;
  downloadJsonBtn.disabled = !hasDesigns;
  setStatus(
    `Done: ${state.designs.length} design(s) in ${(message.elapsedMs / 1000).toFixed(1)}s.`,
    "ok",
  );
}

function appendResultsRow(design) {
  const row = document.createElement("tr");
  row.dataset.index = String(design.index);

  const cells = [
    String(design.index),
    String(design.seed),
    design.score.toFixed(4),
    design.recovery.toFixed(4),
  ];
  for (const value of cells) {
    const td = document.createElement("td");
    td.textContent = value;
    row.appendChild(td);
  }
  const seqCell = document.createElement("td");
  seqCell.className = "seq-cell";
  seqCell.textContent = design.sequence;
  row.appendChild(seqCell);

  row.addEventListener("click", () => selectDesign(design.index));
  resultsBody.appendChild(row);
}

async function selectDesign(index) {
  for (const row of resultsBody.querySelectorAll("tr")) {
    row.classList.toggle("selected", row.dataset.index === String(index));
  }
  const design = state.designs.find((d) => d.index === index);
  if (!design || !state.structure || !state.parsed) return;

  const viewer = await ensureViewer();
  if (!viewer) return;

  const keys = state.structure.residue_keys;
  const perResidue = [];
  for (let i = 0; i < design.nReal; i += 1) {
    const token = design.tokens[i];
    const logp = design.logProbs[i * 21 + token];
    perResidue.push({ ...keys[i], value: Math.exp(logp) });
  }
  const pdbText = writePdbWithBfactors(state.parsed, perResidue);
  viewer.load(pdbText, "structure");
  viewer.setColor("plddt");
}

// ---------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------

function downloadBlob(text, filename, mime) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

downloadFastaBtn.addEventListener("click", () => {
  downloadBlob(toFasta(state.designs), "designs.fasta", "text/plain");
});

downloadJsonBtn.addEventListener("click", () => {
  const payload = state.designs.map((d) => ({
    index: d.index,
    seed: d.seed,
    temperature: d.temperature,
    score: d.score,
    recovery: d.recovery,
    sequence: d.sequence,
  }));
  downloadBlob(JSON.stringify(payload, null, 2), "designs.json", "application/json");
});

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------

refreshThreadsNote();
