// ProtonPotts pH design in the browser: block descent with the per-block graphs (ADR decision 11).
//
// JS port of src/aminx/families/protonpotts_mpnn/ph_design.py (design_structure) for method "block_descent" on the Potts
// backend. The graphs do the arithmetic; this module owns the plan, the sweep, the convergence test, the uniforms and the
// bookkeeping, exactly as the host loop of tests/protonpotts/test_ph_export.py does:
//
//   table graph -> merged pair table -> field_at (chunks) -> plan (ph_plan.mjs) -> zblock per block -> pool
//   -> sweeps of visit calls until a sweep changes nothing -> energy graph and field_at for the score.
//
// Graph contracts: scripts/browser_validation/protonpotts_ph_export_gate.py (names and shapes in the manifest).

import { ProtonPottsInputError, VOCABULARY, buildProtonPottsInputs } from "./protonpotts_inputs.mjs";
import {
  DEFAULT_DEP_MAP,
  PhPlanError,
  blockTable,
  planFromCenterTypes,
  planFromExplicitCenters,
  validTokenMask,
} from "./ph_plan.mjs";
import { run } from "./protonpotts_scorer.mjs";

const SENTINEL = 1_000_000;
const THREE_OF = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY", H: "HIS", I: "ILE",
  L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER", T: "THR", W: "TRP", Y: "TYR", V: "VAL",
};
const f32 = Math.fround;

export const DEFAULT_CONFIG = Object.freeze({
  binderChain: null,
  centerTypes: ["HIS-P", "ASP-P", "GLU-P"],
  explicitCenters: [],
  depMap: DEFAULT_DEP_MAP,
  forbiddenTokens: ["HIS-A", "ASP-A", "GLU-A", "UNK"],
  temperature: 0.05,
  samplesPerSite: 2,
  combinedLambda: 0.3,
  blockSize: 3,
  neighbourK: 16,
  maxMutations: 20,
  infillScope: "neighbourhood",
  repetitiveWindowParents: ["ARG", "LYS", "HIS", "ASP", "GLU"],
  blockMaxRounds: 10,
});

/** Parent three-letter residue of a token index (X has none): HIS-P is HIS, K is LYS. */
function parentOf(index) {
  const symbol = VOCABULARY[index];
  if (symbol === "X") return null;
  return symbol.length > 1 ? symbol.slice(0, 3) : THREE_OF[symbol];
}

/** (V,) float32 mask, 1.0 where a token's parent residue is in `parents` (port of ph_descent.rep_class_mask). */
export function repClassMask(parents) {
  const wanted = new Set(parents);
  return Float32Array.from(VOCABULARY, (_s, i) => {
    const name = parentOf(i);
    return name !== null && wanted.has(name) ? 1 : 0;
  });
}

export class PhDesigner {
  /**
   * @param {object} ort onnxruntime-web
   * @param {object} manifest the PH manifest (family "protonpottsmpnn_ph"), already restricted to this bucket's entry
   * @param {object} scoringEntry the scoring manifest's entry for the same bucket (table graph meta)
   * @param {number} bucket
   * @param {{table: Uint8Array, graphs: Object<string, Uint8Array>}} models  graph bytes keyed by manifest graph name
   */
  constructor(ort, manifest, scoringEntry, bucket, sessions) {
    this.ort = ort;
    this.manifest = manifest;
    this.scoringEntry = scoringEntry;
    this.bucket = bucket;
    this.sessions = sessions; // { table, energy, [graphName]: InferenceSession }
    this.entry = manifest.buckets.find((b) => b.bucket === bucket);
    if (!this.entry) throw new Error(`PH manifest has no bucket ${bucket}`);
  }

  static async create(ort, manifest, scoringEntry, bucket, models) {
    const sessions = { table: await ort.InferenceSession.create(models.table), energy: await ort.InferenceSession.create(models.energy) };
    const entry = manifest.buckets.find((b) => b.bucket === bucket);
    for (const name of Object.keys(entry.graphs)) sessions[name] = await ort.InferenceSession.create(models.graphs[name]);
    return new PhDesigner(ort, manifest, scoringEntry, bucket, sessions);
  }

  async _run(graphName, named) {
    return run(this.ort, this.sessions[graphName], this.entry.graphs[graphName], named);
  }

  /** Conditional energies of the whole chain: chunked field_at calls (never the scatter-add full-chain function). */
  async _field(table, eIdx, seq, length) {
    const { chunk } = this.manifest;
    const vocab = VOCABULARY.length;
    const field = new Float32Array(this.bucket * vocab);
    for (let start = 0; start < length; start += chunk) {
      const positions = new Int32Array(chunk);
      for (let k = 0; k < chunk; k += 1) positions[k] = Math.min(start + k, length - 1);
      const [rows] = await this._run(`field_L${this.bucket}`, { table, e_idx: eIdx, seq: { data: seq }, positions: { data: positions } });
      for (let k = 0; k < Math.min(chunk, length - start); k += 1) {
        field.set(rows.data.subarray(k * vocab, (k + 1) * vocab), (start + k) * vocab);
      }
    }
    return field;
  }

  /**
   * Design one structure. `labels` as buildProtonPottsInputs. `uniforms` is an array of Float32Array, one per sample, needed when
   * config.temperature > 0 (the replay contract of design_structure); at temperature <= 0 none are used.
   * Returns designs sorted by final Potts energy (stable), like design_structure.
   */
  async design(pdbText, labels, config = {}, uniforms = null) {
    const cfg = { ...DEFAULT_CONFIG, ...config };
    const built = buildProtonPottsInputs(pdbText, this.bucket, labels);
    const L = built.lTotal;
    const B = this.bucket;
    const vocab = VOCABULARY.length;
    if (cfg.blockSize !== this.manifest.block_size) throw new PhPlanError(`block_size ${cfg.blockSize}: this export has ${this.manifest.block_size}`);
    if (!cfg.binderChain) throw new PhPlanError("binderChain is required for pH design");
    const sampling = cfg.temperature > 0;
    if (sampling && (!uniforms || uniforms.length !== cfg.samplesPerSite)) {
      throw new PhPlanError(`temperature > 0 needs one uniform array per sample (${cfg.samplesPerSite})`);
    }

    const [table, eIdxT] = await run(this.ort, this.sessions.table, this.scoringEntry.graphs.table, built.inputs);
    const eIdx = Int32Array.from(eIdxT.data);
    const seq = new Int32Array(B);
    seq.set(built.tokens);
    const binder = new Uint8Array(B);
    const resId = new Int32Array(B);
    built.kept.forEach(([chain, number], i) => {
      binder[i] = chain === cfg.binderChain ? 1 : 0;
      resId[i] = number;
    });
    if (!binder.some((x) => x)) throw new ProtonPottsInputError(`binder chain ${cfg.binderChain} has no designable position`);

    const slots = eIdx.length / B;
    const planOptions = { infillScope: cfg.infillScope, neighbourK: cfg.neighbourK, maxMutations: cfg.maxMutations };
    let plan;
    if (cfg.explicitCenters.length) {
      plan = planFromExplicitCenters(eIdx, B, slots, binder, resId, cfg.explicitCenters, cfg.depMap, planOptions);
    } else {
      if (!cfg.centerTypes.length) throw new PhPlanError("block_descent needs centerTypes or explicitCenters");
      const field = await this._field(table, { data: eIdx }, seq, L);
      plan = planFromCenterTypes(field, vocab, eIdx, B, slots, binder, resId, cfg.centerTypes, cfg.depMap, planOptions);
    }
    if (!plan || plan.designable.length === 0) return [];

    const P = plan.pins.length;
    const { blocks, valid } = blockTable(eIdx, slots, plan.designable, cfg.blockSize);
    const nBlocks = plan.designable.length;
    const depth = this.manifest.depth;
    const pinPositions = Int32Array.from(plan.pins.map((p) => p.position));
    const pinProt = Int32Array.from(plan.pins.map((p) => p.protIdx));
    const pinDep = new Int32Array(P * depth);
    const pinDepValid = new Uint8Array(P * depth);
    plan.pins.forEach((pin, i) => {
      pin.depIdxs.forEach((d, j) => {
        pinDep[i * depth + j] = d;
        pinDepValid[i * depth + j] = 1;
      });
    });
    const validTokens = validTokenMask(cfg.forbiddenTokens);
    const repMask = repClassMask(cfg.repetitiveWindowParents);
    const residueNumber = Int32Array.from({ length: B }, (_, i) => (binder[i] ? resId[i] : -SENTINEL - 100 * i));
    const pinned = (s) => {
      const out = Int32Array.from(s);
      pinPositions.forEach((p, i) => {
        out[p] = pinProt[i];
      });
      return out;
    };
    const tableIn = { data: table.data };
    const common = {
      table: tableIn,
      e_idx: { data: eIdx },
      pin_positions: { data: pinPositions },
      pin_prot: { data: pinProt },
      pin_dep: { data: pinDep },
      pin_dep_valid: { data: pinDepValid },
      valid_tokens: { data: validTokens },
    };
    const blockAt = (n) => ({
      block: { data: blocks.slice(n * cfg.blockSize, (n + 1) * cfg.blockSize) },
      block_valid: { data: valid.slice(n * cfg.blockSize, (n + 1) * cfg.blockSize) },
    });

    // z-scales once from the pinned seed sequence (the same for every sample), then the weights
    const seed = pinned(seq);
    const varH = new Float32Array(B);
    const varS = new Float32Array(B);
    const hasH = new Int32Array(B);
    const hasS = new Int32Array(B);
    for (let n = 0; n < nBlocks; n += 1) {
      const [vh, hh, vs, hs] = await this._run(`zblock_L${B}_P${P}`, { ...common, seq: { data: seed }, ...blockAt(n) });
      varH[n] = vh.data[0];
      hasH[n] = hh.data[0];
      varS[n] = vs.data[0];
      hasS[n] = hs.data[0];
    }
    const lam = cfg.combinedLambda;
    const [zscales, wh, wsel] = await this._run(`pool_L${B}`, {
      var_h: { data: varH }, has_h: { data: hasH }, var_s: { data: varS }, has_s: { data: hasS },
      w_h: { data: Float32Array.of(f32(1 - lam)) }, w_s: { data: Float32Array.of(f32(lam)) },
    });

    const designs = [];
    for (let sample = 0; sample < cfg.samplesPerSite; sample += 1) {
      const u = sampling ? uniforms[sample] : null;
      const work = pinned(seq);
      let draws = 0;
      let rounds = 0;
      let changed = true;
      while (changed && rounds < cfg.blockMaxRounds) {
        changed = false;
        for (let n = 0; n < nBlocks; n += 1) {
          const uniform = sampling ? u[Math.min(draws, u.length - 1)] : 0;
          const [digits] = await this._run(`visit_L${B}_P${P}`, {
            ...common,
            seq: { data: work },
            ...blockAt(n),
            rep_mask: { data: repMask },
            residue_number: { data: residueNumber },
            wh: { data: wh.data },
            wsel: { data: wsel.data },
            uniform: { data: Float32Array.of(uniform) },
            temperature: { data: Float32Array.of(cfg.temperature) },
          });
          if (sampling) draws += 1;
          for (let c = 0; c < cfg.blockSize; c += 1) {
            if (!valid[n * cfg.blockSize + c]) continue;
            const pos = blocks[n * cfg.blockSize + c];
            if (digits.data[c] !== work[pos]) changed = true;
            work[pos] = digits.data[c];
          }
        }
        rounds += 1;
      }
      designs.push(await this._score(table, eIdx, work, plan, sample, draws, L));
    }
    // stable sort by final energy (ties keep plan-then-sample order)
    return designs.map((d, i) => [d, i]).sort((a, b) => a[0].finalPottsEnergy - b[0].finalPottsEnergy || a[1] - b[1]).map(([d]) => d);
  }

  async _score(table, eIdx, work, plan, sample, draws, length) {
    const rows = this.scoringEntry.graphs.energy.input_shapes[3][0];
    const seqs = new Int32Array(rows * this.bucket);
    seqs.set(work.subarray(0, this.bucket), 0);
    const [energies] = await run(this.ort, this.sessions.energy, this.scoringEntry.graphs.energy, {
      table: { data: table.data },
      e_idx: { data: eIdx },
      pad_valid: { data: Uint8Array.from({ length: this.bucket }, (_, i) => (i < length ? 1 : 0)) },
      sequences: { data: seqs },
    });
    const gaps = [];
    if (plan.pins.length) {
      const { chunk } = this.manifest;
      const positions = new Int32Array(chunk);
      plan.pins.forEach((p, i) => {
        positions[i] = p.position;
      });
      const [fieldRows] = await this._run(`field_L${this.bucket}`, { table: { data: table.data }, e_idx: { data: eIdx }, seq: { data: work }, positions: { data: positions } });
      const vocab = VOCABULARY.length;
      plan.pins.forEach((p, i) => {
        // numpy: float32 row values, mean in float32, difference in float32, then float()
        let sum = 0;
        for (const d of p.depIdxs) sum = f32(sum + fieldRows.data[i * vocab + d]);
        const mean = f32(sum / p.depIdxs.length);
        gaps.push(f32(fieldRows.data[i * vocab + p.protIdx] - mean));
      });
    }
    return {
      sequence: Int32Array.from(work.subarray(0, length)),
      method: "block_descent",
      sample,
      pins: plan.pins,
      designable: plan.designable,
      label: plan.label,
      finalPottsEnergy: energies.data[0],
      selectiveEnergy: gaps.reduce((a, b) => a + b, 0),
      selectiveEnergies: gaps,
      nDraws: draws,
    };
  }
}
