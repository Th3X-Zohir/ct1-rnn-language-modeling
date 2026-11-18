// Character-level GRU language model in plain JavaScript (inference only).
// Mirrors PyTorch nn.GRU: gates ordered (r, z, n);
//   r = σ(W_ir x + b_ir + W_hr h + b_hr),  z = σ(W_iz x + b_iz + W_hz h + b_hz)
//   n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn)),  h' = (1 − z) ⊙ n + z ⊙ h
(function (root) {
  function f16ToF32(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  function decode(t) {
    const bin = typeof atob === "function" ? atob(t.b64) : Buffer.from(t.b64, "base64").toString("binary");
    const u16 = new Uint16Array(bin.length / 2);
    for (let i = 0; i < u16.length; i++) u16[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = f16ToF32(u16[i]);
    return { data: out, shape: t.shape };
  }
  const sig = (x) => 1 / (1 + Math.exp(-x));

  class CharGRU {
    constructor(meta, tensors) {
      this.meta = meta; this.H = meta.hidden; this.L = meta.layers; this.V = meta.vocab.length;
      this.stoi = Object.fromEntries(meta.vocab.map((c, i) => [c, i]));
      const T = Object.fromEntries(Object.entries(tensors).map(([k, v]) => [k, decode(v)]));
      this.emb = T["embed.weight"]; this.E = this.emb.shape[1];
      this.layers = [];
      for (let l = 0; l < this.L; l++) {
        this.layers.push({ Wih: T[`rnn.weight_ih_l${l}`], Whh: T[`rnn.weight_hh_l${l}`],
          bih: T[`rnn.bias_ih_l${l}`].data, bhh: T[`rnn.bias_hh_l${l}`].data });
      }
      this.Wo = T["head.weight"].data; this.bo = T["head.bias"].data;
      this.reset();
    }
    reset() { this.h = Array.from({ length: this.L }, () => new Float32Array(this.H)); }
    static matvec(W, x, b) {            // W: (rows x cols) row-major
      const [rows, cols] = W.shape, d = W.data, y = new Float32Array(rows);
      for (let r = 0; r < rows; r++) {
        let s = b ? b[r] : 0; const o = r * cols;
        for (let c = 0; c < cols; c++) s += d[o + c] * x[c];
        y[r] = s;
      }
      return y;
    }
    step(idx) {                          // feed one character index, return logits
      let x = this.emb.data.subarray(idx * this.E, (idx + 1) * this.E);
      const H = this.H;
      for (let l = 0; l < this.L; l++) {
        const P = this.layers[l], h = this.h[l];
        const gi = CharGRU.matvec(P.Wih, x, P.bih), gh = CharGRU.matvec(P.Whh, h, P.bhh);
        const hn = new Float32Array(H);
        for (let j = 0; j < H; j++) {
          const r = sig(gi[j] + gh[j]), z = sig(gi[H + j] + gh[H + j]);
          const n = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
          hn[j] = (1 - z) * n + z * h[j];
        }
        this.h[l] = hn; x = hn;
      }
      const logits = new Float32Array(this.V);
      for (let v = 0; v < this.V; v++) {
        let s = this.bo[v]; const o = v * H;
        for (let j = 0; j < H; j++) s += this.Wo[o + j] * x[j];
        logits[v] = s;
      }
      return logits;
    }
    feed(text) { let lg = null; for (const ch of text) if (ch in this.stoi) lg = this.step(this.stoi[ch]); return lg; }
    pick(logits, { strategy = "nucleus", temperature = 0.8, topK = 5, topP = 0.9 } = {}) {
      if (strategy === "greedy") { let b = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[b]) b = i; return b; }
      const T = Math.max(temperature, 1e-3); let m = -Infinity;
      for (const v of logits) m = Math.max(m, v / T);
      let p = Array.from(logits, (v, i) => [Math.exp(v / T - m), i]);
      const Z = p.reduce((a, [v]) => a + v, 0); p = p.map(([v, i]) => [v / Z, i]).sort((a, b) => b[0] - a[0]);
      if (strategy === "topk") p = p.slice(0, topK);
      if (strategy === "nucleus") { let c = 0, k = 0; while (k < p.length && c < topP) { c += p[k][0]; k++; } p = p.slice(0, Math.max(1, k)); }
      const S = p.reduce((a, [v]) => a + v, 0); let u = Math.random() * S;
      for (const [v, i] of p) { u -= v; if (u <= 0) return i; }
      return p[p.length - 1][1];
    }
  }
  root.CharGRU = CharGRU;
  if (typeof module !== "undefined") module.exports = { CharGRU };
})(typeof window !== "undefined" ? window : globalThis);
