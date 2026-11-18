// Verify the JavaScript GRU against PyTorch reference logits:  node demo/check_js.js
const fs = require("fs"), path = require("path"), vm = require("vm");
const ctx = { window: {}, Buffer, Math, Float32Array, Uint16Array, Object, Array, console };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, "model_weights.js"), "utf8"), ctx);
const { CharGRU } = require("./gru.js");
const ref = JSON.parse(fs.readFileSync(path.join(__dirname, "reference.json"), "utf8"));
const m = new CharGRU(ctx.window.MODEL_META, ctx.window.MODEL_TENSORS);
const t0 = Date.now(); const lg = m.feed(ref.prompt); const ms = (Date.now() - t0) / ref.prompt.length;
let maxd = 0; lg.forEach((v, i) => { maxd = Math.max(maxd, Math.abs(v - ref.logits[i])); });
const argJs = lg.indexOf(Math.max(...lg)), argPy = ref.logits.indexOf(Math.max(...ref.logits));
console.log(`max |logit diff| = ${maxd.toExponential(2)}, same argmax: ${argJs === argPy}, ${ms.toFixed(2)} ms/char`);
// short greedy continuation as a smoke test
m.reset(); let logits = m.feed(ref.prompt), s = "";
for (let i = 0; i < 120; i++) { const k = m.pick(logits, { strategy: "greedy" }); s += m.meta.vocab[k]; logits = m.step(k); }
console.log("greedy continuation:", JSON.stringify(s));
process.exit(maxd < 1e-3 && argJs === argPy ? 0 : 1);
