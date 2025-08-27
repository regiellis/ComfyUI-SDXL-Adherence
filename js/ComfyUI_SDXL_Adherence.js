// ComfyUI-SDXL-Adherence/web/js/comfyui_sdxl_adherence.js
import { app } from "../../scripts/app.js";

const EXT_NAME = "itsjustregi.sdxl_adherence_ui";
const LS_KEY_PRESETS = "sdxl_adherence_style_presets_v1";

// ---------- helpers ----------
function getConnectedNodeTitle(node, slotName) {
  try {
    const slotIndex = node.findInputSlot(slotName);
    if (slotIndex < 0) return null;
    const link = node.inputs[slotIndex]?.link;
    if (!link) return null;
    const l = app.graph.links[link];
    const srcNode = app.graph.getNodeById(l.origin_id);
    return srcNode?.title || null;
  } catch {
    return null;
  }
}

function readLastOutput(node) {
  // Comfy exposes node.widgets_values only for inputs.
  // For text outputs we rely on node._adherence_cache set in onAfterExecution.
  return node._adherence_cache || {};
}

function savePresetToLS(name, data) {
  const all = JSON.parse(localStorage.getItem(LS_KEY_PRESETS) || "{}");
  all[name] = data;
  localStorage.setItem(LS_KEY_PRESETS, JSON.stringify(all));
}
function loadPresetsFromLS() {
  return JSON.parse(localStorage.getItem(LS_KEY_PRESETS) || "{}");
}

function addToolbarButton(node, label, onClick, tooltip = "") {
  node.addWidget("button", label, null, () => onClick(node), { serialize: false, tooltip });
}

function badge(ctx, node, text, x=8, y=22) {
  ctx.save();
  ctx.fillStyle = "#222a";
  ctx.fillRect(x-6, y-10, ctx.measureText(text).width + 18, 20);
  ctx.fillStyle = "#fff";
  ctx.font = "12px Inter, ui-sans-serif";
  ctx.fillText(text, x, y+3);
  ctx.restore();
}

function warn(node, msg) {
  app.ui.dialog && app.ui.dialog.show("Adherence", msg);
  console.warn(`[${EXT_NAME}] ${msg}`, node);
}

// ---------- node decorators ----------
function decoratePromptStyler(nodeType, nodeData) {
  const origOnDrawFg = nodeType.prototype.onDrawForeground;
  nodeType.prototype.onDrawForeground = function(ctx) {
    try {
      // Read current inputs
      const style = this.widgets?.find(w => w.name === "style")?.value ?? "none";
      const widthW = this.widgets?.find(w => w.name === "width");
      const heightW = this.widgets?.find(w => w.name === "height");
      const w = parseInt(widthW?.value || 0) || "";
      const h = parseInt(heightW?.value || 0) || "";
      const dims = (w && h) ? `${w}×${h}` : "";
      const aspect = (w && h) ? (w/h) : null;

      let extra = `style:${style}`;
      if (dims) extra += ` · ${dims}`;
      if (aspect && (aspect > 1.3 || aspect < 0.8)) {
        extra += aspect > 1.3 ? " · wide" : " · tall";
      }
      badge(ctx, this, extra, 8, 18);
    } catch {}
    if (origOnDrawFg) return origOnDrawFg.apply(this, arguments);
  };

  // Buttons
  nodeType.prototype.onAdded = function() {
    // Help: Normalize Negatives
    addToolbarButton(this, "Normalize?", (n) => {
      app.ui.dialog.show(
        "Normalize Negatives",
        "Keeps factual defects (extra fingers/limbs, text, watermark, low contrast, blurry) and drops style words that fight your preset (e.g., cinematic, bokeh, hdr). Also dedupes noisy synonyms."
      );
    }, "Explain what gets kept/dropped");

    // Preview texts
    addToolbarButton(this, "Preview Splits", (n) => {
      const out = readLastOutput(n);
      const early = out.early_text ?? "<run once to populate>";
      const late  = out.late_text  ?? "<run once to populate>";
      const neg   = out.neg_text   ?? "<run once to populate>";
      const ess   = out.essentials_text ?? "<run once>";
      app.ui.dialog.show("Prompt Splits",
        `Early:\n${early}\n\nLate:\n${late}\n\nNegatives:\n${neg}\n\nEssentials:\n${ess}`);
    }, "Show early/late/neg/essentials from last execution");

    // Presets save/load
    addToolbarButton(this, "Save Preset", (n) => {
      const name = prompt("Preset name?");
      if (!name) return;
      const data = {};
      for (const w of (n.widgets||[])) {
        if (["prompt","negative"].includes(w.name)) continue;
        data[w.name] = w.value;
      }
      savePresetToLS(name, data);
      app.ui.toast?.add({text:`Saved preset "${name}"`, timeout: 2000});
    }, "Save current sliders/choices to localStorage");

    addToolbarButton(this, "Load Preset", (n) => {
      const all = loadPresetsFromLS();
      const names = Object.keys(all);
      if (!names.length) return warn(n, "No presets saved yet.");
      const name = prompt(`Preset to load?\n${names.join(", ")}`);
      if (!name || !all[name]) return;
      const data = all[name];
      for (const w of (n.widgets||[])) {
        if (data.hasOwnProperty(w.name)) w.value = data[w.name];
      }
      n.setDirtyCanvas(true);
    }, "Load a saved preset");

    // Soft warning if late_text will be empty: we can't know up-front, but hint user
    this._adherence_cache = {};
  };

  // Cache outputs after execution (so Preview Splits can show text)
  const origOnAfterExec = nodeType.prototype.onAfterExecute;
  nodeType.prototype.onAfterExecute = function(output) {
    try {
      const [early, late, neg, ess] = output?.["STRING"] || [];
      this._adherence_cache = {
        early_text: early, late_text: late, neg_text: neg, essentials_text: ess
      };
    } catch {
      // ignore
    }
    if (origOnAfterExec) return origOnAfterExec.apply(this, arguments);
  };
}

function decorateDualClipEncode(nodeType, nodeData) {
  const origOnDrawFg = nodeType.prototype.onDrawForeground;
  nodeType.prototype.onDrawForeground = function(ctx) {
    try {
      const mix = this.widgets?.find(w => w.name === "early_late_mix")?.value ?? 0.4;
      const lock = this.widgets?.find(w => w.name === "essentials_lock")?.value ?? 0.35;
      badge(ctx, this, `mix:${mix.toFixed(2)} · lock:${lock.toFixed(2)}`, 8, 18);
    } catch {}
    if (origOnDrawFg) return origOnDrawFg.apply(this, arguments);
  };

  nodeType.prototype.onAdded = function() {
    addToolbarButton(this, "Check Wiring", (n) => {
      const posOut = n.outputs?.find(o => o.name?.includes("cond_positive"));
      const negOut = n.outputs?.find(o => o.name?.includes("cond_negative"));
      if (!posOut || !negOut) return warn(n, "This encoder should output both positive and negative.");
      const posLinked = app.graph.links.some(l => l.origin_id === n.id && l.origin_slot === n.findOutputSlot(posOut.name));
      const negLinked = app.graph.links.some(l => l.origin_id === n.id && l.origin_slot === n.findOutputSlot(negOut.name));
      if (!posLinked || !negLinked) warn(n, "KSampler should receive BOTH cond_positive and cond_negative.");
      else app.ui.toast?.add({text:"Looks good: both conds connected.", timeout: 2000});
    }, "Verify both conds are connected to a sampler");
  };
}

function decorateSmartLatent(nodeType, nodeData) {
  const origOnDrawFg = nodeType.prototype.onDrawForeground;
  nodeType.prototype.onDrawForeground = function(ctx) {
    try {
      const mode = this.widgets?.find(w => w.name === "mode")?.value ?? "empty";
      const w = this.widgets?.find(w => w.name === "width")?.value ?? "";
      const h = this.widgets?.find(w => w.name === "height")?.value ?? "";
      badge(ctx, this, `${mode} · ${w||"?"}×${h||"?"}`, 8, 18);
    } catch {}
    if (origOnDrawFg) return origOnDrawFg.apply(this, arguments);
  };

  // After execute, expose actual dims in title tip
  const origOnAfterExec = nodeType.prototype.onAfterExecute;
  nodeType.prototype.onAfterExecute = function(output) {
    try {
      const dimsJson = (output?.["STRING"] || [])[0];
      if (dimsJson) {
        const info = JSON.parse(dimsJson);
        this.help = `latent: ${info.W}×${info.H} (tile ${info.tile})`;
      }
    } catch {}
    if (origOnAfterExec) return origOnAfterExec.apply(this, arguments);
  };
}

function decorateAlignHints(nodeType, nodeData) {
  const origOnDrawFg = nodeType.prototype.onDrawForeground;
  nodeType.prototype.onDrawForeground = function(ctx) {
    try {
      const snap = this.widgets?.find(w => w.name === "snap_mode")?.value ?? "pad_up";
      badge(ctx, this, `snap:${snap}`, 8, 18);
    } catch {}
    if (origOnDrawFg) return origOnDrawFg.apply(this, arguments);
  };

  nodeType.prototype.onAdded = function() {
    addToolbarButton(this, "Why align?", (n) => {
      app.ui.dialog.show(
        "Align Hints",
        "Ensures canny/depth/lineart exactly match the latent’s W×H (64-multiples). Prevents ControlNet drift and sampler shape errors."
      );
    });
  };
}

// ---------- extension registration ----------
app.registerExtension({
  name: EXT_NAME,

  async setup() {
    console.log(`[${EXT_NAME}] UI loaded`);
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const name = nodeData?.name || "";

    // Match your Python node names (use the display names you set in __init__.py)
    if (name.includes("SDXL Prompt Styler")) {
      decoratePromptStyler(nodeType, nodeData);
    }
    if (name.includes("SDXL Dual CLIP Encode")) {
      decorateDualClipEncode(nodeType, nodeData);
    }
    if (name.includes("Smart Latent")) {
      decorateSmartLatent(nodeType, nodeData);
    }
    if (name.includes("Align Hints To Latent")) {
      decorateAlignHints(nodeType, nodeData);
    }

    // Optional: generic description override
    if (name.includes("SDXL")) {
      nodeData.description = (nodeData.description || "") +
        "\n\nAdherence UI: titles show style/dims, toolbar has Preview & Presets. Encoder outputs both conds.";
    }
  },

  async nodeCreated(node, app_) {
    // Soft graph sanity checks when a KSampler is added
    if (node?.comfyClass === "KSampler") {
      setTimeout(() => {
        // find upstream DualClip node
        const linksIn = node.inputs || [];
        const posIdx = node.findInputSlot?.("positive") ?? -1;
        const negIdx = node.findInputSlot?.("negative") ?? -1;
        const posLink = posIdx >= 0 ? node.inputs[posIdx]?.link : null;
        const negLink = negIdx >= 0 ? node.inputs[negIdx]?.link : null;
        if (!posLink || !negLink) {
          warn(node, "KSampler missing positive/negative conditioning—connect from SDXL Dual CLIP Encode.");
        }
      }, 0);
    }
  }
});
