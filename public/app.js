// IndicOCR Studio - Frontend Logic

(function () {
  // State
  let currentFile = null;
  let currentBase64 = null;
  let currentImageUrl = null;
  let lastResultData = null;
  let naturalImgWidth = 0;
  let naturalImgHeight = 0;

  // Configuration
  const savedApiUrl = localStorage.getItem("indicocr_api_url");
  let apiBaseUrl = savedApiUrl || (window.location.protocol.startsWith("http") ? window.location.origin : "http://localhost:8000");

  // DOM Elements
  const serverStatus = document.getElementById("serverStatus");
  const settingsBtn = document.getElementById("settingsBtn");
  const settingsPanel = document.getElementById("settingsPanel");
  const closeSettingsBtn = document.getElementById("closeSettingsBtn");
  const apiUrlInput = document.getElementById("apiUrlInput");
  const saveApiUrlBtn = document.getElementById("saveApiUrlBtn");

  const tabFileBtn = document.getElementById("tabFileBtn");
  const tabUrlBtn = document.getElementById("tabUrlBtn");
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const urlInputContainer = document.getElementById("urlInputContainer");
  const imageUrlInput = document.getElementById("imageUrlInput");
  const loadUrlBtn = document.getElementById("loadUrlBtn");

  const minConfRange = document.getElementById("minConfRange");
  const confValue = document.getElementById("confValue");

  const previewBar = document.getElementById("previewBar");
  const thumbnailImg = document.getElementById("thumbnailImg");
  const previewName = document.getElementById("previewName");
  const previewSize = document.getElementById("previewSize");
  const clearImageBtn = document.getElementById("clearImageBtn");
  const extractBtn = document.getElementById("extractBtn");
  const btnSpinner = document.getElementById("btnSpinner");
  const btnText = document.getElementById("btnText");

  const alertBox = document.getElementById("alertBox");
  const alertTitle = document.getElementById("alertTitle");
  const alertMessage = document.getElementById("alertMessage");
  const closeAlertBtn = document.getElementById("closeAlertBtn");

  const resultsSection = document.getElementById("resultsSection");
  const statTime = document.getElementById("statTime");
  const statBlocks = document.getElementById("statBlocks");
  const statConfidence = document.getElementById("statConfidence");
  const statDimensions = document.getElementById("statDimensions");

  const resultImage = document.getElementById("resultImage");
  const boxOverlay = document.getElementById("boxOverlay");
  const toggleBoxesCheck = document.getElementById("toggleBoxesCheck");
  const toggleLabelsCheck = document.getElementById("toggleLabelsCheck");

  const tabMarkdownBtn = document.getElementById("tabMarkdownBtn");
  const tabBlocksBtn = document.getElementById("tabBlocksBtn");
  const tabJsonBtn = document.getElementById("tabJsonBtn");
  const blockTabCount = document.getElementById("blockTabCount");
  const panelMarkdown = document.getElementById("panelMarkdown");
  const panelBlocks = document.getElementById("panelBlocks");
  const panelJson = document.getElementById("panelJson");

  const extractedTextPre = document.getElementById("extractedTextPre");
  const blocksList = document.getElementById("blocksList");
  const jsonViewerPre = document.getElementById("jsonViewerPre");
  const copyOutputBtn = document.getElementById("copyOutputBtn");
  const downloadJsonBtn = document.getElementById("downloadJsonBtn");

  // Init
  apiUrlInput.value = apiBaseUrl;
  checkHealth();
  setInterval(checkHealth, 15000);

  // Settings Handlers
  settingsBtn.addEventListener("click", () => {
    settingsPanel.classList.toggle("hidden");
  });
  closeSettingsBtn.addEventListener("click", () => {
    settingsPanel.classList.add("hidden");
  });
  saveApiUrlBtn.addEventListener("click", () => {
    const val = apiUrlInput.value.trim().replace(/\/+$/, "");
    if (val) {
      apiBaseUrl = val;
      localStorage.setItem("indicocr_api_url", val);
      settingsPanel.classList.add("hidden");
      checkHealth();
    }
  });

  // Health Check
  async function checkHealth() {
    try {
      const resp = await fetch(`${apiBaseUrl}/health`, { method: "GET" });
      if (resp.ok) {
        const data = await resp.json();
        serverStatus.className = "status-indicator online";
        serverStatus.querySelector(".status-text").textContent = data.model_loaded
          ? `API Ready (${data.workers_available} worker)`
          : "Loading Model...";
      } else {
        throw new Error("HTTP " + resp.status);
      }
    } catch (e) {
      serverStatus.className = "status-indicator offline";
      serverStatus.querySelector(".status-text").textContent = "API Offline";
    }
  }

  // Tab switching (File vs URL)
  tabFileBtn.addEventListener("click", () => {
    tabFileBtn.classList.add("active");
    tabUrlBtn.classList.remove("active");
    dropZone.classList.remove("hidden");
    urlInputContainer.classList.add("hidden");
  });

  tabUrlBtn.addEventListener("click", () => {
    tabUrlBtn.classList.add("active");
    tabFileBtn.classList.remove("active");
    urlInputContainer.classList.remove("hidden");
    dropZone.classList.add("hidden");
  });

  // Confidence slider
  minConfRange.addEventListener("input", (e) => {
    confValue.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // File Upload Handlers
  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Paste from clipboard
  window.addEventListener("paste", (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf("image") !== -1) {
        const file = items[i].getAsFile();
        handleFileSelected(file);
        break;
      }
    }
  });

  // URL Loader
  loadUrlBtn.addEventListener("click", () => {
    const url = imageUrlInput.value.trim();
    if (!url) return;
    currentImageUrl = url;
    currentBase64 = null;
    currentFile = null;

    thumbnailImg.src = url;
    previewName.textContent = url.split("/").pop().split("?")[0] || "Remote Image";
    previewSize.textContent = "URL source";
    previewBar.classList.remove("hidden");
    hideAlert();
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith("image/")) {
      showAlert("Invalid File", "Please select a valid image file (PNG, JPG, WebP, etc.).");
      return;
    }
    currentFile = file;
    currentImageUrl = null;

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      thumbnailImg.src = dataUrl;
      // Extract raw base64 without the data:image/...;base64, prefix
      currentBase64 = dataUrl.split(",")[1];
      previewName.textContent = file.name;
      previewSize.textContent = formatBytes(file.size);
      previewBar.classList.remove("hidden");
      hideAlert();
    };
    reader.readAsDataURL(file);
  }

  clearImageBtn.addEventListener("click", () => {
    currentFile = null;
    currentBase64 = null;
    currentImageUrl = null;
    fileInput.value = "";
    imageUrlInput.value = "";
    thumbnailImg.src = "";
    previewBar.classList.add("hidden");
    hideAlert();
  });

  // Extract Action
  extractBtn.addEventListener("click", async () => {
    if (!currentBase64 && !currentImageUrl) {
      showAlert("No Image Selected", "Please select or drop an image first.");
      return;
    }

    setLoading(true);
    hideAlert();

    const payload = {
      min_confidence: parseFloat(minConfRange.value),
    };

    if (currentBase64) {
      payload.image_base64 = currentBase64;
    } else {
      payload.image_url = currentImageUrl;
    }

    try {
      const resp = await fetch(`${apiBaseUrl}/extract`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const json = await resp.json();

      if (!resp.ok || !json.success) {
        throw new Error(json.error || `Server returned error (${resp.status})`);
      }

      displayResults(json.data);
    } catch (err) {
      console.error(err);
      showAlert(
        "Extraction Failed",
        err.message || "Failed to communicate with IndicOCR API. Ensure the API server is running."
      );
    } finally {
      setLoading(false);
    }
  });

  function setLoading(loading) {
    extractBtn.disabled = loading;
    if (loading) {
      btnSpinner.classList.remove("hidden");
      btnText.textContent = "Processing OCR...";
    } else {
      btnSpinner.classList.add("hidden");
      btnText.textContent = "Extract Text";
    }
  }

  // Display Results
  function displayResults(data) {
    lastResultData = data;
    resultsSection.classList.remove("hidden");

    // Stats
    const totalTime = data.timings?.total ? `${data.timings.total.toFixed(2)}s` : "--";
    statTime.textContent = totalTime;
    statBlocks.textContent = data.block_count || (data.blocks ? data.blocks.length : 0);
    blockTabCount.textContent = statBlocks.textContent;

    const meanConf = data.mean_confidence != null ? `${(data.mean_confidence * 100).toFixed(1)}%` : "--";
    statConfidence.textContent = meanConf;

    if (data.image_size && data.image_size.length === 2) {
      statDimensions.textContent = `${data.image_size[0]} × ${data.image_size[1]}`;
      naturalImgWidth = data.image_size[0];
      naturalImgHeight = data.image_size[1];
    }

    // Extracted Markdown
    extractedTextPre.textContent = data.text || "(No text detected)";

    // Raw JSON
    jsonViewerPre.textContent = JSON.stringify(data, null, 2);

    // Blocks list
    renderBlocksList(data.blocks || []);

    // Load Image into Canvas
    const imgSrc = currentBase64 ? `data:image/jpeg;base64,${currentBase64}` : currentImageUrl;
    resultImage.onload = () => {
      if (!naturalImgWidth || !naturalImgHeight) {
        naturalImgWidth = resultImage.naturalWidth;
        naturalImgHeight = resultImage.naturalHeight;
      }
      renderBoxes(data.blocks || []);
    };
    resultImage.src = imgSrc;

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Render bounding boxes over image
  function renderBoxes(blocks) {
    boxOverlay.innerHTML = "";
    if (!resultImage.clientWidth || !resultImage.clientHeight) return;

    const scaleX = resultImage.clientWidth / naturalImgWidth;
    const scaleY = resultImage.clientHeight / naturalImgHeight;

    blocks.forEach((block, idx) => {
      const [x1, y1, x2, y2] = block.bbox_xyxy;
      const left = x1 * scaleX;
      const top = y1 * scaleY;
      const width = (x2 - x1) * scaleX;
      const height = (y2 - y1) * scaleY;

      const box = document.createElement("div");
      const typeClass = getTypeClass(block.label, block.type);
      box.className = `ocr-box ${typeClass}`;
      box.id = `box-${idx}`;
      box.style.left = `${left}px`;
      box.style.top = `${top}px`;
      box.style.width = `${width}px`;
      box.style.height = `${height}px`;

      const labelTag = document.createElement("div");
      labelTag.className = "ocr-box-label";
      labelTag.textContent = `#${block.order} ${block.label || "Block"}`;
      box.appendChild(labelTag);

      // Hover interactions
      box.addEventListener("mouseenter", () => highlightBlock(idx, true));
      box.addEventListener("mouseleave", () => highlightBlock(idx, false));
      box.addEventListener("click", () => {
        switchTab("blocks");
        const card = document.getElementById(`card-${idx}`);
        if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      });

      boxOverlay.appendChild(box);
    });

    updateBoxVisibility();
  }

  // Render right-side blocks cards
  function renderBlocksList(blocks) {
    blocksList.innerHTML = "";
    if (!blocks.length) {
      blocksList.innerHTML = `<p style="color: var(--text-muted); font-size: 13px;">No blocks found above the confidence threshold.</p>`;
      return;
    }

    blocks.forEach((block, idx) => {
      const card = document.createElement("div");
      card.className = "block-card";
      card.id = `card-${idx}`;

      const typeClass = getTypeClass(block.label, block.type);
      const confPct = block.confidence != null ? `${(block.confidence * 100).toFixed(1)}%` : "";

      card.innerHTML = `
        <div class="block-header">
          <div class="block-meta">
            <span class="block-rank-badge">#${block.order}</span>
            <span class="legend-chip legend-${typeClass.replace('type-', '')}">${block.label || block.type}</span>
          </div>
          <span class="block-conf" title="IndicDocLayout detection confidence">${confPct}</span>
        </div>
        <div class="block-body">${escapeHtml(block.text || "")}</div>
      `;

      card.addEventListener("mouseenter", () => highlightBlock(idx, true));
      card.addEventListener("mouseleave", () => highlightBlock(idx, false));
      blocksList.appendChild(card);
    });
  }

  function highlightBlock(idx, active) {
    const box = document.getElementById(`box-${idx}`);
    const card = document.getElementById(`card-${idx}`);
    if (box) box.classList.toggle("highlighted", active);
    if (card) card.classList.toggle("highlighted", active);
  }

  // Helper for layout classification styling
  function getTypeClass(label = "", type = "") {
    const l = label.toLowerCase();
    const t = type.toLowerCase();
    if (l.includes("title") || l.includes("header") || l.includes("headline")) return "type-title";
    if (l.includes("table")) return "type-table";
    if (l.includes("image") || l.includes("advertisement") || t.includes("picture")) return "type-image";
    if (l.includes("paragraph") || t.includes("text")) return "type-para";
    return "type-other";
  }

  // Checkbox visibility toggles
  toggleBoxesCheck.addEventListener("change", updateBoxVisibility);
  toggleLabelsCheck.addEventListener("change", updateBoxVisibility);

  function updateBoxVisibility() {
    const showBoxes = toggleBoxesCheck.checked;
    const showLabels = toggleLabelsCheck.checked;

    boxOverlay.style.display = showBoxes ? "block" : "none";
    const labels = boxOverlay.querySelectorAll(".ocr-box-label");
    labels.forEach((l) => (l.style.display = showLabels ? "block" : "none"));
  }

  // Resize listener to recompute box scaling
  window.addEventListener("resize", () => {
    if (lastResultData && lastResultData.blocks) {
      renderBoxes(lastResultData.blocks);
    }
  });

  // Output Tabs
  tabMarkdownBtn.addEventListener("click", () => switchTab("markdown"));
  tabBlocksBtn.addEventListener("click", () => switchTab("blocks"));
  tabJsonBtn.addEventListener("click", () => switchTab("json"));

  function switchTab(name) {
    tabMarkdownBtn.classList.toggle("active", name === "markdown");
    tabBlocksBtn.classList.toggle("active", name === "blocks");
    tabJsonBtn.classList.toggle("active", name === "json");

    panelMarkdown.classList.toggle("active", name === "markdown");
    panelBlocks.classList.toggle("active", name === "blocks");
    panelJson.classList.toggle("active", name === "json");
  }

  // Copy & Download
  copyOutputBtn.addEventListener("click", () => {
    let content = "";
    if (panelMarkdown.classList.contains("active")) {
      content = extractedTextPre.textContent;
    } else if (panelBlocks.classList.contains("active")) {
      content = lastResultData ? lastResultData.blocks.map(b => `[#${b.order} ${b.label}]: ${b.text}`).join("\n\n") : "";
    } else {
      content = jsonViewerPre.textContent;
    }

    navigator.clipboard.writeText(content).then(() => {
      const originalText = copyOutputBtn.querySelector("span").textContent;
      copyOutputBtn.querySelector("span").textContent = "Copied!";
      setTimeout(() => (copyOutputBtn.querySelector("span").textContent = originalText), 1800);
    });
  });

  downloadJsonBtn.addEventListener("click", () => {
    if (!lastResultData) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(lastResultData, null, 2));
    const a = document.createElement("a");
    a.setAttribute("href", dataStr);
    a.setAttribute("download", `ocr_result_${Date.now()}.json`);
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  // Alerts
  function showAlert(title, message) {
    alertTitle.textContent = title;
    alertMessage.textContent = message;
    alertBox.classList.remove("hidden");
  }

  function hideAlert() {
    alertBox.classList.add("hidden");
  }

  closeAlertBtn.addEventListener("click", hideAlert);

  // Formatting helpers
  function formatBytes(bytes) {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
})();
