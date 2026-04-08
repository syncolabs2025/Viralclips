(() => {
  "use strict";

  // ── DOM refs ────────────────────────────────────────────────────────────
  const dropZone      = document.getElementById("drop-zone");
  const fileInput     = document.getElementById("file-input");
  const selectedFile  = document.getElementById("selected-file");
  const uploadBtn     = document.getElementById("upload-btn");
  const errorBox      = document.getElementById("error-box");

  const statusSection = document.getElementById("status-section");
  const statusSpinner = document.getElementById("status-spinner");
  const statusLabel   = document.getElementById("status-label");
  const progressFill  = document.getElementById("progress-fill");
  const progressLabel = document.getElementById("progress-label");

  const clipsSection  = document.getElementById("clips-section");
  const clipsList     = document.getElementById("clips-list");

  const downloadBtn   = document.getElementById("download-btn");

  // ── State ────────────────────────────────────────────────────────────────
  let selectedVideoFile = null;
  let pollTimer = null;
  let currentJobId = null;

  // ── File selection ────────────────────────────────────────────────────────
  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });

  function setFile(file) {
    selectedVideoFile = file;
    selectedFile.textContent = `${file.name}  (${formatBytes(file.size)})`;
    uploadBtn.disabled = false;
    clearError();
  }

  // ── Upload ────────────────────────────────────────────────────────────────
  uploadBtn.addEventListener("click", async () => {
    if (!selectedVideoFile) return;

    uploadBtn.disabled = true;
    clearError();
    downloadBtn.style.display = "none";
    clipsSection.style.display = "none";
    clipsList.innerHTML = "";

    const formData = new FormData();
    formData.append("file", selectedVideoFile);

    try {
      const res = await fetch("/upload", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Upload failed");
      }
      const { job_id } = await res.json();
      currentJobId = job_id;
      startPolling(job_id);
    } catch (err) {
      showError(err.message);
      uploadBtn.disabled = false;
    }
  });

  // ── Polling ───────────────────────────────────────────────────────────────
  function startPolling(jobId) {
    statusSection.style.display = "block";
    setStatus("Queued…", 5);

    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => poll(jobId), 3000);
    poll(jobId); // immediate first check
  }

  async function poll(jobId) {
    try {
      const res = await fetch(`/jobs/${jobId}`);
      if (!res.ok) return;
      const data = await res.json();
      handleJobUpdate(data);
    } catch (_) {
      // network hiccup — keep polling
    }
  }

  function handleJobUpdate(data) {
    const { status, status_label, clips_done, clips_total, clips, error, download_ready } = data;

    // Progress heuristic
    const pct = progressPct(status, clips_done, clips_total);
    progressFill.style.width = `${pct}%`;

    const isTerminal = status === "complete" || status === "failed";
    statusSpinner.style.display = isTerminal ? "none" : "block";
    setStatus(status_label, pct);

    if (clips_total > 0) {
      progressLabel.textContent =
        status === "reframing"
          ? `Reframing ${clips_done} / ${clips_total} clips…`
          : "";
    }

    // Render clips as soon as we have them
    if (clips && clips.length > 0) renderClips(clips);

    if (status === "complete") {
      clearInterval(pollTimer);
      if (download_ready) {
        downloadBtn.style.display = "block";
        downloadBtn.onclick = () => {
          window.location.href = `/jobs/${data.job_id}/download`;
        };
      }
    }

    if (status === "failed") {
      clearInterval(pollTimer);
      showError(error || "Processing failed. Check worker logs.");
      uploadBtn.disabled = false;
    }
  }

  function progressPct(status, done, total) {
    const map = {
      pending:      5,
      transcribing: 20,
      detecting:    40,
      reframing:    total > 0 ? 50 + Math.round((done / total) * 45) : 50,
      complete:     100,
      failed:       100,
    };
    return map[status] ?? 0;
  }

  function renderClips(clips) {
    clipsSection.style.display = "block";
    clipsList.innerHTML = "";

    clips.forEach((clip, i) => {
      const row = document.createElement("div");
      row.className = "clip-row";

      const statusBadge = clipStatusBadge(clip.status);
      row.innerHTML = `
        <div>
          <div class="clip-time">${formatTime(clip.start_time)} – ${formatTime(clip.end_time)}</div>
          <div style="margin-top:0.25rem">${statusBadge}</div>
        </div>
        <div class="clip-reason">${escapeHtml(clip.reason)}</div>
      `;
      clipsList.appendChild(row);
    });
  }

  function clipStatusBadge(status) {
    const map = {
      pending:    ["badge-pending",  "Queued"],
      processing: ["badge-running",  "Reframing"],
      complete:   ["badge-complete", "Done"],
      failed:     ["badge-failed",   "Failed"],
    };
    const [cls, label] = map[status] || ["badge-pending", status];
    return `<span class="badge ${cls}">${label}</span>`;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  function setStatus(label, pct) {
    statusLabel.textContent = label;
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.style.display = "block";
  }

  function clearError() {
    errorBox.style.display = "none";
    errorBox.textContent = "";
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  }

  function formatTime(secs) {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
})();
