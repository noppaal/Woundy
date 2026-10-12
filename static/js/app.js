/**
 * WoundGuard AI - Frontend Controller
 * Deteksi & Klasifikasi Luka Kulit (YOLOv8 + Forward Chaining Rule-Based System)
 * Ahmad Naufal Ramadhan - Universitas Telkom 2026
 */

document.addEventListener('DOMContentLoaded', () => {
  // State Application
  const state = {
    activeTab: 'studio',
    inputMode: 'upload', // 'upload' | 'webcam' | 'sample'
    selectedFile: null,
    capturedBase64: null,
    selectedSampleFilename: null,
    confThreshold: 0.25,
    roboflowApiKey: '',
    cameraStream: null,
    cameraFacingMode: 'user', // 'user' (front/laptop webcam) | 'environment' (mobile back)
    currentDetections: [],
    userAnswers: {},
    history: JSON.parse(localStorage.getItem('wound_history') || '[]'),
    chartsInitialized: false
  };

  // DOM Elements
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const btnModeUpload = document.getElementById('btn-mode-upload');
  const btnModeWebcam = document.getElementById('btn-mode-webcam');
  const uploadContainer = document.getElementById('upload-container');
  const webcamContainer = document.getElementById('webcam-container');
  const fileInput = document.getElementById('file-input');
  const dropzone = document.getElementById('dropzone');
  const btnBrowseFile = document.getElementById('btn-browse-file');
  const webcamVideo = document.getElementById('webcam-video');
  const webcamCanvas = document.getElementById('webcam-canvas');
  const btnCaptureCam = document.getElementById('btn-capture-cam');
  const btnSwitchCam = document.getElementById('btn-switch-cam');
  const btnStopCam = document.getElementById('btn-stop-cam');
  const btnRunDetect = document.getElementById('btn-run-detect');
  const confSlider = document.getElementById('conf-slider');
  const confValLabel = document.getElementById('conf-val-label');
  const roboflowKeyInput = document.getElementById('roboflow-key-input');
  const btnToggleKeyVis = document.getElementById('btn-toggle-key-vis');
  const samplesGallery = document.getElementById('samples-gallery');

  // Result Elements
  const emptyStateView = document.getElementById('empty-state-view');
  const imageResultWrapper = document.getElementById('image-result-wrapper');
  const annotatedImageDisplay = document.getElementById('annotated-image-display');
  const loadingOverlay = document.getElementById('loading-overlay');
  const detectionChipsContainer = document.getElementById('detection-chips-container');
  const triagePanelContainer = document.getElementById('triage-panel-container');
  const triagePriorityBanner = document.getElementById('triage-priority-banner');
  const urgencyIcon = document.getElementById('urgency-icon');
  const priorityTitleText = document.getElementById('priority-title-text');
  const emergencyCallWrapper = document.getElementById('emergency-call-wrapper');
  const multiWoundAlert = document.getElementById('multi-wound-alert');
  const multiWoundText = document.getElementById('multi-wound-text');
  const interactiveQuestionsBox = document.getElementById('interactive-questions-box');
  const triageQuestionsList = document.getElementById('triage-questions-list');
  const firedRulesContainer = document.getElementById('fired-rules-container');
  const resultActionsBar = document.getElementById('result-actions-bar');
  const btnResetStudio = document.getElementById('btn-reset-studio');
  const btnDownloadResult = document.getElementById('btn-download-result');

  // ================= TAB NAVIGATION =================
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });

  function switchTab(tabId) {
    state.activeTab = tabId;
    tabButtons.forEach(b => b.classList.toggle('active', b.getAttribute('data-tab') === tabId));
    tabPanes.forEach(p => p.classList.toggle('active', p.id === `pane-${tabId}`));

    if (tabId === 'metrics' && !state.chartsInitialized) {
      loadMetricsAndRenderCharts();
    } else if (tabId === 'rules') {
      loadRulesKnowledgeBase();
    } else if (tabId === 'history') {
      renderHistoryList();
    }

    // Stop webcam if switching away from studio
    if (tabId !== 'studio' && state.cameraStream) {
      stopCamera();
    }
  }

  // ================= INPUT MODE TOGGLE =================
  btnModeUpload.addEventListener('click', () => {
    state.inputMode = 'upload';
    btnModeUpload.classList.add('active');
    btnModeWebcam.classList.remove('active');
    uploadContainer.classList.remove('hidden');
    webcamContainer.classList.add('hidden');
    stopCamera();
  });

  btnModeWebcam.addEventListener('click', () => {
    state.inputMode = 'webcam';
    btnModeWebcam.classList.add('active');
    btnModeUpload.classList.remove('active');
    uploadContainer.classList.add('hidden');
    webcamContainer.classList.remove('hidden');
    startCamera();
  });

  // ================= FILE UPLOAD & DROPZONE =================
  btnBrowseFile.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  function handleSelectedFile(file) {
    state.selectedFile = file;
    state.capturedBase64 = null;
    state.selectedSampleFilename = null;
    state.inputMode = 'upload';
    clearSampleSelection();

    // Show preview on empty state
    const reader = new FileReader();
    reader.onload = (ev) => {
      annotatedImageDisplay.src = ev.target.result;
      emptyStateView.style.display = 'none';
      imageResultWrapper.style.display = 'flex';
      btnRunDetect.disabled = false;
      btnRunDetect.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    };
    reader.readAsDataURL(file);
  }

  // ================= WEBCAM STREAM =================
  async function startCamera() {
    try {
      if (state.cameraStream) {
        stopCamera();
      }

      let stream = null;
      // Percobaan 1: Coba resolusi ideal dan mode kamera saat ini
      try {
        const constraints = {
          video: {
            facingMode: state.cameraFacingMode,
            width: { ideal: 1280 },
            height: { ideal: 720 }
          },
          audio: false
        };
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (firstErr) {
        console.warn('Mode kamera spesifik gagal, mencoba mode fallback default:', firstErr);
        // Percobaan 2: Fallback kamera standar tanpa batasan resolusi/facingMode
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }

      state.cameraStream = stream;
      webcamVideo.srcObject = stream;
    } catch (err) {
      console.warn('Webcam error:', err);
      alert('Tidak dapat mengakses webcam: ' + err.message + 
            '\n\nKemungkinan penyebab:\n' +
            '1. Kamera sedang dipakai oleh aplikasi lain (WhatsApp, Zoom, Teams, dll).\n' +
            '2. Izin kamera di browser atau Windows belum diaktifkan.\n' +
            '3. Silakan tutup aplikasi lain atau gunakan mode "Unggah Berkas".');
      btnModeUpload.click();
    }
  }

  function stopCamera() {
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach(track => track.stop());
      state.cameraStream = null;
      webcamVideo.srcObject = null;
    }
  }

  btnStopCam.addEventListener('click', stopCamera);

  btnSwitchCam.addEventListener('click', () => {
    state.cameraFacingMode = state.cameraFacingMode === 'user' ? 'environment' : 'user';
    startCamera();
  });

  btnCaptureCam.addEventListener('click', () => {
    if (!webcamVideo.videoWidth) return;
    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;
    const ctx = webcamCanvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0);
    const b64 = webcamCanvas.toDataURL('image/jpeg', 0.92);
    
    state.capturedBase64 = b64;
    state.selectedFile = null;
    state.selectedSampleFilename = null;
    clearSampleSelection();

    annotatedImageDisplay.src = b64;
    emptyStateView.style.display = 'none';
    imageResultWrapper.style.display = 'flex';
    btnRunDetect.disabled = false;

    // Trigger detection directly on snapshot
    executeDetection();
  });

  // ================= SAMPLE SHOWCASE =================
  async function loadSampleImages() {
    try {
      const resp = await fetch('/api/samples');
      const data = await resp.json();
      if (data.success && data.samples.length > 0) {
        samplesGallery.innerHTML = '';
        data.samples.forEach(s => {
          const card = document.createElement('div');
          card.className = 'sample-thumbnail-card';
          card.innerHTML = `
            <img src="${s.url}" alt="${s.label}" loading="lazy">
            <span class="sample-tag" style="color: ${s.color};">${s.label}</span>
          `;
          card.addEventListener('click', () => {
            selectSample(s.filename, card);
          });
          samplesGallery.appendChild(card);
        });
      } else {
        samplesGallery.innerHTML = '<div class="sample-placeholder">Tidak ada sampel citra.</div>';
      }
    } catch (e) {
      samplesGallery.innerHTML = '<div class="sample-placeholder">Gagal memuat sampel.</div>';
    }
  }

  function selectSample(filename, cardElement) {
    state.selectedSampleFilename = filename;
    state.selectedFile = null;
    state.capturedBase64 = null;

    clearSampleSelection();
    cardElement.classList.add('selected');

    annotatedImageDisplay.src = `/sample_images/${filename}`;
    emptyStateView.style.display = 'none';
    imageResultWrapper.style.display = 'flex';
    btnRunDetect.disabled = false;

    // Auto-trigger detection on sample click for fast UX
    executeDetection();
  }

  function clearSampleSelection() {
    document.querySelectorAll('.sample-thumbnail-card').forEach(c => c.classList.remove('selected'));
  }

  // ================= SETTINGS CONTROLS =================
  confSlider.addEventListener('input', (e) => {
    state.confThreshold = parseInt(e.target.value) / 100;
    confValLabel.textContent = `${e.target.value}%`;
  });

  roboflowKeyInput.addEventListener('input', (e) => {
    state.roboflowApiKey = e.target.value.trim();
  });

  btnToggleKeyVis.addEventListener('click', () => {
    const isPass = roboflowKeyInput.type === 'password';
    roboflowKeyInput.type = isPass ? 'text' : 'password';
    btnToggleKeyVis.textContent = isPass ? 'Tutup' : 'Lihat';
  });

  // ================= EXECUTE DETECTION =================
  btnRunDetect.addEventListener('click', () => executeDetection());

  async function executeDetection() {
    if (!state.selectedFile && !state.capturedBase64 && !state.selectedSampleFilename) {
      alert('Pilih berkas foto, gunakan kamera, atau klik sampel uji terlebih dahulu.');
      return;
    }

    loadingOverlay.style.display = 'flex';
    btnRunDetect.disabled = true;

    const formData = new FormData();
    formData.append('conf_threshold', state.confThreshold);
    if (state.roboflowApiKey) {
      formData.append('roboflow_api_key', state.roboflowApiKey);
    }
    formData.append('user_answers', JSON.stringify(state.userAnswers));

    if (state.selectedFile) {
      formData.append('image', state.selectedFile);
    } else if (state.capturedBase64) {
      formData.append('image_base64', state.capturedBase64);
    } else if (state.selectedSampleFilename) {
      formData.append('sample_filename', state.selectedSampleFilename);
    }

    try {
      const resp = await fetch('/api/detect', {
        method: 'POST',
        body: formData
      });

      const res = await resp.json();
      loadingOverlay.style.display = 'none';
      btnRunDetect.disabled = false;

      if (!res.success) {
        alert('Gagal mendeteksi luka: ' + (res.error || 'Terjadi kesalahan sistem.'));
        return;
      }

      // Update Result UI
      state.currentDetections = res.detections;
      annotatedImageDisplay.src = res.annotated_image;
      resultActionsBar.style.display = 'flex';

      renderDetectionChips(res.detections);
      renderTriagePanel(res.triage);

      // Save to History safely without blocking result
      try {
        createAndSaveHistory({
          timestamp: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          date: new Date().toLocaleDateString('id-ID'),
          imageSrc: res.annotated_image,
          detections: res.detections,
          priority: res.triage.overall_priority,
          priorityLevel: res.triage.overall_priority_level
        });
      } catch (histErr) {
        console.warn("History save error ignored:", histErr);
      }

    } catch (err) {
      loadingOverlay.style.display = 'none';
      btnRunDetect.disabled = false;
      alert('Terjadi kesalahan saat memproses citra: ' + err.message);
    }
  }

  // ================= RENDER DETECTION CHIPS =================
  function renderDetectionChips(detections) {
    detectionChipsContainer.innerHTML = '';
    detectionChipsContainer.style.display = 'flex';

    const classHex = {
      "Abrasions": "#8622FF",
      "Bruise": "#FF8000",
      "Burn": "#00FFCE",
      "Cut": "#00B7EB",
      "No Abnormality": "#22C55E",
      "Blister": "#F43F5E"
    };

    if (detections.length === 0) {
      detectionChipsContainer.innerHTML = `
        <div class="wound-chip" style="color: #22C55E; border-color: #22C55E;">
          <span>✅ Kulit Normal (No Abnormality) - 95%</span>
        </div>
      `;
      return;
    }

    detections.forEach(d => {
      const chip = document.createElement('div');
      const color = classHex[d.class_name] || '#00FFCE';
      chip.className = 'wound-chip';
      chip.style.color = color;
      chip.style.borderColor = color;
      chip.innerHTML = `
        <span>📌 ${d.class_name} (${Math.round(d.confidence * 100)}%)</span>
      `;
      detectionChipsContainer.appendChild(chip);
    });
  }

  // ================= RENDER TRIAGE PANEL =================
  function renderTriagePanel(triage) {
    triagePanelContainer.style.display = 'block';

    // 1. Urgency Banner
    const pLevel = triage.overall_priority_level;
    const pText = triage.overall_priority;

    triagePriorityBanner.className = 'priority-banner';
    if (pLevel === 0) {
      triagePriorityBanner.classList.add('priority-rendah');
      urgencyIcon.textContent = '🛡️';
      priorityTitleText.textContent = 'Kondisi Normal (Tidak Butuh Penanganan Darurat)';
      emergencyCallWrapper.style.display = 'none';
    } else if (pLevel === 1) {
      triagePriorityBanner.classList.add('priority-rendah');
      urgencyIcon.textContent = '🟢';
      priorityTitleText.textContent = 'Prioritas Rendah (Perawatan Mandiri di Rumah)';
      emergencyCallWrapper.style.display = 'none';
    } else if (pLevel === 2) {
      triagePriorityBanner.classList.add('priority-sedang');
      urgencyIcon.textContent = '🟡';
      priorityTitleText.textContent = 'Prioritas Sedang (Perawatan Higienis & Pengawasan)';
      emergencyCallWrapper.style.display = 'none';
    } else if (pLevel >= 3) {
      triagePriorityBanner.classList.add(triage.is_emergency ? 'priority-darurat' : 'priority-tinggi');
      urgencyIcon.textContent = '🚨';
      priorityTitleText.textContent = `${pText.toUpperCase()}: Segera ke Fasilitas Kesehatan / IGD!`;
      emergencyCallWrapper.style.display = 'block';
    }

    // 2. Multi-wound alert
    if (triage.is_multi_wound && triage.summary_advice.length > 0) {
      multiWoundAlert.style.display = 'flex';
      multiWoundText.textContent = triage.summary_advice.join(' ');
    } else {
      multiWoundAlert.style.display = 'none';
    }

    // 3. Interactive Questions (Forward Chaining Secondary Triage)
    renderInteractiveQuestions(triage.fired_rules);

    // 4. Fired Rules Action Cards
    renderFiredRulesCards(triage.fired_rules);
  }

  function renderInteractiveQuestions(firedRules) {
    triageQuestionsList.innerHTML = '';
    let hasQuestions = false;

    firedRules.forEach(r => {
      if (r.questions && r.questions.length > 0) {
        r.questions.forEach(q => {
          hasQuestions = true;
          const isChecked = !!state.userAnswers[q.id];
          const row = document.createElement('div');
          row.className = 'triage-question-row';
          row.innerHTML = `
            <span class="triage-question-text">${q.text}</span>
            <label class="triage-switch">
              <input type="checkbox" id="chk-${q.id}" ${isChecked ? 'checked' : ''}>
              <span class="slider-toggle"></span>
            </label>
          `;

          const inputChk = row.querySelector(`#chk-${q.id}`);
          inputChk.addEventListener('change', (e) => {
            state.userAnswers[q.id] = e.target.checked;
            refineTriageAdvice();
          });

          triageQuestionsList.appendChild(row);
        });
      }
    });

    interactiveQuestionsBox.style.display = hasQuestions ? 'block' : 'none';
  }

  async function refineTriageAdvice() {
    try {
      const resp = await fetch('/api/triage/refine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          detections: state.currentDetections,
          user_answers: state.userAnswers
        })
      });
      const data = await resp.json();
      if (data.success) {
        renderTriagePanel(data.triage);
      }
    } catch (e) {
      console.error('Refine triage error:', e);
    }
  }

  function renderFiredRulesCards(firedRules) {
    firedRulesContainer.innerHTML = '';

    firedRules.forEach(r => {
      const card = document.createElement('div');
      card.className = `rule-action-card ${r.priority_level >= 3 ? 'severity-high' : ''}`;

      const subtypeText = r.subtype ? `<span style="color: var(--accent-cyan); font-size: 0.82rem; font-weight: 600;"> &bull; ${r.subtype}</span>` : '';

      let actionsHtml = '';
      if (r.actions && r.actions.length > 0) {
        actionsHtml = `
          <ul class="action-steps-list">
            ${r.actions.map(act => `<li class="action-step-item">${act}</li>`).join('')}
          </ul>
        `;
      }

      let warningsHtml = '';
      if (r.warnings && r.warnings.length > 0) {
        warningsHtml = `
          <div class="clinical-warnings">
            <strong>⚠️ PERINGATAN MEDIS PENTING:</strong>
            ${r.warnings.map(w => `<p style="margin-top: 4px;">&bull; ${w}</p>`).join('')}
          </div>
        `;
      }

      let notesHtml = '';
      if (r.notes) {
        notesHtml = `<div class="clinical-notes">💡 Catatan Klinis: ${r.notes}</div>`;
      }

      const pColorMap = {
        "Rendah": "#22C55E",
        "Sedang": "#F59E0B",
        "Tinggi": "#EF4444",
        "N/A": "#6B7280"
      };
      const pColor = pColorMap[r.priority] || '#3B82F6';

      card.innerHTML = `
        <div class="rule-card-header">
          <div class="rule-title-group">
            <span class="rule-id-tag">${r.rule_id}</span>
            <span class="rule-wound-name">${r.label_id}</span>
            ${subtypeText}
          </div>
          <span class="rule-badge-prio" style="background: ${pColor}25; color: ${pColor}; border: 1px solid ${pColor}50;">
            Prioritas: ${r.priority}
          </span>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 8px;">
          <strong>Kondisi (IF):</strong> ${r.condition_matched}
        </div>
        ${actionsHtml}
        ${warningsHtml}
        ${notesHtml}
      `;

      firedRulesContainer.appendChild(card);
    });
  }

  // ================= RESET STUDIO =================
  btnResetStudio.addEventListener('click', () => {
    state.selectedFile = null;
    state.capturedBase64 = null;
    state.selectedSampleFilename = null;
    state.currentDetections = [];
    state.userAnswers = {};

    annotatedImageDisplay.src = '';
    imageResultWrapper.style.display = 'none';
    emptyStateView.style.display = 'block';
    detectionChipsContainer.style.display = 'none';
    triagePanelContainer.style.display = 'none';
    resultActionsBar.style.display = 'none';
    btnRunDetect.disabled = true;
    fileInput.value = '';
    clearSampleSelection();
  });

  btnDownloadResult.addEventListener('click', () => {
    if (!annotatedImageDisplay.src) return;
    const a = document.createElement('a');
    a.href = annotatedImageDisplay.src;
    a.download = `WoundGuard_Detection_${Date.now()}.jpg`;
    a.click();
  });

  // ================= METRICS TAB & CHARTS =================
  async function loadMetricsAndRenderCharts() {
    try {
      const resp = await fetch('/api/metrics');
      const data = await resp.json();
      if (!data.success) return;

      const benchmark = data.benchmark;
      const metricsPerClass = benchmark.metrics_per_class;
      const tbody = document.getElementById('metrics-table-body');
      tbody.innerHTML = '';

      const classNames = Object.keys(metricsPerClass);
      const ap50Values = [];
      const f1Values = [];

      classNames.forEach(cls => {
        const item = metricsPerClass[cls];
        ap50Values.push(item.mAP_50 * 100);
        f1Values.push(item.f1_score * 100);

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${cls}</strong></td>
          <td>${(item.precision * 100).toFixed(1)}%</td>
          <td>${(item.recall * 100).toFixed(1)}%</td>
          <td><strong style="color: var(--accent-cyan);">${(item.f1_score * 100).toFixed(1)}%</strong></td>
          <td>${(item.mAP_50 * 100).toFixed(1)}%</td>
          <td>${(item.mAP_50_95 * 100).toFixed(1)}%</td>
          <td>${item.instances}</td>
        `;
        tbody.appendChild(tr);
      });

      // 1. Chart Class Performance
      const ctxClass = document.getElementById('chart-class-performance').getContext('2d');
      new Chart(ctxClass, {
        type: 'bar',
        data: {
          labels: classNames,
          datasets: [
            {
              label: 'mAP@0.5 (%)',
              data: ap50Values,
              backgroundColor: 'rgba(0, 255, 206, 0.7)',
              borderColor: '#00FFCE',
              borderWidth: 1,
              borderRadius: 6
            },
            {
              label: 'F1-Score (%)',
              data: f1Values,
              backgroundColor: 'rgba(134, 34, 255, 0.7)',
              borderColor: '#8622FF',
              borderWidth: 1,
              borderRadius: 6
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: '#F3F4F6', font: { family: 'Plus Jakarta Sans' } } }
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              ticks: { color: '#9CA3AF' },
              grid: { color: 'rgba(255, 255, 255, 0.05)' }
            },
            x: {
              ticks: { color: '#9CA3AF' },
              grid: { display: false }
            }
          }
        }
      });

      // 2. Chart Dataset Splits
      const splits = benchmark.dataset_info.splits;
      const ctxSplit = document.getElementById('chart-dataset-split').getContext('2d');
      new Chart(ctxSplit, {
        type: 'doughnut',
        data: {
          labels: [
            `Train: ${splits.train} (${splits.train_percent}%)`,
            `Valid: ${splits.valid} (${splits.valid_percent}%)`,
            `Test: ${splits.test} (${splits.test_percent}%)`
          ],
          datasets: [{
            data: [splits.train, splits.valid, splits.test],
            backgroundColor: [
              '#00FFCE',
              '#3B82F6',
              '#8622FF'
            ],
            borderColor: '#0B0F19',
            borderWidth: 3
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'bottom',
              labels: { color: '#F3F4F6', font: { family: 'Plus Jakarta Sans' } }
            }
          }
        }
      });

      state.chartsInitialized = true;
    } catch (e) {
      console.error('Failed to load metrics:', e);
    }
  }

  // ================= RULES TAB =================
  async function loadRulesKnowledgeBase() {
    const container = document.getElementById('full-rules-table-container');
    if (container.children.length > 0) return;

    try {
      const resp = await fetch('/api/rules');
      const data = await resp.json();
      if (!data.success) return;

      const rules = data.rules;
      Object.keys(rules).forEach(k => {
        const r = rules[k];
        const card = document.createElement('div');
        card.className = 'rule-detail-card';

        let subHtml = '';
        if (r.sub_rules) {
          subHtml = `
            <div style="margin-top: 10px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
              <strong style="color: var(--accent-cyan); font-size: 0.8rem;">Diferensiasi Keparahan (Sub-Aturan):</strong>
              <div style="margin-top: 6px; font-size: 0.78rem;">
                <p><strong>A. ${r.sub_rules.mild.subtype}</strong> [Prioritas: ${r.sub_rules.mild.priority}]</p>
                <p style="margin-top: 4px;"><strong>B. ${r.sub_rules.severe.subtype}</strong> [Prioritas: ${r.sub_rules.severe.priority}]</p>
              </div>
            </div>
          `;
        }

        let actionList = '';
        if (r.actions) {
          actionList = `<ul class="action-steps-list">${r.actions.map(a => `<li class="action-step-item">${a}</li>`).join('')}</ul>`;
        }

        card.innerHTML = `
          <div class="rule-header">
            <span class="rule-id-tag" style="font-size: 0.9rem;">${r.id}</span>
            <span class="rule-badge-prio" style="background: rgba(255,255,255,0.1); color: #FFF;">
              Prioritas: ${r.base_priority}
            </span>
          </div>
          <h3 style="font-size: 1.05rem; margin-bottom: 8px; color: #FFF;">${r.label_id}</h3>
          <div class="rule-condition">
            <strong>Kondisi (IF):</strong> ${r.condition_desc}
          </div>
          <div style="margin-top: 8px;">
            <strong style="font-size: 0.82rem; color: var(--text-secondary);">Tindakan (THEN):</strong>
            ${actionList}
          </div>
          ${subHtml}
          ${r.notes ? `<div class="clinical-notes" style="margin-top: 10px;">💡 ${r.notes}</div>` : ''}
        `;
        container.appendChild(card);
      });
    } catch (e) {
      console.error('Failed to load rules:', e);
    }
  }

  // ================= DATASET SYNC =================
  const btnSyncDataset = document.getElementById('btn-sync-dataset');
  const syncRoboflowKey = document.getElementById('sync-roboflow-key');
  const syncStatusMsg = document.getElementById('sync-status-msg');

  btnSyncDataset.addEventListener('click', async () => {
    const key = syncRoboflowKey.value.trim();
    if (!key) {
      alert('Masukkan Roboflow API Key terlebih dahulu.');
      return;
    }

    btnSyncDataset.disabled = true;
    syncStatusMsg.innerHTML = '<span style="color: var(--accent-cyan);">⏳ Sedang menghubungi server Roboflow dan mengunduh dataset...</span>';

    try {
      const resp = await fetch('/api/roboflow/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key })
      });
      const res = await resp.json();
      btnSyncDataset.disabled = false;
      if (res.success) {
        syncStatusMsg.innerHTML = `<span style="color: #22C55E;">✅ ${res.message}</span>`;
      } else {
        syncStatusMsg.innerHTML = `<span style="color: #EF4444;">❌ ${res.error}</span>`;
      }
    } catch (e) {
      btnSyncDataset.disabled = false;
      syncStatusMsg.innerHTML = `<span style="color: #EF4444;">❌ Kesalahan jaringan: ${e.message}</span>`;
    }
  });

  // ================= HISTORY LOGS =================
  function createAndSaveHistory(item) {
    // Generate very small 60x60 thumbnail to save localStorage quota (< 1KB per item)
    const img = new Image();
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = 60;
        canvas.height = 60;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 60, 60);
        item.thumbSrc = canvas.toDataURL('image/jpeg', 0.6);
      } catch (e) {
        item.thumbSrc = '';
      }
      finalizeSaveHistory(item);
    };
    img.onerror = () => finalizeSaveHistory(item);
    img.src = item.imageSrc;
  }

  function finalizeSaveHistory(item) {
    // Store in RAM session
    state.history.unshift(item);
    if (state.history.length > 25) state.history.pop();

    // Prepare compact version for localStorage (strip giant imageSrc)
    const compactList = state.history.slice(0, 15).map(h => ({
      timestamp: h.timestamp,
      date: h.date,
      thumbSrc: h.thumbSrc || '',
      detections: h.detections,
      priority: h.priority,
      priorityLevel: h.priorityLevel
    }));

    try {
      localStorage.setItem('wound_history', JSON.stringify(compactList));
    } catch (e) {
      console.warn('LocalStorage quota limit reached, clearing old entries:', e);
      try {
        localStorage.removeItem('wound_history');
      } catch (ign) {}
    }
  }

  function renderHistoryList() {
    const container = document.getElementById('history-items-container');
    if (state.history.length === 0) {
      container.innerHTML = '<div class="empty-history">Belum ada riwayat deteksi pada sesi ini.</div>';
      return;
    }

    container.innerHTML = '';
    state.history.forEach((h, idx) => {
      const row = document.createElement('div');
      row.className = 'history-item';

      const detectedClasses = h.detections.map(d => d.class_name).join(', ') || 'No Abnormality';

      row.innerHTML = `
        <div class="history-item-left">
          <img src="${h.imageSrc || h.thumbSrc || ''}" class="history-thumb" alt="Thumbnail">
          <div>
            <div style="font-weight: 700; font-size: 0.9rem; color: #FFF;">${detectedClasses}</div>
            <div style="font-size: 0.75rem; color: var(--text-secondary);">${h.date} &bull; ${h.timestamp}</div>
          </div>
        </div>
        <div>
          <span class="rule-badge-prio" style="background: rgba(255,255,255,0.08); color: var(--accent-cyan);">
            ${h.priority}
          </span>
        </div>
      `;
      container.appendChild(row);
    });
  }

  const btnClearHistory = document.getElementById('btn-clear-history');
  btnClearHistory.addEventListener('click', () => {
    state.history = [];
    localStorage.removeItem('wound_history');
    renderHistoryList();
  });

  // Load sample images immediately on page start
  loadSampleImages();
});
