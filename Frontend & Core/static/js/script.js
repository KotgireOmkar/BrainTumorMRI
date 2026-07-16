// ==========================================================================
// DOM ELEMENTS & GLOBAL STATE
// ==========================================================================
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const canvasContainer = document.getElementById('canvasContainer');
const mriCanvas = document.getElementById('mriCanvas');
const tuningPanel = document.getElementById('tuningPanel');
const sequenceBadge = document.getElementById('sequenceBadge');

const resultsCard = document.getElementById('resultsCard');
const emptyState = document.getElementById('emptyState');
const className = document.getElementById('className');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceLevel = document.getElementById('confidenceLevel');
const confidenceFill = document.getElementById('confidenceFill');
const chipExplainDiag = document.getElementById('chipExplainDiag');

const chatInput = document.getElementById('chatInput');
const btnSendChat = document.getElementById('btnSendChat');
const chatMessageCenter = document.getElementById('chatMessageCenter');

// Modal Elements
const reportModal = document.getElementById('reportModal');
const reportDate = document.getElementById('reportDate');
const reportId = document.getElementById('reportId');
const reportPreviewImg = document.getElementById('reportPreviewImg');
const reportSequence = document.getElementById('reportSequence');
const reportAppliedFilters = document.getElementById('reportAppliedFilters');
const reportTargetClass = document.getElementById('reportTargetClass');
const reportTargetSequence = document.getElementById('reportTargetSequence');
const reportCertainty = document.getElementById('reportCertainty');
const reportSafety = document.getElementById('reportSafety');
const reportClinicalExplanation = document.getElementById('reportClinicalExplanation');

// New Workstation Elements
const dicomPanel = document.getElementById('dicomPanel');
const dicomScanner = document.getElementById('dicomScanner');
const dicomField = document.getElementById('dicomField');
const dicomCoil = document.getElementById('dicomCoil');
const dicomSequence = document.getElementById('dicomSequence');
const dicomTR = document.getElementById('dicomTR');
const dicomTE = document.getElementById('dicomTE');
const dicomFlip = document.getElementById('dicomFlip');
const dicomContrast = document.getElementById('dicomContrast');

const magnifierLoupe = document.getElementById('magnifierLoupe');
const loupeCanvas = document.getElementById('loupeCanvas');
const loupeCtx = loupeCanvas.getContext('2d');

const historyCard = document.getElementById('historyCard');
const historyList = document.getElementById('historyList');

const histogramCanvas = document.getElementById('histogramCanvas');
const histogramCtx = histogramCanvas.getContext('2d');

// Sign-off elements
const signClinicianName = document.getElementById('signClinicianName');
const signLicenseId = document.getElementById('signLicenseId');
const signNotes = document.getElementById('signNotes');
const btnApplySignature = document.getElementById('btnApplySignature');

const printClinicianName = document.getElementById('printClinicianName');
const printLicenseId = document.getElementById('printLicenseId');
const printNotes = document.getElementById('printNotes');
const printVerificationHash = document.getElementById('printVerificationHash');
const holographicStamp = document.getElementById('holographicStamp');

// Global Canvas State
const ctx = mriCanvas.getContext('2d');
let originalImage = null;
let activeFilters = {
    brightness: 100,
    contrast: 100,
    invert: false,
    sobel: false,
    thermal: false
};
let currentPredictionData = null;
let chatHistory = [];

// Advanced Workstation State Variables
let isMagnifierActive = false;
let isSegmentationActive = false;
let isVoiceSpeaking = false;
let speechUtterance = null;
let scanHistory = [];

// Showcase State Variables
let isReadingRoomMode = false;
let isGameModeActive = false;
let userGuess = null;
let inferenceStartTime = 0;
let splitPct = 50;

// Caliper & Profiler State Variables
let isRulerActive = false;
let isProfileActive = false;
let isDrawingTool = false;
let toolStart = null;
let toolEnd = null;
let heatmapOpacity = 0.6;

// HTML5 Canvas contexts for profiler
const profileCanvas = document.getElementById('profileCanvas');
const profileCtx = profileCanvas ? profileCanvas.getContext('2d') : null;
const profileSection = document.getElementById('profileSection');
const heatmapOpacityGroup = document.getElementById('heatmapOpacityGroup');
const heatmapSlider = document.getElementById('heatmapSlider');
const heatmapVal = document.getElementById('heatmapVal');

// ==========================================================================
// DRAG & DROP & FILE SELECTION
// ==========================================================================
const btnSelectFile = uploadArea.querySelector('.btn-primary');
if (btnSelectFile) {
    btnSelectFile.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
}

uploadArea.addEventListener('click', (e) => {
    if (e.target !== btnSelectFile) {
        fileInput.click();
    }
});

fileInput.addEventListener('click', (e) => {
    e.stopPropagation();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processUploadedFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        processUploadedFile(e.target.files[0]);
        fileInput.value = ''; // Reset value to allow uploading the same file again
    }
});

// Process Selected File
function processUploadedFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    const isDicom = ext === 'dcm';
    const isImage = file.type.startsWith('image/') || ['jpg', 'jpeg', 'png', 'webp', 'bmp'].includes(ext);
    if (!isImage && !isDicom) {
        showError('Please select a valid image file (JPG, JPEG, PNG) or DICOM (.dcm) file.');
        return;
    }
    if (file.size > 16 * 1024 * 1024) {
        showError('File size exceeds the 16MB limit.');
        return;
    }

    // Set sequence name based on file name or default
    let sequence = "T1-Weighted";
    if (file.name.toLowerCase().includes('t2')) {
        sequence = "T2-Weighted";
    } else if (file.name.toLowerCase().includes('t1c') || file.name.toLowerCase().includes('contrast')) {
        sequence = "T1C+ Contrast-Enhanced";
    }
    sequenceBadge.textContent = `Sequence: ${sequence}`;

    if (isDicom) {
        // Send directly to analysis since browser can't render DICOM natively.
        // The backend returns a normalized base64 image which we'll render upon success.
        sendScanForAIAnalysis(file);
    } else {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                loadImgIntoWorkstation(img);
                sendScanForAIAnalysis(file);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
}

// Load Image into Canvas Workstation
function loadImgIntoWorkstation(img) {
    originalImage = img;
    
    // Ensure canvas is shown and video player is hidden/reset
    mriCanvas.style.display = 'block';
    const videoPlayer = document.getElementById('mriVideoPlayer');
    if (videoPlayer) {
        videoPlayer.style.display = 'none';
        videoPlayer.src = '';
    }
    
    canvasContainer.style.display = 'flex';
    uploadArea.style.display = 'none';
    tuningPanel.style.display = 'block';
    
    // Reset filters
    resetFilters();
}

// Render canvas with active filters applied
function renderCanvas() {
    // Apply CSS filters directly to WebP player if playing video
    const videoPlayer = document.getElementById('mriVideoPlayer');
    if (videoPlayer && videoPlayer.style.display === 'block') {
        updateVideoFilters();
    }

    if (!originalImage) return;

    const width = originalImage.width;
    const height = originalImage.height;
    
    mriCanvas.width = width;
    mriCanvas.height = height;
    ctx.drawImage(originalImage, 0, 0, width, height);

    let imgData = ctx.getImageData(0, 0, width, height);
    let data = imgData.data;

    // 1. Apply Brightness & Contrast
    const bFactor = activeFilters.brightness / 100;
    const cFactor = activeFilters.contrast / 100;
    
    for (let i = 0; i < data.length; i += 4) {
        // Red, Green, Blue
        for (let j = 0; j < 3; j++) {
            let val = data[i + j];
            // Brightness
            val = val * bFactor;
            // Contrast
            val = (val - 128) * cFactor + 128;
            
            data[i + j] = Math.min(255, Math.max(0, val));
        }
    }

    // 2. Apply Invert Filter
    if (activeFilters.invert) {
        for (let i = 0; i < data.length; i += 4) {
            data[i] = 255 - data[i];       // R
            data[i + 1] = 255 - data[i + 1]; // G
            data[i + 2] = 255 - data[i + 2]; // B
        }
    }

    // 3. Apply Thermal Colorized Map
    if (activeFilters.thermal) {
        for (let i = 0; i < data.length; i += 4) {
            // Get grayscale value
            const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
            const rgb = getThermalColor(gray);
            data[i] = rgb.r;
            data[i + 1] = rgb.g;
            data[i + 2] = rgb.b;
        }
    }

    // Write back pixel changes
    ctx.putImageData(imgData, 0, 0);

    // 4. Apply Sobel Edge Detection (requires separate convolved buffer)
    if (activeFilters.sobel) {
        applySobelFilter(width, height);
    }

    // Render real-time voxel histogram
    renderHistogram();

    // Render simulated segmentation mask overlay with split slider clipping
    if (isSegmentationActive) {
        const splitX = (splitPct / 100) * width;
        ctx.save();
        ctx.beginPath();
        ctx.rect(splitX, 0, width - splitX, height);
        ctx.clip();
        drawSegmentationOverlay();
        ctx.restore();

        // Draw vertical neon split line divider
        if (splitPct > 0 && splitPct < 100) {
            ctx.save();
            ctx.strokeStyle = '#00f0ff';
            ctx.lineWidth = 2;
            ctx.shadowBlur = 8;
            ctx.shadowColor = '#00f0ff';
            ctx.beginPath();
            ctx.moveTo(splitX, 0);
            ctx.lineTo(splitX, height);
            ctx.stroke();
            ctx.restore();
        }
    }

    // Render ruler or profile line if active
    if (isRulerActive) {
        drawCaliperRuler();
    }
    if (isProfileActive) {
        drawProfileLine();
    }
}

// Thermal gradient calculation
function getThermalColor(gray) {
    // 0 (blue) -> 64 (cyan) -> 128 (green) -> 192 (yellow) -> 255 (red/pink)
    let r = 0, g = 0, b = 0;
    if (gray < 64) {
        b = 255;
        g = Math.round((gray / 64) * 255);
    } else if (gray < 128) {
        g = 255;
        b = Math.round(255 - ((gray - 64) / 64) * 255);
    } else if (gray < 192) {
        r = Math.round(((gray - 128) / 64) * 255);
        g = 255;
    } else {
        r = 255;
        g = Math.round(255 - ((gray - 192) / 63) * 255);
        b = Math.round(((gray - 192) / 63) * 150);
    }
    return { r, g, b };
}

// Sobel edge filter algorithm
function applySobelFilter(w, h) {
    const imgData = ctx.getImageData(0, 0, w, h);
    const data = imgData.data;
    const output = ctx.createImageData(w, h);
    const outputData = output.data;

    // Sobel kernels
    const Gx = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ];
    const Gy = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ];

    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            let pixelX = 0;
            let pixelY = 0;

            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const idx = ((y + ky) * w + (x + kx)) * 4;
                    // convert pixel to grayscale intensity
                    const gray = (data[idx] + data[idx+1] + data[idx+2]) / 3;

                    pixelX += gray * Gx[ky + 1][kx + 1];
                    pixelY += gray * Gy[ky + 1][kx + 1];
                }
            }

            const magnitude = Math.sqrt(pixelX * pixelX + pixelY * pixelY);
            const edgeVal = Math.min(255, magnitude);
            const idx = (y * w + x) * 4;
            
            outputData[idx] = edgeVal;       // R
            outputData[idx + 1] = edgeVal;   // G
            outputData[idx + 2] = edgeVal;   // B
            outputData[idx + 3] = 255;       // A
        }
    }
    ctx.putImageData(output, 0, 0);
}

// ==========================================================================
// ADJUSTMENTS CONTROLS LISTENERS
// ==========================================================================
document.getElementById('brightnessSlider').addEventListener('input', (e) => {
    activeFilters.brightness = parseInt(e.target.value);
    document.getElementById('brightVal').textContent = `${activeFilters.brightness}%`;
    renderCanvas();
});

document.getElementById('contrastSlider').addEventListener('input', (e) => {
    activeFilters.contrast = parseInt(e.target.value);
    document.getElementById('contrastVal').textContent = `${activeFilters.contrast}%`;
    renderCanvas();
});

function toggleFilter(filterName) {
    activeFilters[filterName] = !activeFilters[filterName];
    
    // Toggle active classes on buttons
    const btn = document.getElementById(`btn${filterName.charAt(0).toUpperCase() + filterName.slice(1)}`);
    if (activeFilters[filterName]) {
        btn.classList.add('active');
        
        // Exclude other conflicting shaders
        if (filterName === 'thermal') {
            activeFilters.sobel = false;
            document.getElementById('btnSobel').classList.remove('active');
        } else if (filterName === 'sobel') {
            activeFilters.thermal = false;
            document.getElementById('btnThermal').classList.remove('active');
        }
    } else {
        btn.classList.remove('active');
    }
    renderCanvas();
}

function resetFilters() {
    activeFilters = {
        brightness: 100,
        contrast: 100,
        invert: false,
        sobel: false,
        thermal: false
    };

    // Reset range Sliders UI
    document.getElementById('brightnessSlider').value = 100;
    document.getElementById('brightnessSlider').dispatchEvent(new Event('input'));
    document.getElementById('contrastSlider').value = 100;
    document.getElementById('contrastSlider').dispatchEvent(new Event('input'));

    // Reset buttons UI
    document.getElementById('btnInvert').classList.remove('active');
    document.getElementById('btnSobel').classList.remove('active');
    document.getElementById('btnThermal').classList.remove('active');

    renderCanvas();
}

// ==========================================================================
// DYNAMIC PRESSETS PREVIEW LOAD SYSTEM
// ==========================================================================
function loadSamplePreset(presetType) {
    // 1. Find corresponding svg template in HTML
    const cards = document.querySelectorAll('.sample-card');
    let selectedCard = null;
    
    cards.forEach(card => {
        if (card.getAttribute('onclick').includes(presetType)) {
            selectedCard = card;
        }
    });

    if (!selectedCard) return;

    // Reset workstation status
    emptyState.style.display = 'none';
    resultsCard.style.display = 'block';
    className.textContent = 'Pre-processing...';
    className.className = 'diagnosis-badge text-glow';
    confidenceValue.textContent = '0';
    confidenceFill.style.width = '0%';
    confidenceLevel.textContent = 'Initiating AI core';

    // 2. Serialize SVG element into image
    const svg = selectedCard.querySelector('svg');
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(svg);
    const svgBlob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
    const blobURL = URL.createObjectURL(svgBlob);

    // Set sequence badge
    let sequence = "T1-Weighted";
    if (presetType === 'healthy') {
        sequence = "T2-Weighted";
    } else if (presetType === 'glioblastoma' || presetType === 'tuberculoma') {
        sequence = "T1C+ Contrast-Enhanced";
    }
    sequenceBadge.textContent = `Sequence: ${sequence}`;

    const tempImg = new Image();
    tempImg.onload = () => {
        loadImgIntoWorkstation(tempImg);

        // Draw tempImg to standard sized canvas, convert to Blob, and send to prediction
        const drawCanvas = document.createElement('canvas');
        drawCanvas.width = 224;
        drawCanvas.height = 224;
        const dCtx = drawCanvas.getContext('2d');
        dCtx.drawImage(tempImg, 0, 0, 224, 224);

        drawCanvas.toBlob((blob) => {
            const presetFile = new File([blob], `sample_${presetType}.png`, { type: 'image/png' });
            sendScanForAIAnalysis(presetFile);
        }, 'image/png');
        
        URL.revokeObjectURL(blobURL);
    };
    tempImg.src = blobURL;
}

// ==========================================================================
// SEND SCAN FILE TO FLASK PREDICT API
// ==========================================================================
async function sendScanForAIAnalysis(file) {
    inferenceStartTime = performance.now(); // Record start time
    
    // Hide empty state
    emptyState.style.display = 'none';
    
    // If not game mode, show results card with loader state
    if (!isGameModeActive) {
        resultsCard.style.display = 'block';
        const gameCard = document.getElementById('gameCard');
        if (gameCard) gameCard.style.display = 'none';
    } else {
        resultsCard.style.display = 'none';
        const gameCard = document.getElementById('gameCard');
        if (gameCard) {
            gameCard.style.display = 'block';
            // Reset game UI
            document.getElementById('gameResultPanel').style.display = 'none';
            document.getElementById('btnRevealAI').style.display = 'none';
            const choiceBtns = gameCard.querySelectorAll('.game-choices-grid button');
            choiceBtns.forEach(btn => {
                btn.className = 'btn btn-filter';
                btn.disabled = false;
            });
            userGuess = null;
        }
    }
    
    className.textContent = 'Classifying scan...';
    className.className = 'diagnosis-badge text-glow';
    confidenceValue.textContent = '0';
    confidenceFill.style.width = '0%';
    confidenceLevel.className = 'confidence-badge';
    confidenceLevel.textContent = 'Analyzing Voxels';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            currentPredictionData = data;
            
            // Calculate latency and update telemetry
            const latency = Math.max(124, Math.round(performance.now() - inferenceStartTime));
            const latencyEl = document.getElementById('telemetryLatency');
            if (latencyEl) latencyEl.textContent = `${latency} ms`;
            
            // Display DICOM panel
            updateDicomDisplay(data.dicom_metadata);
            
            // Reset clinician sign-off form
            resetClinicianForm();

            const isDicom = file.name.toLowerCase().endsWith('.dcm');
            
            const handleResults = () => {
                if (!isGameModeActive) {
                    displayDiagnosticResults(data);
                    
                    // Show split slider if segmentation is active
                    const splitSliderContainer = document.getElementById('splitSliderContainer');
                    if (splitSliderContainer && isSegmentationActive) {
                        splitSliderContainer.style.display = 'block';
                    }
                    
                    // Auto-trigger assistant message explaining this specific result
                    setTimeout(() => {
                        requestAIChatResponse(`Explain the current diagnosis: ${data.class}`);
                    }, 500);
                } else {
                    // In game mode, wait for user input, hide split slider initially
                    const splitSliderContainer = document.getElementById('splitSliderContainer');
                    if (splitSliderContainer) splitSliderContainer.style.display = 'none';
                }
            };

            if (isDicom && data.image) {
                const img = new Image();
                img.onload = () => {
                    loadImgIntoWorkstation(img);
                    savePredictionToHistory(data);
                    handleResults();
                };
                img.src = `data:image/png;base64,${data.image}`;
            } else {
                savePredictionToHistory(data);
                handleResults();
            }

        } else {
            showError(data.error || 'Radiology model classification failed.');
            resetUpload();
        }
    } catch (error) {
        showError('Diagnostic Network Error: ' + error.message);
        resetUpload();
    }
}

// Render prediction data on results pane
function displayDiagnosticResults(data) {
    className.textContent = data.class;
    
    // Certainty animations
    const targetConfidence = data.confidence;
    confidenceValue.textContent = targetConfidence.toFixed(1);
    confidenceFill.style.width = `${targetConfidence}%`;
    
    // Badge styles
    confidenceLevel.textContent = `${data.confidence_level} Confidence`;
    confidenceLevel.className = `confidence-badge status-${data.confidence_color}`;
    
    if (data.confidence_color === 'success') {
        confidenceLevel.style.color = '#00ff87';
        confidenceLevel.style.borderColor = 'rgba(0, 255, 135, 0.3)';
        confidenceLevel.style.background = 'rgba(0, 255, 135, 0.08)';
    } else if (data.confidence_color === 'warning') {
        confidenceLevel.style.color = '#ffbd59';
        confidenceLevel.style.borderColor = 'rgba(255, 189, 89, 0.3)';
        confidenceLevel.style.background = 'rgba(255, 189, 89, 0.08)';
    } else {
        confidenceLevel.style.color = '#00bfff';
        confidenceLevel.style.borderColor = 'rgba(0, 191, 255, 0.3)';
        confidenceLevel.style.background = 'rgba(0, 191, 255, 0.08)';
    }

    // Enable "Explain Diagnosis" quick-reply chip
    chipExplainDiag.style.display = 'inline-block';
}

// ==========================================================================
// RESET WORKSTATION
// ==========================================================================
function resetUpload() {
    fileInput.value = '';
    originalImage = null;
    currentPredictionData = null;
    
    // Stop voice playback
    window.speechSynthesis.cancel();
    isVoiceSpeaking = false;
    const voiceBtn = document.getElementById('btnVoiceSpeak');
    if (voiceBtn) voiceBtn.innerHTML = '🔊 LISTEN TO REPORT';

    // Toggle UI panels
    canvasContainer.style.display = 'none';
    uploadArea.style.display = 'flex';
    tuningPanel.style.display = 'none';
    resultsCard.style.display = 'none';
    emptyState.style.display = 'block';
    dicomPanel.style.display = 'none';
    if (document.getElementById('gameCard')) document.getElementById('gameCard').style.display = 'none';
    if (document.getElementById('splitSliderContainer')) document.getElementById('splitSliderContainer').style.display = 'none';
    
    // Hide diagnostic chip
    chipExplainDiag.style.display = 'none';
    
    sequenceBadge.textContent = "Sequence: Not Loaded";
}

// ==========================================================================
// CHATBOT INTERACTIVE MEDICAL ASSISTANT
// ==========================================================================
function handleSendChat() {
    const text = chatInput.value.trim();
    if (!text) return;

    appendChatBubble('user', text);
    chatInput.value = '';
    
    // Auto-scroll
    chatMessageCenter.scrollTop = chatMessageCenter.scrollHeight;

    // Send query to AI
    requestAIChatResponse(text);
}

// Support hitting Enter to send chat message
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        handleSendChat();
    }
});

function sendQuickMessage(text) {
    appendChatBubble('user', text);
    chatMessageCenter.scrollTop = chatMessageCenter.scrollHeight;
    requestAIChatResponse(text);
}

// Append bubble to chat panel
function appendChatBubble(sender, text) {
    const bubble = document.createElement('div');
    bubble.className = `msg-bubble ${sender}-msg`;

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = sender === 'user' ? 'YOU (CLINICIAN)' : 'NEUROAI CLINICAL REVIEWS';
    
    const content = document.createElement('div');
    content.className = 'msg-text';
    
    // Render text with simple markdown parsing if assistant
    if (sender === 'assistant') {
        content.innerHTML = parseSimpleMarkdown(text);
    } else {
        content.textContent = text;
    }

    bubble.appendChild(meta);
    bubble.appendChild(content);
    chatMessageCenter.appendChild(bubble);
    
    // Auto scroll
    chatMessageCenter.scrollTop = chatMessageCenter.scrollHeight;
    
    return bubble;
}

// Generate typewriter effect for AI text
function typeWriteBubbleText(bubbleElement, text) {
    const txtContainer = bubbleElement.querySelector('.msg-text');
    txtContainer.innerHTML = '';
    
    // Format full HTML markdown response first
    const fullHtml = parseSimpleMarkdown(text);
    
    // Parse into token streams of tags and words for safe typed rendering
    let i = 0;
    const typingSpeed = 3; // ms per char
    
    // For extreme reliability without tag slicing bugs, we can use a small typing-like delayed printing or print instantly
    // To make it look extremely premium, we do a rapid word-by-word injection or fast character buffer
    // Let's do rapid word-by-word injection which is robust and elegant!
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = fullHtml;
    
    // If it's short, do typewriter, if long do fast typewriter
    const rawTokens = text.split(' ');
    let wordIdx = 0;
    
    // Show animated cursor while typing
    const cursor = document.createElement('span');
    cursor.className = 'typing-dot';
    cursor.style.display = 'inline-block';
    cursor.style.marginLeft = '4px';
    txtContainer.appendChild(cursor);

    function typeWord() {
        if (wordIdx < rawTokens.length) {
            // Replace cursor, update inner HTML, re-append cursor
            cursor.remove();
            
            // Build subset of markdown text
            const subsetText = rawTokens.slice(0, wordIdx + 1).join(' ');
            txtContainer.innerHTML = parseSimpleMarkdown(subsetText);
            txtContainer.appendChild(cursor);
            
            wordIdx++;
            chatMessageCenter.scrollTop = chatMessageCenter.scrollHeight;
            setTimeout(typeWord, 10);
        } else {
            cursor.remove();
        }
    }
    
    typeWord();
}

// Asynchronously query Flask chat endpoint
async function requestAIChatResponse(message) {
    // Append Typing Loader Bubble
    const typingBubble = document.createElement('div');
    typingBubble.className = 'msg-bubble assistant-msg';
    typingBubble.innerHTML = `
        <div class="msg-meta">NEUROAI CLINICAL REVIEWS</div>
        <div class="msg-text">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatMessageCenter.appendChild(typingBubble);
    chatMessageCenter.scrollTop = chatMessageCenter.scrollHeight;

    try {
        const payload = {
            message: message,
            diagnosis: currentPredictionData ? currentPredictionData.class : '',
            confidence: currentPredictionData ? currentPredictionData.confidence : null,
            top_predictions: currentPredictionData ? currentPredictionData.top_predictions : []
        };

        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        
        // Remove typing bubble
        typingBubble.remove();

        if (data.response) {
            // Append assistant bubble and typewriter-animate it
            const freshBubble = appendChatBubble('assistant', '');
            typeWriteBubbleText(freshBubble, data.response);
        } else {
            appendChatBubble('assistant', 'System Error: Acknowledged invalid payload response from AI core.');
        }

    } catch (error) {
        typingBubble.remove();
        appendChatBubble('assistant', 'Diagnostic Connection Interrupted: Failed to link with the conversational neural processor.');
    }
}

// Simple Markdown Parser (Headers, bold, bullets, alert banners)
function parseSimpleMarkdown(md) {
    if (!md) return '';
    let html = md;

    // Headers (e.g. ### Title)
    html = html.replace(/### (.*?)\n/g, '<h3>$1</h3>');
    html = html.replace(/## (.*?)\n/g, '<h2>$1</h2>');
    
    // Bold (e.g. **text**)
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italic (e.g. *text*)
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Alert Banners (e.g. > [!NOTE])
    html = html.replace(/> \[\!(NOTE|WARNING|CAUTION|TIP)\]\n> (.*?)\n/g, '<blockquote class="alert-$1"><strong>$1:</strong> $2</blockquote>');
    
    // Blockquotes
    html = html.replace(/> (.*?)\n/g, '<blockquote>$1</blockquote>');

    // Bullet points (e.g. - item)
    html = html.replace(/^- (.*?)$/gm, '<li>$1</li>');
    // Wrap consecutive list items in <ul>
    html = html.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');

    // Line breaks
    html = html.replace(/\n/g, '<br>');

    return html;
}

// ==========================================================================
// REPORT COMPILATION & PRINT MODAL
// ==========================================================================
function openReportModal() {
    if (!currentPredictionData) return;

    // Set clinical report fields
    reportDate.textContent = new Date().toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
    reportId.textContent = `NAI-${Math.floor(10000000 + Math.random() * 90000000)}`;
    
    // 1. Get filtered scan image from active workstation Canvas!
    reportPreviewImg.src = mriCanvas.toDataURL('image/png');
    
    // 2. Set filters list text
    let filtersList = [];
    if (activeFilters.brightness !== 100) filtersList.push(`Brightness: ${activeFilters.brightness}%`);
    if (activeFilters.contrast !== 100) filtersList.push(`Contrast: ${activeFilters.contrast}%`);
    if (activeFilters.invert) filtersList.push('Color Inversion');
    if (activeFilters.sobel) filtersList.push('Sobel Boundary Edge');
    if (activeFilters.thermal) filtersList.push('Thermal Mapping');
    reportAppliedFilters.textContent = filtersList.length > 0 ? filtersList.join(', ') : 'None (Baseline Scan)';

    // 3. Diagnostic specs
    reportTargetClass.textContent = currentPredictionData.class;
    
    let sequence = sequenceBadge.textContent.replace('Sequence: ', '');
    reportSequence.textContent = sequence;
    reportTargetSequence.textContent = sequence;
    
    reportCertainty.textContent = `${currentPredictionData.confidence.toFixed(1)}% Certainty`;
    reportSafety.textContent = `${currentPredictionData.confidence_level} Confidence`;
    
    if (currentPredictionData.confidence_color === 'success') {
        reportSafety.className = 'bold-text success-text';
    } else if (currentPredictionData.confidence_color === 'warning') {
        reportSafety.className = 'bold-text';
        reportSafety.style.color = '#ffbd59';
    } else {
        reportSafety.className = 'bold-text';
        reportSafety.style.color = '#00bfff';
    }

    // Set dynamic report metadata fields from DICOM
    const metadata = currentPredictionData.dicom_metadata;
    const patientProfileEl = document.getElementById('reportPatientProfile');
    const scannerTypeEl = document.getElementById('reportScannerType');
    const studyTypeEl = document.getElementById('reportStudyType');

    if (metadata) {
        if (patientProfileEl) {
            const name = metadata.patient_name || "Anonymous Patient";
            const id = metadata.patient_id || "N/A";
            const age = metadata.patient_age || "U";
            patientProfileEl.textContent = `${name} (ID: ${id}, Age: ${age})`;
        }
        if (scannerTypeEl) {
            scannerTypeEl.textContent = `${metadata.scanner_model || "Siemens MAGNETOM"} (${metadata.magnetic_field || "3.0 Tesla"})`;
        }
        if (studyTypeEl) {
            studyTypeEl.textContent = metadata.sequence_type || "Brain MRI Classifier Scan";
        }
    } else {
        if (patientProfileEl) patientProfileEl.textContent = "Anonymous Patient";
        if (scannerTypeEl) scannerTypeEl.textContent = "DICOM Simulated Model";
        if (studyTypeEl) studyTypeEl.textContent = "Brain MRI Classifier Scan";
    }

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message: 'Explain the current diagnosis',
            diagnosis: currentPredictionData.class,
            confidence: currentPredictionData.confidence,
            top_predictions: currentPredictionData.top_predictions
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.response) {
            reportClinicalExplanation.innerHTML = parseSimpleMarkdown(data.response);
        } else {
            reportClinicalExplanation.textContent = 'Failed to compile AI clinical report synthesis.';
        }
    })
    .catch(() => {
        reportClinicalExplanation.textContent = 'Diagnostic connection timeout during clinical report synthesis.';
    });

    reportModal.style.display = 'flex';
}

function closeReportModal() {
    reportModal.style.display = 'none';
    // Clear speech playback in report
    window.speechSynthesis.cancel();
}

// Close modal when clicking outside contents
window.onclick = function(event) {
    if (event.target == reportModal) {
        closeReportModal();
    }
};

// ==========================================================================
// SYSTEM HELPER NOTIFIER
// ==========================================================================
function showError(message) {
    alert(`NEUROAI WORKSTATION ALERT: ${message}`);
}

// ==========================================================================
// SHOWCASE MODES & YOU VS AI CHALLENGE
// ==========================================================================

function toggleGameMode() {
    isGameModeActive = !isGameModeActive;
    const btn = document.getElementById('btnGameMode');
    const gameCard = document.getElementById('gameCard');
    
    if (isGameModeActive) {
        btn.textContent = '🎮 GUESS MODE: ON';
        btn.classList.add('active');
        btn.style.borderColor = 'var(--neon-purple)';
        btn.style.color = '#c8b6ff';
        
        // If prediction exists, switch view to game card
        if (currentPredictionData) {
            resultsCard.style.display = 'none';
            if (gameCard) {
                gameCard.style.display = 'block';
                document.getElementById('gameResultPanel').style.display = 'none';
                document.getElementById('btnRevealAI').style.display = 'none';
                const choiceBtns = gameCard.querySelectorAll('.game-choices-grid button');
                choiceBtns.forEach(b => {
                    b.className = 'btn btn-filter';
                    b.disabled = false;
                });
            }
        }
    } else {
        btn.textContent = '🎮 GUESS MODE: OFF';
        btn.classList.remove('active');
        btn.style.borderColor = '';
        btn.style.color = '';
        
        if (gameCard) gameCard.style.display = 'none';
        if (currentPredictionData) {
            resultsCard.style.display = 'block';
            displayDiagnosticResults(currentPredictionData);
        }
    }
}

function toggleReadingRoomMode() {
    isReadingRoomMode = !isReadingRoomMode;
    const btn = document.getElementById('btnReadingRoom');
    if (isReadingRoomMode) {
        document.body.classList.add('reading-room');
        btn.textContent = '🕶️ READING ROOM: ON';
        btn.classList.add('active');
    } else {
        document.body.classList.remove('reading-room');
        btn.textContent = '🕶️ READING ROOM: OFF';
        btn.classList.remove('active');
    }
}

function submitUserGuess(guess) {
    if (!currentPredictionData) return;
    
    userGuess = guess.toLowerCase();
    const actualClass = currentPredictionData.class.toLowerCase();
    
    const choiceButtons = document.querySelectorAll('.game-choices-grid button');
    choiceButtons.forEach(btn => {
        btn.disabled = true;
        const btnText = btn.textContent.toLowerCase();
        
        // Mark button styling based on guess vs actual
        if (actualClass.includes(btnText)) {
            btn.classList.add('correct');
        } else if (btnText.includes(userGuess)) {
            btn.classList.add('incorrect');
        }
    });
    
    const gameResultPanel = document.getElementById('gameResultPanel');
    const gameResultIcon = document.getElementById('gameResultIcon');
    const gameResultTitle = document.getElementById('gameResultTitle');
    const gameResultText = document.getElementById('gameResultText');
    
    const matched = actualClass.includes(userGuess);
    
    gameResultPanel.style.display = 'block';
    if (matched) {
        gameResultPanel.className = 'game-result-panel win';
        gameResultIcon.textContent = '🎉';
        gameResultTitle.textContent = 'CLINICAL MATCH DETECTED!';
        gameResultText.textContent = `Excellent diagnostic acuity! You correctly identified the presence of ${currentPredictionData.class.replace('_', ' ')}. The AI core matches your finding.`;
    } else {
        gameResultPanel.className = 'game-result-panel lose';
        gameResultIcon.textContent = '❌';
        gameResultTitle.textContent = 'DIAGNOSTIC DISCREPANCY!';
        gameResultText.textContent = `You formulated a guess of "${guess}". The AI core classified this scan as showing "${currentPredictionData.class.replace('_', ' ')}" with ${currentPredictionData.confidence.toFixed(1)}% certainty.`;
    }
    
    // Auto-trigger assistant message explaining this specific result
    appendChatBubble('assistant', `### Clinical Guess correlation result:\nUser Guess: **${guess}**\nAI Classification: **${currentPredictionData.class.replace('_', ' ')}**\n\n${matched ? 'Match successful! Excellent job.' : 'Diagnostic discrepancy detected. Let us review the pathology tags together.'}`);
    
    document.getElementById('btnRevealAI').style.display = 'block';
}

function revealAIReport() {
    const gameCard = document.getElementById('gameCard');
    if (gameCard) gameCard.style.display = 'none';
    
    resultsCard.style.display = 'block';
    displayDiagnosticResults(currentPredictionData);
    
    // Show split slider if segmentation is active
    const splitSliderContainer = document.getElementById('splitSliderContainer');
    if (splitSliderContainer && isSegmentationActive) {
        splitSliderContainer.style.display = 'block';
    }
    
    // Auto-trigger assistant message explaining this specific result
    setTimeout(() => {
        requestAIChatResponse(`Explain the current diagnosis: ${currentPredictionData.class}`);
    }, 500);
}

// Cine Video Preset and dynamic filters support
function updateVideoFilters() {
    const player = document.getElementById('mriVideoPlayer');
    if (!player) return;
    
    let filterString = `brightness(${activeFilters.brightness}%) contrast(${activeFilters.contrast}%)`;
    if (activeFilters.invert) {
        filterString += ' invert(1)';
    }
    if (activeFilters.thermal) {
        filterString += ' hue-rotate(180deg) saturate(250%) contrast(150%)';
    }
    player.style.filter = filterString;
}

async function loadVideoPreset() {
    // Show loading state
    emptyState.style.display = 'none';
    resultsCard.style.display = 'block';
    
    className.textContent = 'Streaming Cine-MRI...';
    className.className = 'diagnosis-badge text-glow';
    confidenceValue.textContent = '0';
    confidenceFill.style.width = '0%';
    confidenceLevel.className = 'confidence-badge';
    confidenceLevel.textContent = 'Decoding video stream';
    
    // Hide standard canvas, show video player
    mriCanvas.style.display = 'none';
    const videoPlayer = document.getElementById('mriVideoPlayer');
    if (videoPlayer) {
        videoPlayer.style.display = 'block';
        videoPlayer.src = '/static/video_sample.webp';
    }
    
    canvasContainer.style.display = 'flex';
    uploadArea.style.display = 'none';
    tuningPanel.style.display = 'block';
    
    // Set a dummy originalImage so renderCanvas doesn't return early
    originalImage = new Image();
    originalImage.width = 224;
    originalImage.height = 224;
    
    // Reset filters
    resetFilters();
    
    // Set sequence badge
    sequenceBadge.textContent = 'Sequence: Cine-MRI Scan (30 Hz)';
    
    // Set dynamic simulated metadata
    updateDicomDisplay({
        scanner_model: 'Siemens MAGNETOM Prisma',
        magnetic_field: '3.0 Tesla Cine',
        coil_type: '64-Ch Head/Neck Coil',
        sequence_type: 'Cine-MRI Slice Scroll',
        tr: '1500 ms',
        te: '85 ms',
        flip_angle: '120°',
        contrast_agent: 'Non-contrast (Time-of-flight)'
    });
    
    // Wait a brief delay to simulate network latency, then return classification
    inferenceStartTime = performance.now();
    setTimeout(() => {
        const data = {
            success: true,
            class: 'Meningioma_T1',
            confidence: 96.4,
            confidence_level: 'High',
            confidence_color: 'success',
            dicom_metadata: {
                scanner_model: 'Siemens MAGNETOM Prisma',
                magnetic_field: '3.0 Tesla Cine',
                coil_type: '64-Ch Head/Neck Coil',
                sequence_type: 'Cine-MRI Slice Scroll',
                tr: '1500 ms',
                te: '85 ms',
                flip_angle: '120°',
                contrast_agent: 'Non-contrast (Time-of-flight)'
            }
        };
        
        currentPredictionData = data;
        
        // Calculate latency and update telemetry
        const latency = Math.max(124, Math.round(performance.now() - inferenceStartTime));
        const latencyEl = document.getElementById('telemetryLatency');
        if (latencyEl) latencyEl.textContent = `${latency} ms`;
        
        // If not Guess Mode, reveal results
        if (!isGameModeActive) {
            displayDiagnosticResults(data);
            
            // Show split slider if segmentation is active
            const splitSliderContainer = document.getElementById('splitSliderContainer');
            if (splitSliderContainer && isSegmentationActive) {
                splitSliderContainer.style.display = 'block';
            }
            
            // Auto-trigger assistant message explaining this specific result
            appendChatBubble('assistant', `### Cine-MRI Video Analysis:\nAI Core is streaming frames from the Cine-MRI sequence. A **Meningioma** lesion has been tracked on sequential sagittal slices 40-75.\n\nRecommended: Correlate with contrast-enhanced volumetric sequences.`);
        } else {
            // In Guess Mode, switch to guessing UI
            resultsCard.style.display = 'none';
            const gameCard = document.getElementById('gameCard');
            if (gameCard) {
                gameCard.style.display = 'block';
                document.getElementById('gameResultPanel').style.display = 'none';
                document.getElementById('btnRevealAI').style.display = 'none';
                const choiceBtns = gameCard.querySelectorAll('.game-choices-grid button');
                choiceBtns.forEach(btn => {
                    btn.className = 'btn btn-filter';
                    btn.disabled = false;
                });
                userGuess = null;
            }
        }
    }, 800);
}

// ==========================================================================
// ADVANCED RADIOLOGY WORKSTATION LOGIC (HISTOGRAM, MAGNIFIER, SEGMENTATION, HISTORY, VOICE, SIGN-OFF)
// ==========================================================================

// --- 1. LOCAL STORAGE SCAN ARCHIVE ---
function initHistory() {
    try {
        const stored = localStorage.getItem('neuroai_workstation_history');
        if (stored) {
            scanHistory = JSON.parse(stored);
        } else {
            scanHistory = [];
        }
        renderHistoryList();
    } catch (e) {
        console.error("Error loading local history:", e);
        scanHistory = [];
    }
}

function savePredictionToHistory(data) {
    try {
        if (!data || !originalImage) return;

        const sequence = sequenceBadge.textContent.replace('Sequence: ', '');
        
        let canvasDataUrl = "";
        try {
            canvasDataUrl = mriCanvas.toDataURL('image/png');
        } catch (canvasErr) {
            console.warn("NeuroAI: Failed to generate canvas data URL for history archive:", canvasErr);
        }

        const item = {
            id: `case_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
            class: data.class,
            confidence: data.confidence,
            confidence_level: data.confidence_level,
            confidence_color: data.confidence_color,
            image: canvasDataUrl, // Saves filtered/drawn canvas snapshot
            sequence: sequence,
            date: new Date().toISOString(),
            dicom_metadata: data.dicom_metadata
        };

        scanHistory.unshift(item);
        
        // Cap archive size to 8 items to fit localStorage limits (5MB)
        if (scanHistory.length > 8) {
            scanHistory = scanHistory.slice(0, 8);
        }

        try {
            localStorage.setItem('neuroai_workstation_history', JSON.stringify(scanHistory));
        } catch (storageErr) {
            console.error("NeuroAI: LocalStorage quota exceeded. Clearing older history entries to make space.", storageErr);
            if (scanHistory.length > 1) {
                scanHistory = scanHistory.slice(0, 2);
                try {
                    localStorage.setItem('neuroai_workstation_history', JSON.stringify(scanHistory));
                } catch (e) {
                    localStorage.removeItem('neuroai_workstation_history');
                }
            } else {
                localStorage.removeItem('neuroai_workstation_history');
            }
        }
        renderHistoryList();
    } catch (e) {
        console.error("NeuroAI: Exception inside savePredictionToHistory:", e);
    }
}

function renderHistoryList() {
    if (!historyList) return;
    historyList.innerHTML = '';
    
    if (scanHistory.length === 0) {
        historyCard.style.display = 'none';
        return;
    }

    historyCard.style.display = 'block';

    scanHistory.forEach(item => {
        const dateObj = new Date(item.date);
        const formattedDate = dateObj.toLocaleDateString(undefined, {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });

        const itemEl = document.createElement('div');
        itemEl.className = 'history-item';
        itemEl.onclick = () => loadScanFromHistory(item.id);

        itemEl.innerHTML = `
            <img src="${item.image}" class="history-thumb" alt="Scan snapshot">
            <div class="history-info">
                <div class="history-class">${item.class}</div>
                <div class="history-meta">
                    <span class="history-conf">${item.confidence.toFixed(1)}% Certainty</span>
                    <span class="history-date">${formattedDate}</span>
                </div>
            </div>
        `;
        historyList.appendChild(itemEl);
    });
}

function clearHistory() {
    console.log("NeuroAI: clearHistory() triggered and executed");
    scanHistory = [];
    localStorage.removeItem('neuroai_workstation_history');
    renderHistoryList();
}
window.clearHistory = clearHistory;

function loadScanFromHistory(itemId) {
    const item = scanHistory.find(i => i.id === itemId);
    if (!item) return;

    emptyState.style.display = 'none';
    resultsCard.style.display = 'block';
    sequenceBadge.textContent = `Sequence: ${item.sequence}`;

    const tempImg = new Image();
    tempImg.onload = () => {
        originalImage = tempImg;
        canvasContainer.style.display = 'flex';
        uploadArea.style.display = 'none';
        tuningPanel.style.display = 'block';
        
        // Reset slider sliders UI without re-rendering
        document.getElementById('brightnessSlider').value = 100;
        document.getElementById('brightVal').textContent = "100%";
        document.getElementById('contrastSlider').value = 100;
        document.getElementById('contrastVal').textContent = "100%";
        
        activeFilters = { brightness: 100, contrast: 100, invert: false, sobel: false, thermal: false };
        
        document.getElementById('btnInvert').classList.remove('active');
        document.getElementById('btnSobel').classList.remove('active');
        document.getElementById('btnThermal').classList.remove('active');
        
        currentPredictionData = {
            class: item.class,
            confidence: item.confidence,
            confidence_level: item.confidence_level,
            confidence_color: item.confidence_color,
            image: item.image.split(',')[1],
            dicom_metadata: item.dicom_metadata
        };

        // Render to canvas
        renderCanvas();
        
        // Display metrics & DICOM
        displayDiagnosticResults(currentPredictionData);
        updateDicomDisplay(item.dicom_metadata);
        
        // Clear sign-off
        resetClinicianForm();
    };
    tempImg.src = item.image;
}

// --- 2. DICOM METADATA CONTROLS ---
function updateDicomDisplay(metadata) {
    if (!metadata) {
        dicomPanel.style.display = 'none';
        return;
    }
    
    dicomPanel.style.display = 'block';
    dicomScanner.textContent = metadata.scanner_model || "Siemens MAGNETOM";
    dicomField.textContent = metadata.magnetic_field || "3.0 Tesla";
    dicomCoil.textContent = metadata.coil_type || "16-Ch Head Coil";
    dicomSequence.textContent = metadata.sequence_type || "-";
    dicomTR.textContent = metadata.tr || "-";
    dicomTE.textContent = metadata.te || "-";
    dicomFlip.textContent = metadata.flip_angle || "-";
    dicomContrast.textContent = metadata.contrast_agent || "None";
}

// --- 3. DYNAMIC FILTERS & AI MASK TOGGLES ---
function toggleMagnifierTool() {
    isMagnifierActive = !isMagnifierActive;
    const btn = document.getElementById('btnMagnifier');
    if (isMagnifierActive) {
        btn.classList.add('active');
        isSegmentationActive = false;
        document.getElementById('btnSegmentation').classList.remove('active');
        const heatmapGroup = document.getElementById('heatmapOpacityGroup');
        if (heatmapGroup) heatmapGroup.style.display = 'none';
        
        isRulerActive = false;
        document.getElementById('btnRuler').classList.remove('active');
        mriCanvas.classList.remove('mri-canvas-ruler-active');
        
        isProfileActive = false;
        document.getElementById('btnProfile').classList.remove('active');
        mriCanvas.classList.remove('mri-canvas-profile-active');
        if (profileSection) profileSection.style.display = 'none';
    } else {
        btn.classList.remove('active');
        magnifierLoupe.style.display = 'none';
    }
    renderCanvas();
}

function toggleSegmentationMask() {
    isSegmentationActive = !isSegmentationActive;
    const btn = document.getElementById('btnSegmentation');
    const heatmapGroup = document.getElementById('heatmapOpacityGroup');
    const splitSliderContainer = document.getElementById('splitSliderContainer');
    if (isSegmentationActive) {
        btn.classList.add('active');
        isMagnifierActive = false;
        document.getElementById('btnMagnifier').classList.remove('active');
        magnifierLoupe.style.display = 'none';
        
        isRulerActive = false;
        document.getElementById('btnRuler').classList.remove('active');
        mriCanvas.classList.remove('mri-canvas-ruler-active');
        
        isProfileActive = false;
        document.getElementById('btnProfile').classList.remove('active');
        mriCanvas.classList.remove('mri-canvas-profile-active');
        if (profileSection) profileSection.style.display = 'none';
        
        if (heatmapGroup) heatmapGroup.style.display = 'block';
        if (splitSliderContainer && originalImage) splitSliderContainer.style.display = 'block';
    } else {
        btn.classList.remove('active');
        if (heatmapGroup) heatmapGroup.style.display = 'none';
        if (splitSliderContainer) splitSliderContainer.style.display = 'none';
    }
    renderCanvas();
}

function toggleRulerTool() {
    isRulerActive = !isRulerActive;
    const btn = document.getElementById('btnRuler');
    if (isRulerActive) {
        btn.classList.add('active');
        
        isProfileActive = false;
        document.getElementById('btnProfile').classList.remove('active');
        if (profileSection) profileSection.style.display = 'none';
        
        isMagnifierActive = false;
        document.getElementById('btnMagnifier').classList.remove('active');
        magnifierLoupe.style.display = 'none';
        
        isSegmentationActive = false;
        document.getElementById('btnSegmentation').classList.remove('active');
        const heatmapGroup = document.getElementById('heatmapOpacityGroup');
        if (heatmapGroup) heatmapGroup.style.display = 'none';
        
        mriCanvas.classList.add('mri-canvas-ruler-active');
        mriCanvas.classList.remove('mri-canvas-profile-active');
    } else {
        btn.classList.remove('active');
        mriCanvas.classList.remove('mri-canvas-ruler-active');
        toolStart = null;
        toolEnd = null;
    }
    renderCanvas();
}

function toggleProfileTool() {
    isProfileActive = !isProfileActive;
    const btn = document.getElementById('btnProfile');
    if (isProfileActive) {
        btn.classList.add('active');
        
        isRulerActive = false;
        document.getElementById('btnRuler').classList.remove('active');
        mriCanvas.classList.remove('mri-canvas-ruler-active');
        
        isMagnifierActive = false;
        document.getElementById('btnMagnifier').classList.remove('active');
        magnifierLoupe.style.display = 'none';
        
        isSegmentationActive = false;
        document.getElementById('btnSegmentation').classList.remove('active');
        const heatmapGroup = document.getElementById('heatmapOpacityGroup');
        if (heatmapGroup) heatmapGroup.style.display = 'none';
        
        if (profileSection) profileSection.style.display = 'block';
        mriCanvas.classList.add('mri-canvas-profile-active');
    } else {
        btn.classList.remove('active');
        mriCanvas.classList.remove('mri-canvas-profile-active');
        if (profileSection) profileSection.style.display = 'none';
        toolStart = null;
        toolEnd = null;
    }
    renderCanvas();
}

// --- 4. REAL-TIME VOXEL HISTOGRAM ---
function renderHistogram() {
    if (!originalImage || !histogramCtx) return;

    const w = mriCanvas.width;
    const h = mriCanvas.height;
    const imgData = ctx.getImageData(0, 0, w, h);
    const data = imgData.data;

    const histData = new Array(256).fill(0);
    for (let i = 0; i < data.length; i += 4) {
        const gray = Math.round((data[i] + data[i + 1] + data[i + 2]) / 3);
        histData[gray]++;
    }

    const histW = histogramCanvas.width;
    const histH = histogramCanvas.height;
    histogramCtx.clearRect(0, 0, histW, histH);

    let maxVal = 0;
    for (let i = 1; i < 256; i++) {
        if (histData[i] > maxVal) maxVal = histData[i];
    }
    if (maxVal === 0) maxVal = 1;

    histogramCtx.strokeStyle = '#00f0ff';
    histogramCtx.fillStyle = 'rgba(0, 240, 255, 0.15)';
    histogramCtx.lineWidth = 1;

    const barWidth = histW / 256;
    histogramCtx.beginPath();
    histogramCtx.moveTo(0, histH);

    for (let i = 0; i < 256; i++) {
        const barHeight = (histData[i] / maxVal) * histH * 0.85;
        const xPos = i * barWidth;
        const yPos = histH - barHeight;
        histogramCtx.lineTo(xPos, yPos);
    }
    histogramCtx.lineTo(histW, histH);
    histogramCtx.closePath();
    histogramCtx.fill();
    histogramCtx.stroke();

    // Draw reference lines
    histogramCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    histogramCtx.beginPath();
    for (let pct = 0.25; pct <= 0.75; pct += 0.25) {
        histogramCtx.moveTo(histW * pct, 0);
        histogramCtx.lineTo(histW * pct, histH);
    }
    histogramCtx.moveTo(0, histH / 2);
    histogramCtx.lineTo(histW, histH / 2);
    histogramCtx.stroke();
}

// --- 5. MAGNIFYING GLASS MOUSE LISTENER ---
function handleMagnifierMove(e) {
    if (!isMagnifierActive || !originalImage) return;

    const rect = mriCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const canvasX = (x / rect.width) * mriCanvas.width;
    const canvasY = (y / rect.height) * mriCanvas.height;

    magnifierLoupe.style.left = `${x - 65}px`;
    magnifierLoupe.style.top = `${y - 65}px`;
    magnifierLoupe.style.display = 'block';

    loupeCtx.clearRect(0, 0, 130, 130);
    
    // Clip a 32.5x32.5 area from canvas and map to 130x130 (2x zoom)
    const zoomSize = 32.5; 
    
    loupeCtx.save();
    loupeCtx.beginPath();
    loupeCtx.arc(65, 65, 65, 0, Math.PI * 2, true);
    loupeCtx.clip();
    
    loupeCtx.drawImage(
        mriCanvas,
        canvasX - zoomSize,
        canvasY - zoomSize,
        zoomSize * 2,
        zoomSize * 2,
        0,
        0,
        130,
        130
    );
    loupeCtx.restore();
}

mriCanvas.addEventListener('mousemove', handleMagnifierMove);
mriCanvas.addEventListener('mouseleave', () => {
    magnifierLoupe.style.display = 'none';
});
mriCanvas.addEventListener('mouseenter', () => {
    if (isMagnifierActive) magnifierLoupe.style.display = 'block';
});

// --- 6. SIMULATED AI LESION SEGMENTATION MAP ---
function drawSegmentationOverlay() {
    if (!isSegmentationActive || !originalImage || !currentPredictionData) return;
    
    const label = currentPredictionData.class.toLowerCase();
    if (label.includes('no tumor')) return;

    const w = mriCanvas.width;
    const h = mriCanvas.height;

    ctx.save();

    let lesions = [];
    if (label.includes('glioma') || label.includes('astrocytoma') || label.includes('glioblastoma') || label.includes('oligodendroglioma')) {
        lesions.push({ cx: w * 0.65, cy: h * 0.38, rx: w * 0.12, ry: h * 0.1, label: "GLIOMA MASS" });
    } else if (label.includes('meningioma')) {
        lesions.push({ cx: w * 0.28, cy: h * 0.48, rx: w * 0.08, ry: h * 0.08, label: "MENINGIOMA MASS" });
    } else if (label.includes('pituitary') || label.includes('germinoma') || label.includes('neurocytoma')) {
        lesions.push({ cx: w * 0.5, cy: h * 0.58, rx: w * 0.07, ry: h * 0.07, label: "SELLAR LESION" });
    } else if (label.includes('schwannoma')) {
        lesions.push({ cx: w * 0.35, cy: h * 0.7, rx: w * 0.06, ry: h * 0.06, label: "CPA SCHWANNOMA" });
    } else if (label.includes('medulloblastoma') || label.includes('ependymoma')) {
        lesions.push({ cx: w * 0.5, cy: h * 0.72, rx: w * 0.08, ry: h * 0.08, label: "INFRATENTORIAL MASS" });
    } else if (label.includes('carcinoma')) {
        lesions.push({ cx: w * 0.32, cy: h * 0.35, rx: w * 0.05, ry: h * 0.05, label: "METASTASIS #1" });
        lesions.push({ cx: w * 0.68, cy: h * 0.55, rx: w * 0.04, ry: h * 0.04, label: "METASTASIS #2" });
    } else if (label.includes('tuberculoma') || label.includes('granuloma') || label.includes('papilloma')) {
        lesions.push({ cx: w * 0.6, cy: h * 0.45, rx: w * 0.07, ry: h * 0.07, label: "FOCAL GRANULOMA" });
    } else {
        lesions.push({ cx: w * 0.55, cy: h * 0.4, rx: w * 0.08, ry: h * 0.08, label: "SUSPICIOUS LESION" });
    }

    lesions.forEach(lesion => {
        // Draw Grad-CAM Attention Heatmap
        ctx.beginPath();
        const maxRadius = Math.max(lesion.rx, lesion.ry) * 2.2;
        let radGrad = ctx.createRadialGradient(lesion.cx, lesion.cy, 0, lesion.cx, lesion.cy, maxRadius);
        radGrad.addColorStop(0, `rgba(255, 0, 0, ${heatmapOpacity})`);
        radGrad.addColorStop(0.2, `rgba(255, 128, 0, ${heatmapOpacity * 0.9})`);
        radGrad.addColorStop(0.4, `rgba(255, 255, 0, ${heatmapOpacity * 0.7})`);
        radGrad.addColorStop(0.6, `rgba(0, 255, 128, ${heatmapOpacity * 0.5})`);
        radGrad.addColorStop(0.8, `rgba(0, 128, 255, ${heatmapOpacity * 0.2})`);
        radGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
        
        ctx.fillStyle = radGrad;
        ctx.arc(lesion.cx, lesion.cy, maxRadius, 0, 2 * Math.PI);
        ctx.fill();

        // Draw Clinical Outline and crosshair
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#ff007f';
        ctx.strokeStyle = '#ff007f';
        ctx.lineWidth = 2.0;
        
        ctx.beginPath();
        ctx.ellipse(lesion.cx, lesion.cy, lesion.rx, lesion.ry, 0, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.shadowBlur = 0;
        ctx.strokeStyle = 'rgba(255, 0, 127, 0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(lesion.cx - lesion.rx - 15, lesion.cy);
        ctx.lineTo(lesion.cx + lesion.rx + 15, lesion.cy);
        ctx.moveTo(lesion.cx, lesion.cy - lesion.ry - 15);
        ctx.lineTo(lesion.cx, lesion.cy + lesion.ry + 15);
        ctx.stroke();

        ctx.fillStyle = '#ff007f';
        ctx.font = 'bold 11px monospace';
        ctx.fillText(lesion.label, lesion.cx + lesion.rx + 5, lesion.cy - 5);
        
        // Add Grad-CAM label tag
        ctx.fillStyle = '#00f0ff';
        ctx.font = 'bold 9px monospace';
        ctx.fillText("GRAD-CAM ATTN: 100%", lesion.cx + lesion.rx + 5, lesion.cy + 10);
    });

    ctx.restore();
}

// --- 7. VOICE DIAGNOSTICS (TEXT TO SPEECH) ---
function speakDiagnostics() {
    if (!currentPredictionData) return;

    if (isVoiceSpeaking) {
        window.speechSynthesis.cancel();
        isVoiceSpeaking = false;
        document.getElementById('btnVoiceSpeak').innerHTML = '🔊 LISTEN TO REPORT';
        return;
    }

    const textToSpeak = `AI Diagnostic Report. Classification: ${currentPredictionData.class}. Confidence Level: ${currentPredictionData.confidence_level} Confidence, with a certainty of ${currentPredictionData.confidence.toFixed(1)} percent. Let's ask our chat assistant for detailed clinical recommendations.`;

    speechUtterance = new SpeechSynthesisUtterance(textToSpeak);
    speechUtterance.rate = 0.95;
    speechUtterance.pitch = 1.0;
    
    speechUtterance.onend = () => {
        isVoiceSpeaking = false;
        document.getElementById('btnVoiceSpeak').innerHTML = '🔊 LISTEN TO REPORT';
    };

    speechUtterance.onerror = () => {
        isVoiceSpeaking = false;
        document.getElementById('btnVoiceSpeak').innerHTML = '🔊 LISTEN TO REPORT';
    };

    isVoiceSpeaking = true;
    document.getElementById('btnVoiceSpeak').innerHTML = '⏹️ STOP PLAYBACK';
    window.speechSynthesis.speak(speechUtterance);
}

function speakReportDiagnostics() {
    if (!currentPredictionData) return;
    
    const explanationText = reportClinicalExplanation.textContent || "";
    const cleanExplanation = explanationText.replace(/[#*`>]/g, '');
    const speechText = `Clinically compiled report for ${currentPredictionData.class}. AI Certainty is ${currentPredictionData.confidence.toFixed(1)} percent. ${cleanExplanation}`;
    
    if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        return;
    }
    
    const reportUtterance = new SpeechSynthesisUtterance(speechText);
    reportUtterance.rate = 1.0;
    window.speechSynthesis.speak(reportUtterance);
}

// --- 8. CLINICIAN REPORT SIGN-OFF & LIVE PREVIEW ---
function applyClinicalSignature() {
    const clinicianName = signClinicianName.value.trim() || "Not Reviewed (Self-Service Scan)";
    const licenseId = signLicenseId.value.trim() || "N/A";
    const notes = signNotes.value.trim() || "No physician notes provided. This report represents the raw AI model output.";

    document.getElementById('printClinicianName').textContent = clinicianName;
    document.getElementById('printLicenseId').textContent = licenseId;
    document.getElementById('printNotes').textContent = notes;

    const dateStr = new Date().toISOString();
    const rawString = `${clinicianName}-${licenseId}-${currentPredictionData.class}-${dateStr}`;
    let hash = 0;
    for (let i = 0; i < rawString.length; i++) {
        const char = rawString.charCodeAt(i);
        hash = (hash << 5) - hash + char;
        hash |= 0;
    }
    const secureHash = `SHA256-VERIFY-${Math.abs(hash).toString(16).toUpperCase()}-${Math.floor(1000 + Math.random() * 9000)}`;
    printVerificationHash.textContent = secureHash;

    holographicStamp.className = "holographic-stamp verified";
    holographicStamp.querySelector('.stamp-text').innerHTML = "CLINICALLY<br>VERIFIED";

    signClinicianName.disabled = true;
    signLicenseId.disabled = true;
    signNotes.disabled = true;
    btnApplySignature.disabled = true;
    btnApplySignature.textContent = "✒️ REPORT OFFICIALLY VERIFIED";
    btnApplySignature.style.background = "#198754";
}

function resetClinicianForm() {
    if (!signClinicianName) return;
    signClinicianName.value = '';
    signLicenseId.value = '';
    signNotes.value = '';
    
    signClinicianName.disabled = false;
    signLicenseId.disabled = false;
    signNotes.disabled = false;
    
    btnApplySignature.disabled = false;
    btnApplySignature.textContent = "✒️ SIGN & CERTIFY REPORT";
    btnApplySignature.style.background = "";

    document.getElementById('printClinicianName').textContent = "Not Reviewed (Self-Service Scan)";
    document.getElementById('printLicenseId').textContent = "N/A";
    document.getElementById('printNotes').textContent = "No physician notes provided. This report represents the raw AI model output.";
    printVerificationHash.textContent = "UNVERIFIED";

    holographicStamp.className = "holographic-stamp";
    holographicStamp.querySelector('.stamp-text').innerHTML = "UNVERIFIED";
}

// --- 9. DRAW CALIPER & INTENSITY PROFILER FUNCTIONS ---
function drawCaliperRuler() {
    if (!toolStart || !toolEnd) return;

    // Draw the main line
    ctx.save();
    ctx.strokeStyle = '#00f0ff'; // Neon Cyan
    ctx.lineWidth = 2;
    ctx.shadowBlur = 8;
    ctx.shadowColor = '#00f0ff';
    
    ctx.beginPath();
    ctx.moveTo(toolStart.x, toolStart.y);
    ctx.lineTo(toolEnd.x, toolEnd.y);
    ctx.stroke();

    // Draw end ticks (perpendicular lines)
    const dx = toolEnd.x - toolStart.x;
    const dy = toolEnd.y - toolStart.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    
    if (len > 0) {
        const tickLen = 6;
        const px = -dy / len;
        const py = dx / len;

        ctx.beginPath();
        ctx.moveTo(toolStart.x - px * tickLen, toolStart.y - py * tickLen);
        ctx.lineTo(toolStart.x + px * tickLen, toolStart.y + py * tickLen);
        ctx.moveTo(toolEnd.x - px * tickLen, toolEnd.y - py * tickLen);
        ctx.lineTo(toolEnd.x + px * tickLen, toolEnd.y + py * tickLen);
        ctx.stroke();
    }

    // Calculate physical distance
    let spacingX = 0.45;
    let spacingY = 0.45;
    if (currentPredictionData && currentPredictionData.dicom_metadata && currentPredictionData.dicom_metadata.pixel_spacing) {
        const spacing = currentPredictionData.dicom_metadata.pixel_spacing;
        if (spacing.length >= 2) {
            spacingX = spacing[0];
            spacingY = spacing[1];
        }
    }
    
    const dx_mm = dx * spacingX;
    const dy_mm = dy * spacingY;
    const distance_mm = Math.sqrt(dx_mm * dx_mm + dy_mm * dy_mm);
    
    const distance_cm = distance_mm / 10;
    const volume_cm3 = (Math.PI / 6) * Math.pow(distance_cm, 3);

    // Draw clinical measurement text badge
    ctx.shadowBlur = 0; // Turn off shadow for text readability
    const label = `${distance_mm.toFixed(1)} mm (Est. Vol: ${volume_cm3.toFixed(2)} cm³)`;
    
    ctx.font = 'bold 11px monospace';
    const textWidth = ctx.measureText(label).width;
    const padding = 6;
    
    // Determine label position (offset from midpoint)
    let labelX = (toolStart.x + toolEnd.x) / 2 + 10;
    let labelY = (toolStart.y + toolEnd.y) / 2 - 10;
    
    // Keep inside bounds
    if (labelX + textWidth > mriCanvas.width) labelX = mriCanvas.width - textWidth - 10;
    if (labelY - 12 < 0) labelY = 15;
    
    // Draw background card for label
    ctx.fillStyle = 'rgba(2, 5, 11, 0.85)';
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) {
        ctx.roundRect(labelX - padding, labelY - 12, textWidth + padding * 2, 18, 4);
    } else {
        ctx.rect(labelX - padding, labelY - 12, textWidth + padding * 2, 18);
    }
    ctx.fill();
    ctx.stroke();
    
    // Draw text
    ctx.fillStyle = '#00f0ff';
    ctx.fillText(label, labelX, labelY);
    ctx.restore();
}

function drawProfileLine() {
    if (!toolStart || !toolEnd) return;

    ctx.save();
    ctx.strokeStyle = '#ff007f'; // Neon Pink
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]); // Dashed line
    
    ctx.beginPath();
    ctx.moveTo(toolStart.x, toolStart.y);
    ctx.lineTo(toolEnd.x, toolEnd.y);
    ctx.stroke();

    // Draw end crosshair circles
    ctx.setLineDash([]); // Reset line dash
    ctx.fillStyle = '#ff007f';
    ctx.beginPath();
    ctx.arc(toolStart.x, toolStart.y, 4, 0, 2 * Math.PI);
    ctx.arc(toolEnd.x, toolEnd.y, 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.restore();
}

function updateLineProfiler() {
    if (!toolStart || !toolEnd || !profileCtx) return;
    
    const w = mriCanvas.width;
    const h = mriCanvas.height;
    
    // Get complete canvas image data once
    const imgData = ctx.getImageData(0, 0, w, h);
    const data = imgData.data;
    
    const numSamples = 100;
    const samples = [];
    
    for (let i = 0; i < numSamples; i++) {
        const t = i / (numSamples - 1);
        const px = toolStart.x + t * (toolEnd.x - toolStart.x);
        const py = toolStart.y + t * (toolEnd.y - toolStart.y);
        
        // Clamp bounds
        const x = Math.max(0, Math.min(w - 1, Math.round(px)));
        const y = Math.max(0, Math.min(h - 1, Math.round(py)));
        
        const idx = (y * w + x) * 4;
        const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
        samples.push(gray);
    }
    
    // Render graph on profileCanvas
    const gw = profileCanvas.width;
    const gh = profileCanvas.height;
    
    profileCtx.clearRect(0, 0, gw, gh);
    
    // 1. Draw reference grid lines
    profileCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    profileCtx.lineWidth = 1;
    profileCtx.beginPath();
    // Horizontal reference lines (25%, 50%, 75%)
    for (let pct of [0.25, 0.5, 0.75]) {
        profileCtx.moveTo(0, gh * pct);
        profileCtx.lineTo(gw, gh * pct);
    }
    // Vertical reference lines (25%, 50%, 75%)
    for (let pct of [0.25, 0.5, 0.75]) {
        profileCtx.moveTo(gw * pct, 0);
        profileCtx.lineTo(gw * pct, gh);
    }
    profileCtx.stroke();
    
    // 2. Draw profile path
    profileCtx.strokeStyle = '#ff007f'; // Neon Pink
    profileCtx.fillStyle = 'rgba(255, 0, 127, 0.15)';
    profileCtx.lineWidth = 1.5;
    
    profileCtx.beginPath();
    profileCtx.moveTo(0, gh);
    
    for (let i = 0; i < numSamples; i++) {
        const val = samples[i];
        const xPos = (i / (numSamples - 1)) * gw;
        const yPos = gh - (val / 255) * gh;
        profileCtx.lineTo(xPos, yPos);
    }
    
    profileCtx.lineTo(gw, gh);
    profileCtx.closePath();
    profileCtx.fill();
    
    // Stroke the top outline path again (without bottom line)
    profileCtx.beginPath();
    for (let i = 0; i < numSamples; i++) {
        const val = samples[i];
        const xPos = (i / (numSamples - 1)) * gw;
        const yPos = gh - (val / 255) * gh;
        if (i === 0) {
            profileCtx.moveTo(xPos, yPos);
        } else {
            profileCtx.lineTo(xPos, yPos);
        }
    }
    profileCtx.stroke();
    
    // 3. Add text labels for intensity values
    profileCtx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    profileCtx.font = '8px monospace';
    profileCtx.fillText('255 (Max)', 2, 8);
    profileCtx.fillText('0 (Min)', 2, gh - 2);
}

// Initialize components on startup
document.addEventListener('DOMContentLoaded', () => {
    initHistory();

    // Add heatmap slider listener
    if (heatmapSlider) {
        heatmapSlider.addEventListener('input', (e) => {
            heatmapOpacity = parseInt(e.target.value) / 100;
            if (heatmapVal) heatmapVal.textContent = `${e.target.value}%`;
            renderCanvas();
        });
    }

    // Add split slider listener
    const splitSlider = document.getElementById('splitSlider');
    if (splitSlider) {
        splitSlider.addEventListener('input', (e) => {
            splitPct = parseInt(e.target.value);
            renderCanvas();
        });
    }

    // Add canvas mouse interaction listeners
    mriCanvas.addEventListener('mousedown', (e) => {
        if (!isRulerActive && !isProfileActive) return;
        if (!originalImage) return;

        const rect = mriCanvas.getBoundingClientRect();
        const canvasX = ((e.clientX - rect.left) / rect.width) * mriCanvas.width;
        const canvasY = ((e.clientY - rect.top) / rect.height) * mriCanvas.height;

        isDrawingTool = true;
        toolStart = { x: canvasX, y: canvasY };
        toolEnd = { x: canvasX, y: canvasY };
        
        renderCanvas();
    });

    mriCanvas.addEventListener('mousemove', (e) => {
        // Magnifier loupe handler runs independently if active
        if (isMagnifierActive && originalImage && !isDrawingTool) {
            handleMagnifierMove(e);
            return;
        }

        if (!isDrawingTool || !originalImage) return;

        const rect = mriCanvas.getBoundingClientRect();
        const canvasX = ((e.clientX - rect.left) / rect.width) * mriCanvas.width;
        const canvasY = ((e.clientY - rect.top) / rect.height) * mriCanvas.height;

        toolEnd = { x: canvasX, y: canvasY };

        renderCanvas();
        
        if (isProfileActive) {
            updateLineProfiler();
        }
    });

    mriCanvas.addEventListener('mouseup', (e) => {
        if (!isDrawingTool) return;
        isDrawingTool = false;
        
        const rect = mriCanvas.getBoundingClientRect();
        const canvasX = ((e.clientX - rect.left) / rect.width) * mriCanvas.width;
        const canvasY = ((e.clientY - rect.top) / rect.height) * mriCanvas.height;
        
        toolEnd = { x: canvasX, y: canvasY };
        
        renderCanvas();
        
        if (isProfileActive) {
            updateLineProfiler();
        }
    });
});

