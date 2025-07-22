// dashboard.js - Consolidated file with all functionality

// Constants for file paths
const OUTPUT_FOLDER = 'outputs';
const TEXT_TO_IMAGE_OUTPUT_FOLDER = 'text_to_image_outputs';

// Firebase configuration
let firebaseConfig = {
    // These will be filled with your actual Firebase config from the backend
    apiKey: "",
    authDomain: "",
    projectId: "",
    storageBucket: "",
    messagingSenderId: "",
    appId: ""
};

// Global variables
let firebase;
let db;
let storage;
let lastUploadedImage = null;
let lastTextPrompt = null;
let generatedImages = [];
let textGeneratedImages = [];
let currentUser = null;
let currentMode = 'image'; // 'image' or 'text'

// Global variables for history
let historyData = [];
let historyPage = 1;
let historyLimit = 12;
let historyFilters = {
    dateRange: 'all',
    type: 'all',
    status: 'all',
    sortOrder: 'newest',
    startDate: null,
    endDate: null
};

// DOM elements
// Generator and creation elements
let designForm;
let textToImageForm;
let shoeImageInput;
let textPrompt;
let previewBtn;
let previewContainer;
let imagePreview;
let generateBtn;
let generateTextBtn;
let regenerateBtn;
let regenerateTextBtn;
let loadingIndicator;
let textLoadingIndicator;
let resultsContainer;
let textResultsContainer;
let generatedImagesContainer;
let textGeneratedImagesContainer;
let downloadAllBtn;
let downloadAllTextBtn;
let noFavoritesMessage;
let favoritesSlider;
let notificationToast;
let toastTitle;
let toastMessage;
let imageGeneratorBtn;
let textGeneratorBtn;
let generator;
let textGenerator;
let textProgressBar;
let textProgressText;
let generatedPromptDisplay;

// History tab elements
let historyDateFilter;
let historyTypeFilter;
let historyStatusFilter;
let historySortOrder;
let historyStartDate;
let historyEndDate;
let customDateContainer;
let applyCustomDateBtn;
let historyLoadingIndicator;
let noHistoryMessage;
let historyGrid;
let historyItems;
let loadMoreHistoryBtn;

// Production elements
let noProductionMessage;
let productionSlider;
let productionGrid;
let productionItems;
let productionActions;
let downloadAllProductionBtn;
let sendAllToManufacturingBtn;

// Modal elements
let designDetailModal;
let modalDesignImage;
let modalDesignId;
let modalDesignCreated;
let modalDesignType;
let modalDesignStatus;
let modalDesignPromptContainer;
let modalDesignPrompt;
let modalDesignQualityContainer;
let modalDesignQuality;
let modalDesignProductionBtn;
let modalProductionBtnText;
let modalDesignDownloadBtn;

// Debug logging function
function debugLog(message, data) {
    const debug = false; // Set to false in production
    if (debug) {
        console.log(`[DEBUG] ${message}`, data || '');
    }
}

// Initialize dashboard when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard script loaded');
    
    // Initialize elements
    initElements();
    
    // Fetch Firebase config and initialize
    fetchFirebaseConfig();
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize user info
    initUserInfo();
    
    // Setup tab activation listeners
    setupTabListeners();
});

// Initialize all DOM element references
function initElements() {
    // Initialize creation elements
    designForm = document.getElementById('designForm');
    textToImageForm = document.getElementById('textToImageForm');
    shoeImageInput = document.getElementById('shoeImage');
    textPrompt = document.getElementById('textPrompt');
    previewBtn = document.getElementById('previewBtn');
    previewContainer = document.getElementById('previewContainer');
    imagePreview = document.getElementById('imagePreview');
    generateBtn = document.getElementById('generateBtn');
    generateTextBtn = document.getElementById('generateTextBtn');
    regenerateBtn = document.getElementById('regenerateBtn');
    regenerateTextBtn = document.getElementById('regenerateTextBtn');
    loadingIndicator = document.getElementById('loadingIndicator');
    textLoadingIndicator = document.getElementById('textLoadingIndicator');
    resultsContainer = document.getElementById('resultsContainer');
    textResultsContainer = document.getElementById('textResultsContainer');
    generatedImagesContainer = document.getElementById('generatedImages');
    textGeneratedImagesContainer = document.getElementById('textGeneratedImages');
    downloadAllBtn = document.getElementById('downloadAllBtn');
    downloadAllTextBtn = document.getElementById('downloadAllTextBtn');
    noFavoritesMessage = document.getElementById('noFavoritesMessage');
    favoritesSlider = document.getElementById('favoritesSlider');
    notificationToast = document.getElementById('notificationToast');
    toastTitle = document.getElementById('toastTitle');
    toastMessage = document.getElementById('toastMessage');
    imageGeneratorBtn = document.getElementById('imageGeneratorBtn');
    textGeneratorBtn = document.getElementById('textGeneratorBtn');
    generator = document.getElementById('generator');
    textGenerator = document.getElementById('text-generator');
    textProgressBar = document.getElementById('textProgressBar');
    textProgressText = document.getElementById('textProgressText');
    generatedPromptDisplay = document.getElementById('generatedPromptDisplay');
    
    // Initialize history filter elements
    historyDateFilter = document.getElementById('historyDateFilter');
    historyTypeFilter = document.getElementById('historyTypeFilter');
    historyStatusFilter = document.getElementById('historyStatusFilter');
    historySortOrder = document.getElementById('historySortOrder');
    historyStartDate = document.getElementById('historyStartDate');
    historyEndDate = document.getElementById('historyEndDate');
    customDateContainer = document.getElementById('customDateContainer');
    applyCustomDateBtn = document.getElementById('applyCustomDateBtn');
    historyLoadingIndicator = document.getElementById('historyLoadingIndicator');
    noHistoryMessage = document.getElementById('noHistoryMessage');
    historyGrid = document.getElementById('historyGrid');
    historyItems = document.getElementById('historyItems');
    loadMoreHistoryBtn = document.getElementById('loadMoreHistoryBtn');
    
    // Production elements
    noProductionMessage = document.getElementById('noProductionMessage');
    productionSlider = document.getElementById('productionSlider');
    productionGrid = document.getElementById('productionGrid');
    productionItems = document.getElementById('productionItems');
    productionActions = document.getElementById('productionActions');
    downloadAllProductionBtn = document.getElementById('downloadAllProductionBtn');
    sendAllToManufacturingBtn = document.getElementById('sendAllToManufacturingBtn');
    
    // Modal elements
    designDetailModal = document.getElementById('designDetailModal');
    modalDesignImage = document.getElementById('modalDesignImage');
    modalDesignId = document.getElementById('modalDesignId');
    modalDesignCreated = document.getElementById('modalDesignCreated');
    modalDesignType = document.getElementById('modalDesignType');
    modalDesignStatus = document.getElementById('modalDesignStatus');
    modalDesignPromptContainer = document.getElementById('modalDesignPromptContainer');
    modalDesignPrompt = document.getElementById('modalDesignPrompt');
    modalDesignQualityContainer = document.getElementById('modalDesignQualityContainer');
    modalDesignQuality = document.getElementById('modalDesignQuality');
    modalDesignProductionBtn = document.getElementById('modalDesignProductionBtn');
    modalProductionBtnText = document.getElementById('modalProductionBtnText');
    modalDesignDownloadBtn = document.getElementById('modalDesignDownloadBtn');
    
    console.log('Elements initialized');
}



// Set up tab activation listeners
function setupTabListeners() {
    const tabButtons = document.querySelectorAll('[data-bs-toggle="tab"]');
    
    tabButtons.forEach(button => {
        button.addEventListener('shown.bs.tab', function(event) {
            const targetId = event.target.getAttribute('data-bs-target');
            
            // If history tab is activated, load history data
            if (targetId === '#history-panel') {
                debugLog('History tab activated, loading data');
                loadHistoryData();
            }
            
            // If production tab is activated, load production designs
            if (targetId === '#production-panel') {
                debugLog('Production tab activated, loading data');
                loadProductionDesigns();
            }
        });
    });
}

// Fetch Firebase configuration from the server
function fetchFirebaseConfig() {
    console.log('Fetching Firebase config...');
    
    fetch('/firebase-config')
        .then(response => response.json())
        .then(config => {
            // Save the config
            firebaseConfig = config;
            console.log('Firebase config received');
            
            // Initialize Firebase
            try {
                if (firebase && firebase.apps && firebase.apps.length) {
                    console.log('Firebase already initialized');
                } else {
                    console.log('Initializing Firebase');
                    firebase = firebase.initializeApp(firebaseConfig);
                }
                
                if (firebase.firestore) {
                    db = firebase.firestore();
                    console.log('Firestore initialized');
                }
                
                // Initialize authentication state change listener
                initAuthStateListener();
                
                // Load favorites after Firebase is initialized
                setTimeout(loadFavorites, 500);
            } catch (error) {
                console.error('Error initializing Firebase:', error);
            }
        })
        .catch(error => {
            console.error('Error loading Firebase config:', error);
            showNotification('Error', 'Failed to connect to the database. Please try again later.');
        });
}

// Initialize Firebase Auth state listener
function initAuthStateListener() {
    if (!firebase || !firebase.auth) {
        console.log('Firebase Auth not available');
        return;
    }
    
    firebase.auth().onAuthStateChanged(user => {
        if (user) {
            currentUser = user;
            console.log('User is signed in:', user.uid);
            // Load favorites when user signs in
            setTimeout(loadFavorites, 500);
            // Also load production designs and history
            setTimeout(loadProductionDesigns, 800);
            setTimeout(loadHistoryData, 1000);
        } else {
            currentUser = null;
            console.log('User is signed out');
            // Clear favorites when user signs out
            if (favoritesSlider) {
                favoritesSlider.innerHTML = '';
            }
            if (noFavoritesMessage) {
                noFavoritesMessage.classList.remove('d-none');
            }
        }
    });
}

// Set up all event listeners
function setupEventListeners() {
    // Generator mode toggle
    if (imageGeneratorBtn) {
        imageGeneratorBtn.addEventListener('click', () => switchGeneratorMode('image'));
    }
    
    if (textGeneratorBtn) {
        textGeneratorBtn.addEventListener('click', () => switchGeneratorMode('text'));
    }
    
    // Preview uploaded image
    if (previewBtn) {
        previewBtn.addEventListener('click', previewImage);
    }
    
    // Preview on file selection
    if (shoeImageInput) {
        shoeImageInput.addEventListener('change', previewImage);
    }
    
    // Form submissions
    if (designForm) {
        designForm.addEventListener('submit', handleFormSubmit);
    }
    
    if (textToImageForm) {
        textToImageForm.addEventListener('submit', handleTextToImageSubmit);
    }
    
    // Regenerate buttons
    if (regenerateBtn) {
        regenerateBtn.addEventListener('click', handleRegenerate);
    }
    
    if (regenerateTextBtn) {
        regenerateTextBtn.addEventListener('click', handleTextRegenerate);
    }
    
    // Download all buttons
    if (downloadAllBtn) {
        downloadAllBtn.addEventListener('click', downloadAllImages);
    }
    
    if (downloadAllTextBtn) {
        downloadAllTextBtn.addEventListener('click', downloadAllTextImages);
    }
    
    // History filter change events
    if (historyDateFilter) {
        historyDateFilter.addEventListener('change', function() {
            historyFilters.dateRange = this.value;
            if (this.value === 'custom') {
                customDateContainer.classList.remove('d-none');
            } else {
                customDateContainer.classList.add('d-none');
                resetHistoryAndLoad();
            }
        });
    }
    
    if (historyTypeFilter) {
        historyTypeFilter.addEventListener('change', function() {
            historyFilters.type = this.value;
            resetHistoryAndLoad();
        });
    }
    
    if (historyStatusFilter) {
        historyStatusFilter.addEventListener('change', function() {
            historyFilters.status = this.value;
            resetHistoryAndLoad();
        });
    }
    
    if (historySortOrder) {
        historySortOrder.addEventListener('change', function() {
            historyFilters.sortOrder = this.value;
            resetHistoryAndLoad();
        });
    }
    
    // Custom date range
    if (applyCustomDateBtn) {
        applyCustomDateBtn.addEventListener('click', function() {
            const startDate = historyStartDate.value;
            const endDate = historyEndDate.value;
            
            // Validate date range
            const validation = validateCustomDateRange(startDate, endDate);
            
            if (!validation.valid) {
                // Show error message
                showNotification('Invalid Date Range', validation.message);
                return;
            }
            
            // Proceed with valid date range
            historyFilters.startDate = startDate;
            historyFilters.endDate = endDate;
            resetHistoryAndLoad();
        });
    }
    
    // Load more history
    if (loadMoreHistoryBtn) {
        loadMoreHistoryBtn.addEventListener('click', function() {
            historyPage++;
            loadHistoryData(false);
        });
    }
    
    // Production buttons
    if (downloadAllProductionBtn) {
        downloadAllProductionBtn.addEventListener('click', downloadAllProductionDesigns);
    }
    
    if (sendAllToManufacturingBtn) {
        sendAllToManufacturingBtn.addEventListener('click', sendAllToManufacturing);
    }
    
    // Modal buttons
    if (modalDesignProductionBtn) {
        modalDesignProductionBtn.addEventListener('click', function() {
            const designId = modalDesignId.textContent;
            const isInProduction = modalDesignStatus.textContent === 'Sent to Production';
            toggleProduction(designId, !isInProduction);
            
            // Close the modal
            const modalInstance = bootstrap.Modal.getInstance(designDetailModal);
            if (modalInstance) {
                modalInstance.hide();
            }
        });
    }
    
    if (modalDesignDownloadBtn) {
        modalDesignDownloadBtn.addEventListener('click', function() {
            const designId = modalDesignId.textContent;
            const designData = findDesignById(designId);
            if (designData) {
                downloadImage(designData.cloudinary_url || designData.url, `design-${designId}`);
            }
        });
    }
    
    console.log('Event listeners set up');
}

// Initialize user information
function initUserInfo() {
    // Try to get user information from the backend
    fetch('/user-profile')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success && data.profile) {
                // Display username in header
                const usernameElement = document.getElementById('username');
                if (usernameElement) {
                    if (data.profile.username) {
                        usernameElement.textContent = data.profile.username;
                    } else if (data.profile.email) {
                        // Use email if username not available
                        const username = data.profile.email.split('@')[0];
                        usernameElement.textContent = username;
                    } else {
                        usernameElement.textContent = 'User';
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error fetching user profile:', error);
        });
}

// Switch between generator modes
function switchGeneratorMode(mode) {
    currentMode = mode;
    
    if (mode === 'image') {
        // Show image generator, hide text generator
        if (generator) generator.classList.remove('d-none');
        if (textGenerator) textGenerator.classList.add('d-none');
        
        // Update button states
        if (imageGeneratorBtn) {
            imageGeneratorBtn.classList.remove('btn-outline-primary');
            imageGeneratorBtn.classList.add('btn-primary');
        }
        if (textGeneratorBtn) {
            textGeneratorBtn.classList.remove('btn-primary');
            textGeneratorBtn.classList.add('btn-outline-primary');
        }
        
        // Hide text results, show image results if they exist
        if (textResultsContainer) textResultsContainer.classList.add('d-none');
        if (resultsContainer && generatedImages.length > 0) {
            resultsContainer.classList.remove('d-none');
        }
    } else {
        // Show text generator, hide image generator
        if (textGenerator) textGenerator.classList.remove('d-none');
        if (generator) generator.classList.add('d-none');
        
        // Update button states
        if (textGeneratorBtn) {
            textGeneratorBtn.classList.remove('btn-outline-primary');
            textGeneratorBtn.classList.add('btn-primary');
        }
        if (imageGeneratorBtn) {
            imageGeneratorBtn.classList.remove('btn-primary');
            imageGeneratorBtn.classList.add('btn-outline-primary');
        }
        
        // Hide image results, show text results if they exist
        if (resultsContainer) resultsContainer.classList.add('d-none');
        if (textResultsContainer && textGeneratedImages.length > 0) {
            textResultsContainer.classList.remove('d-none');
        }
    }
}

// Preview the uploaded image
function previewImage() {
    const file = shoeImageInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('d-none');
        }
        reader.readAsDataURL(file);
    }
}

// Handle form submission (Generate button for image variations)
function handleFormSubmit(e) {
    e.preventDefault();
    
    // Check if an image was uploaded
    if (!shoeImageInput.files[0]) {
        showNotification('Error', 'Please upload an image first.');
        return;
    }
    
    // Show loading indicator
    loadingIndicator.classList.remove('d-none');
    resultsContainer.classList.add('d-none');
    
    // Create form data
    const formData = new FormData(designForm);
    
    // Save the uploaded image for potential regeneration
    lastUploadedImage = shoeImageInput.files[0];
    
    // Enable regenerate button
    regenerateBtn.disabled = false;
    
    // Call the API to generate images
    fetch('/generate', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to generate images');
        }
        return response.json();
    })
    .then(data => {
        // Hide loading indicator
        loadingIndicator.classList.add('d-none');
        
        // Store generated images data
        generatedImages = data.images;
        
        // Display the generated images
        displayGeneratedImages(data.images);
        
        // Show results container
        resultsContainer.classList.remove('d-none');
        
        // Show success notification
        showNotification('Success', `Successfully generated ${data.images.length} variations.`);
    })
    .catch(error => {
        // Hide loading indicator
        loadingIndicator.classList.add('d-none');
        
        // Show error notification
        showNotification('Error', 'Failed to generate images. Please try again.');
        console.error('Error generating images:', error);
    });
}

// Handle text-to-image form submission
function handleTextToImageSubmit(e) {
    e.preventDefault();
    
    const prompt = textPrompt.value.trim();
    if (!prompt) {
        showNotification('Error', 'Please enter a text prompt.');
        return;
    }
    
    // Show loading indicator with progress
    textLoadingIndicator.classList.remove('d-none');
    textResultsContainer.classList.add('d-none');
    
    // Reset progress bar
    updateProgress(0, 'Initializing models...');
    
    // Get form data
    const formData = {
        text_prompt: prompt,
        steps: parseInt(document.getElementById('generationSteps').value || 300),
        quality_threshold: parseFloat(document.getElementById('qualityThreshold').value || 0.6)
    };
    
    // Save the prompt for potential regeneration
    lastTextPrompt = prompt;
    
    // Enable regenerate button
    regenerateTextBtn.disabled = false;
    
    // Start progress simulation
    simulateProgress();
    
    // Call the API to generate images from text
    fetch('/generate-text-to-image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to generate images from text');
        }
        return response.json();
    })
    .then(data => {
        // Clear any progress interval
        if (window.textProgressInterval) {
            clearInterval(window.textProgressInterval);
        }
        
        // Set progress to 100%
        updateProgress(100, 'Generation complete!');
        
        // Hide loading indicator after a short delay
        setTimeout(() => {
            textLoadingIndicator.classList.add('d-none');
            
            // Store generated images data
            textGeneratedImages = data.images;
            
            // Display the generated images
            displayTextGeneratedImages(data.images, data.prompt);
            
            // Show results container
            textResultsContainer.classList.remove('d-none');
            
            // Show success notification
            showNotification('Success', `Successfully generated ${data.images.length} images from text.`);
        }, 500);
    })
    .catch(error => {
        // Clear any progress interval
        if (window.textProgressInterval) {
            clearInterval(window.textProgressInterval);
        }
        
        // Hide loading indicator
        textLoadingIndicator.classList.add('d-none');
        
        // Show error notification
        showNotification('Error', 'Failed to generate images from text. Please try again.');
        console.error('Error generating images from text:', error);
    });
}

// Simulate progress for text-to-image generation
function simulateProgress() {
    let progress = 0;
    const steps = [
        'Loading models...',
        'Processing text prompt...',
        'Finding initial candidate...',
        'Optimizing latent space...',
        'Generating images...',
        'Applying quality filters...',
        'Finalizing results...'
    ];
    
    const interval = setInterval(() => {
        progress += Math.random() * 15 + 5; // Random increment between 5-20
        if (progress > 95) progress = 95; // Don't reach 100% until actually done
        
        const stepIndex = Math.floor((progress / 100) * steps.length);
        const currentStep = steps[Math.min(stepIndex, steps.length - 1)];
        
        updateProgress(progress, currentStep);
        
        if (progress >= 95) {
            clearInterval(interval);
        }
    }, 1000);
    
    // Store interval ID to clear it when generation completes
    window.textProgressInterval = interval;
}

// Update progress bar
function updateProgress(percentage, text) {
    if (textProgressBar) {
        textProgressBar.style.width = percentage + '%';
        textProgressBar.setAttribute('aria-valuenow', percentage);
    }
    if (textProgressText) {
        textProgressText.textContent = text;
    }
}

// Handle regenerate button click (image variations)
function handleRegenerate() {
    // Check if we have a previously uploaded image
    if (!lastUploadedImage) {
        showNotification('Error', 'No image to regenerate from. Please upload an image first.');
        return;
    }
    
    // Create a new FormData object
    const formData = new FormData();
    formData.append('shoe_image', lastUploadedImage);
    formData.append('variations_count', document.getElementById('variationsCount').value || 4);
    
    // Show loading indicator
    loadingIndicator.classList.remove('d-none');
    resultsContainer.classList.add('d-none');
    
    // Call the API to regenerate images
    fetch('/generate', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to regenerate images');
        }
        return response.json();
    })
    .then(data => {
        // Hide loading indicator
        loadingIndicator.classList.add('d-none');
        
        // Store generated images data
        generatedImages = data.images;
        
        // Display the generated images
        displayGeneratedImages(data.images);
        
        // Show results container
        resultsContainer.classList.remove('d-none');
        
        // Show success notification
        showNotification('Success', `Successfully regenerated ${data.images.length} variations.`);
    })
    .catch(error => {
        // Hide loading indicator
        loadingIndicator.classList.add('d-none');
        
        // Show error notification
        showNotification('Error', 'Failed to regenerate images. Please try again.');
        console.error('Error regenerating images:', error);
    });
}

// Handle text regenerate button click
function handleTextRegenerate() {
    if (!lastTextPrompt) {
        showNotification('Error', 'No prompt to regenerate from. Please enter a text prompt first.');
        return;
    }
    
    // Set the prompt back in the textarea
    if (textPrompt) {
        textPrompt.value = lastTextPrompt;
    }
    
    // Trigger text-to-image generation
    handleTextToImageSubmit(new Event('submit'));
}

// Display generated images in the UI (image variations)
function displayGeneratedImages(images) {
    // Clear previous results
    if (!generatedImagesContainer) {
        console.error("Generated images container not found");
        return;
    }
    
    generatedImagesContainer.innerHTML = '';
    
    // Add each image to the results container
    images.forEach((image, index) => {
        const col = document.createElement('div');
        col.className = 'col-sm-6 col-md-4 col-lg-3 mb-4'; // Updated for smaller cards
        
        const imageContainer = document.createElement('div');
        imageContainer.className = 'generated-image-container';
        imageContainer.dataset.imageId = image.id;
        
        const img = document.createElement('img');
        img.src = image.cloudinary_url || image.url;
        img.alt = `Generated Design ${index + 1}`;
        img.className = 'generated-image';
        
        const controls = document.createElement('div');
        controls.className = 'image-controls-always'; // Use always-visible controls
        
        const favoriteBtn = document.createElement('button');
        favoriteBtn.className = 'favorite-btn';
        favoriteBtn.innerHTML = image.is_favorite ? '<i class="fas fa-industry"></i>' : '<i class="far fa-industry"></i>';
        favoriteBtn.title = image.is_favorite ? 'Remove from Production' : 'Send to Production';
        favoriteBtn.type = 'button';
        favoriteBtn.style.pointerEvents = 'auto';
        favoriteBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            toggleFavorite(image.id, favoriteBtn);
        });
        
        if (image.is_favorite) {
            favoriteBtn.classList.add('active');
        }
        
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'download-btn';
        downloadBtn.innerHTML = '<i class="fas fa-download"></i>';
        downloadBtn.title = 'Download image';
        downloadBtn.type = 'button';
        downloadBtn.style.pointerEvents = 'auto';
        downloadBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            // Use Cloudinary URL for download if available
            downloadImage(image.cloudinary_url || image.url, `design-${index + 1}`);
        });
        
        controls.appendChild(favoriteBtn);
        controls.appendChild(downloadBtn);
        
        imageContainer.appendChild(img);
        imageContainer.appendChild(controls);
        col.appendChild(imageContainer);
        generatedImagesContainer.appendChild(col);
    });
}

// Display text-generated images in the UI
function displayTextGeneratedImages(images, prompt) {
    // Clear previous results
    if (!textGeneratedImagesContainer) {
        console.error("Text generated images container not found");
        return;
    }
    
    textGeneratedImagesContainer.innerHTML = '';
    
    // Display the prompt used
    if (generatedPromptDisplay) {
        generatedPromptDisplay.textContent = `Generated from: "${prompt}"`;
    }
    
    // Add each image to the results container
    images.forEach((image, index) => {
        const col = document.createElement('div');
        col.className = 'col-sm-6 col-md-4 col-lg-3 mb-4'; // Updated for smaller cards
        
        const imageContainer = document.createElement('div');
        imageContainer.className = 'generated-image-container';
        imageContainer.dataset.imageId = image.id;
        
        const img = document.createElement('img');
        img.src = image.cloudinary_url || image.url;
        img.alt = `Text Generated Design ${index + 1}`;
        img.className = 'generated-image';
        
        const controls = document.createElement('div');
        controls.className = 'image-controls-always'; // Use always-visible controls
        
        const favoriteBtn = document.createElement('button');
        favoriteBtn.className = 'favorite-btn';
        favoriteBtn.innerHTML = image.is_favorite ? '<i class="fas fa-industry"></i>' : '<i class="far fa-industry"></i>';
        favoriteBtn.title = image.is_favorite ? 'Remove from Production' : 'Send to Production';
        favoriteBtn.type = 'button';
        favoriteBtn.style.pointerEvents = 'auto';
        favoriteBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            toggleFavorite(image.id, favoriteBtn);
        });
        
        if (image.is_favorite) {
            favoriteBtn.classList.add('active');
        }
        
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'download-btn';
        downloadBtn.innerHTML = '<i class="fas fa-download"></i>';
        downloadBtn.title = 'Download image';
        downloadBtn.type = 'button';
        downloadBtn.style.pointerEvents = 'auto';
        downloadBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            downloadImage(image.cloudinary_url || image.url, `text-generated-${index + 1}`);
        });
        
        // Add quality indicator if loss is available
        if (image.loss !== undefined) {
            const qualityBadge = document.createElement('span');
            qualityBadge.className = 'badge bg-info position-absolute top-0 start-0 m-2';
            qualityBadge.textContent = `Quality: ${(1 - image.loss).toFixed(2)}`;
            qualityBadge.title = `Loss: ${image.loss.toFixed(4)}`;
            imageContainer.appendChild(qualityBadge);
        }
        
        controls.appendChild(favoriteBtn);
        controls.appendChild(downloadBtn);
        
        imageContainer.appendChild(img);
        imageContainer.appendChild(controls);
        col.appendChild(imageContainer);
        textGeneratedImagesContainer.appendChild(col);
    });
}

// Toggle favorite status for an image (now production status)
function toggleFavorite(designId, buttonElement) {
    console.log("Toggling production status for design:", designId);
    
    // Show a loading state on the button
    if (buttonElement) {
        buttonElement.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        buttonElement.disabled = true;
    }
    
    // Make a request to the server to toggle favorite status
    fetch(`/favorites/${designId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        console.log("Server response status:", response.status);
        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Toggle production response:", data);
        
        if (data.success) {
            // Update the UI based on the new production status
            if (buttonElement) {
                if (data.is_favorite) {
                    buttonElement.innerHTML = '<i class="fas fa-industry"></i>';
                    buttonElement.classList.add('active');
                    buttonElement.title = 'Remove from Production';
                    showNotification('Success', 'Added to production');
                } else {
                    buttonElement.innerHTML = '<i class="far fa-industry"></i>';
                    buttonElement.classList.remove('active');
                    buttonElement.title = 'Send to Production';
                    showNotification('Success', 'Removed from production');
                }
                buttonElement.disabled = false;
            }
            
            // Update the is_favorite property in the appropriate array
            const imageIndex = generatedImages.findIndex(img => img.id === designId);
            if (imageIndex !== -1) {
                generatedImages[imageIndex].is_favorite = data.is_favorite;
            }
            
            const textImageIndex = textGeneratedImages.findIndex(img => img.id === designId);
            if (textImageIndex !== -1) {
                textGeneratedImages[textImageIndex].is_favorite = data.is_favorite;
            }
            
            // Update history data if it exists
            updateHistoryItemProductionStatus(designId);
            
            // Refresh production section and favorites
            setTimeout(loadProductionDesigns, 500);
            setTimeout(loadFavorites, 600);
        } else {
            if (buttonElement) {
                // Reset button state
                const currentFavorite = [...generatedImages, ...textGeneratedImages]
                    .find(img => img.id === designId)?.is_favorite || false;
                buttonElement.innerHTML = currentFavorite ? 
                    '<i class="fas fa-industry"></i>' : 
                    '<i class="far fa-industry"></i>';
                buttonElement.disabled = false;
            }
            showNotification('Error', data.error || 'Failed to update production status');
        }
    })
    .catch(error => {
        console.error('Error toggling production status:', error);
        
        if (buttonElement) {
            // Reset button state
            const currentFavorite = [...generatedImages, ...textGeneratedImages]
                .find(img => img.id === designId)?.is_favorite || false;
            buttonElement.innerHTML = currentFavorite ? 
                '<i class="fas fa-industry"></i>' : 
                '<i class="far fa-industry"></i>';
            buttonElement.disabled = false;
        }
        
        showNotification('Error', 'Failed to update production status. Please try again.');
    });
}

// Toggle production status (alias for toggleFavorite with more clarity)
function toggleProduction(designId, addToProduction) {
    // Find button element if it exists
    const buttonElement = document.querySelector(`.generated-image-container[data-design-id="${designId}"] .favorite-btn`);
    toggleFavorite(designId, buttonElement);
}

// // Download all generated images (image variations)
// function downloadAllImages() {
//     if (!window.generatedImages || window.generatedImages.length === 0) {
//         showNotification('Error', 'No images to download');
//         return;
//     }
    
//     // Get all image paths, ensuring we use the correct path properties
//     const imagePaths = window.generatedImages
//         .map(image => {
//             // Log the image object to debug
//             console.log('Image object:', image);
            
//             // Use path if available, otherwise try to construct from id
//             if (image.path) {
//                 return image.path;
//             } else if (image.id) {
//                 return `${OUTPUT_FOLDER}/variation_${image.id}.jpg`;
//             }
//             return null;
//         })
//         .filter(path => path); // Remove any null/undefined values
    
//     console.log('Sending image paths for download:', imagePaths);
    
//     if (imagePaths.length === 0) {
//         showNotification('Error', 'No valid image paths found');
//         return;
//     }
    
//     // Send request to download all images as zip
//     fetch('/download-all', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ image_paths: imagePaths })
//     })
//     .then(response => {
//         if (!response.ok) {
//             throw new Error(`Failed to create zip file: ${response.status} ${response.statusText}`);
//         }
//         return response.blob();
//     })
//     .then(blob => {
//         // Check if the blob is empty or too small (possibly an empty zip)
//         if (blob.size < 100) { // Arbitrary small size that might indicate an empty file
//             console.warn('Warning: Downloaded zip appears to be very small or empty', blob.size);
//         }
        
//         // Create a link to download the zip
//         const link = document.createElement('a');
//         link.href = URL.createObjectURL(blob);
//         link.download = 'shoe-variations.zip';
//         document.body.appendChild(link);
//         link.click();
//         document.body.removeChild(link);
        
//         // Show notification
//         showNotification('Success', 'All images downloaded successfully');
//     })
//     .catch(error => {
//         console.error('Error downloading all images:', error);
//         showNotification('Error', 'Failed to download all images: ' + error.message);
//     });
// }
// Client-side ZIP creation for downloadAllImages
function downloadAllImages() {
    const images = generatedImages;
    
    if (!images || images.length === 0) {
        showNotification('Error', 'No images to download');
        return;
    }
    
    // Show notification while preparing
    showNotification('Processing', 'Preparing variations for download...');
    
    // Create a new JSZip instance
    const zip = new JSZip();
    const promises = [];
    
    // Add each image to the zip
    images.forEach((image, index) => {
        // Determine image URL
        let imageUrl = image.cloudinary_url || image.url;
        
        if (imageUrl) {
            const promise = fetch(imageUrl)
                .then(response => {
                    if (!response.ok) throw new Error(`Failed to fetch image: ${imageUrl}`);
                    return response.blob();
                })
                .then(blob => {
                    zip.file(`variation-${index + 1}.jpg`, blob);
                    return true;
                })
                .catch(error => {
                    console.error(`Error adding image to zip: ${imageUrl}`, error);
                    return false;
                });
            
            promises.push(promise);
        }
    });
    
    // Wait for all fetches to complete
    Promise.all(promises)
        .then(() => zip.generateAsync({ type: 'blob' }))
        .then(blob => {
            // Create a download link
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'shoe-variations.zip';
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            setTimeout(() => {
                URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }, 100);
            
            showNotification('Success', 'Variations downloaded successfully!');
        })
        .catch(error => {
            console.error('Error creating ZIP file:', error);
            showNotification('Error', 'Failed to download images: ' + error.message);
        });
}

// Download all text-generated images
// function downloadAllTextImages() {
//     if (!window.textGeneratedImages || window.textGeneratedImages.length === 0) {
//         showNotification('Error', 'No images to download');
//         return;
//     }
    
//     // Get all image paths, ensuring we use the correct path properties
//     const imagePaths = window.textGeneratedImages
//         .map(image => {
//             // Log the image object to debug
//             console.log('Text image object:', image);
            
//             // Use path if available, otherwise try to construct from id
//             if (image.path) {
//                 return image.path;
//             } else if (image.id) {
//                 return `${TEXT_TO_IMAGE_OUTPUT_FOLDER}/text_to_image_${image.id}.jpg`;
//             }
//             return null;
//         })
//         .filter(path => path); // Remove any null/undefined values
    
//     console.log('Sending text image paths for download:', imagePaths);
    
//     if (imagePaths.length === 0) {
//         showNotification('Error', 'No valid image paths found');
//         return;
//     }
    
//     // Send request to download all images as zip
//     fetch('/download-all', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ image_paths: imagePaths })
//     })
//     .then(response => {
//         if (!response.ok) {
//             throw new Error(`Failed to create zip file: ${response.status} ${response.statusText}`);
//         }
//         return response.blob();
//     })
//     .then(blob => {
//         // Check if the blob is empty or too small (possibly an empty zip)
//         if (blob.size < 100) { // Arbitrary small size that might indicate an empty file
//             console.warn('Warning: Downloaded zip appears to be very small or empty', blob.size);
//         }
        
//         // Create a link to download the zip
//         const link = document.createElement('a');
//         link.href = URL.createObjectURL(blob);
//         link.download = 'text-generated-shoes.zip';
//         document.body.appendChild(link);
//         link.click();
//         document.body.removeChild(link);
        
//         // Show notification
//         showNotification('Success', 'All text-generated images downloaded successfully');
//     })
//     .catch(error => {
//         console.error('Error downloading all text images:', error);
//         showNotification('Error', 'Failed to download all text-generated images: ' + error.message);
//     });
// }
// Client-side ZIP creation for downloadAllTextImages
function downloadAllTextImages() {
    const images = textGeneratedImages;
    
    if (!images || images.length === 0) {
        showNotification('Error', 'No images to download');
        return;
    }
    
    // Show notification while preparing
    showNotification('Processing', 'Preparing text-generated images for download...');
    
    // Create a new JSZip instance
    const zip = new JSZip();
    const promises = [];
    
    // Add each image to the zip
    images.forEach((image, index) => {
        // Determine image URL
        let imageUrl = image.cloudinary_url || image.url;
        
        if (imageUrl) {
            const promise = fetch(imageUrl)
                .then(response => {
                    if (!response.ok) throw new Error(`Failed to fetch image: ${imageUrl}`);
                    return response.blob();
                })
                .then(blob => {
                    zip.file(`text-design-${index + 1}.jpg`, blob);
                    return true;
                })
                .catch(error => {
                    console.error(`Error adding image to zip: ${imageUrl}`, error);
                    return false;
                });
            
            promises.push(promise);
        }
    });
    
    // Wait for all fetches to complete
    Promise.all(promises)
        .then(() => zip.generateAsync({ type: 'blob' }))
        .then(blob => {
            // Create a download link
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'text-generated-shoes.zip';
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            setTimeout(() => {
                URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }, 100);
            
            showNotification('Success', 'Text-generated images downloaded successfully!');
        })
        .catch(error => {
            console.error('Error creating ZIP file:', error);
            showNotification('Error', 'Failed to download images: ' + error.message);
        });
}
// Validate custom date range
function validateCustomDateRange(startDate, endDate) {
    const now = new Date();
    now.setHours(23, 59, 59, 999); // End of today
    
    // Check if dates are provided
    if (!startDate && !endDate) {
        return {
            valid: false,
            message: "Please select at least one date."
        };
    }
    
    // Parse dates if they exist
    const start = startDate ? new Date(startDate) : null;
    const end = endDate ? new Date(endDate) : null;
    
    // Validate start date is not in the future
    if (start && start > now) {
        return {
            valid: false,
            message: "Start date cannot be in the future."
        };
    }
    
    // Validate end date is not in the future
    if (end && end > now) {
        return {
            valid: false,
            message: "End date cannot be in the future."
        };
    }
    
    // If both dates are provided, validate start date is before end date
    if (start && end && start > end) {
        return {
            valid: false,
            message: "Start date must be before end date."
        };
    }
    
    return {
        valid: true,
        message: "Date range is valid."
    };
}

// Load history data
function loadHistoryData(reset = true) {
    if (reset) {
        historyPage = 1;
        if (historyItems) {
            historyItems.innerHTML = '';
        }
    }
    
    debugLog('Loading history data with filters:', historyFilters);
    
    // Show loading indicator
    if (historyLoadingIndicator) {
        historyLoadingIndicator.classList.remove('d-none');
    }
    
    // Hide no history message initially
    if (noHistoryMessage) {
        noHistoryMessage.classList.add('d-none');
    }
    
    // First, try to fetch from /design-history API if available
    fetch('/design-history' + constructHistoryQueryParams())
        .then(response => {
            if (!response.ok) {
                if (response.status === 404) {
                    // API not found, try the alternative approach
                    return fetchDesignsAlternative();
                }
                throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Hide loading indicator
            if (historyLoadingIndicator) {
                historyLoadingIndicator.classList.add('d-none');
            }
            
            processHistoryData(data);
        })
        .catch(error => {
            console.error('Error loading history data:', error);
            
            // Try alternative approach
            fetchDesignsAlternative();
        });
}

// Construct query parameters for history API
function constructHistoryQueryParams() {
    const params = new URLSearchParams();
    
    params.append('date_range', historyFilters.dateRange);
    params.append('type', historyFilters.type);
    params.append('status', historyFilters.status);
    params.append('sort_order', historyFilters.sortOrder);
    params.append('page', historyPage);
    params.append('limit', historyLimit);
    
    if (historyFilters.startDate) {
        params.append('start_date', historyFilters.startDate);
    }
    
    if (historyFilters.endDate) {
        params.append('end_date', historyFilters.endDate);
    }
    
    return '?' + params.toString();
}

// Alternative approach to fetch designs when the API is not available
function fetchDesignsAlternative() {
    debugLog('Using alternative approach to fetch designs');
    
    // Fetch from the favorites endpoint
    return fetch('/favorites')
        .then(response => {
            if (!response.ok) {
                if (response.status === 401) {
                    return { favorites: [] };
                }
                throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
        })
        .then(favoritesData => {
            const favorites = favoritesData.favorites || [];
            
            // Then combine with generated images from memory
            let designs = [...favorites];
            
            // Add designs from generatedImages global array (if it exists)
            if (typeof generatedImages !== 'undefined' && Array.isArray(generatedImages)) {
                // Convert to a compatible format
                const generatedDesigns = generatedImages.map(img => ({
                    design_id: img.id,
                    cloudinary_url: img.cloudinary_url,
                    url: img.url,
                    created_at: new Date().toISOString(), // Use current time as fallback
                    is_favorite: img.is_favorite || false,
                    type: img.type || 'variation',
                    path: img.path
                }));
                
                // Merge with existing designs, avoiding duplicates
                generatedDesigns.forEach(design => {
                    if (!designs.some(d => d.design_id === design.design_id)) {
                        designs.push(design);
                    }
                });
            }
            
            // Add designs from textGeneratedImages global array (if it exists)
            if (typeof textGeneratedImages !== 'undefined' && Array.isArray(textGeneratedImages)) {
                // Convert to a compatible format
                const textDesigns = textGeneratedImages.map(img => ({
                    design_id: img.id,
                    cloudinary_url: img.cloudinary_url,
                    url: img.url,
                    created_at: new Date().toISOString(), // Use current time as fallback
                    is_favorite: img.is_favorite || false,
                    type: 'text_to_image',
                    loss: img.loss,
                    path: img.path,
                    prompt: img.prompt
                }));
                
                // Merge with existing designs, avoiding duplicates
                textDesigns.forEach(design => {
                    if (!designs.some(d => d.design_id === design.design_id)) {
                        designs.push(design);
                    }
                });
            }
            
            // Apply manual filtering
            designs = manuallyFilterDesigns(designs);
            
            // Hide loading indicator
            if (historyLoadingIndicator) {
                historyLoadingIndicator.classList.add('d-none');
            }
            
            // Process the data
            processHistoryData({
                success: true,
                designs: designs,
                total: designs.length,
                page: historyPage,
                pages: Math.ceil(designs.length / historyLimit)
            });
        })
        .catch(error => {
            console.error('Error in alternative history fetch:', error);
            
            // Hide loading indicator
            if (historyLoadingIndicator) {
                historyLoadingIndicator.classList.add('d-none');
            }
            
            // Show error notification
            showNotification('Error', 'Failed to load design history. Please try again.');
            
            // Show no history message
            if (noHistoryMessage) {
                noHistoryMessage.classList.remove('d-none');
            }
            if (historyGrid) {
                historyGrid.classList.add('d-none');
            }
        });
}

// Apply manual filtering to designs (when API filtering is not available)
function manuallyFilterDesigns(designs) {
    let filteredDesigns = [...designs];
    
    // Apply type filter
    if (historyFilters.type !== 'all') {
        filteredDesigns = filteredDesigns.filter(design => design.type === historyFilters.type);
    }
    
    // Apply status filter - this should be applied regardless of date filter
    if (historyFilters.status !== 'all') {
        const isProduction = historyFilters.status === 'production';
        filteredDesigns = filteredDesigns.filter(design => design.is_favorite === isProduction);
    }
    
    // Apply date filter - don't restrict to production-only designs
    if (historyFilters.dateRange !== 'all') {
        const now = new Date();
        
        filteredDesigns = filteredDesigns.filter(design => {
            // Convert string date to Date object
            let createdAt;
            try {
                createdAt = new Date(design.created_at);
                if (isNaN(createdAt.getTime())) {
                    // If invalid date, use current time
                    createdAt = new Date();
                }
            } catch (e) {
                createdAt = new Date();
            }
            
            if (historyFilters.dateRange === 'today') {
                // Today only
                const today = new Date();
                today.setHours(0, 0, 0, 0);
                return createdAt >= today;
            } else if (historyFilters.dateRange === 'week') {
                // Last 7 days
                const weekAgo = new Date();
                weekAgo.setDate(weekAgo.getDate() - 7);
                return createdAt >= weekAgo;
            } else if (historyFilters.dateRange === 'month') {
                // Last 30 days
                const monthAgo = new Date();
                monthAgo.setDate(monthAgo.getDate() - 30);
                return createdAt >= monthAgo;
            } else if (historyFilters.dateRange === 'custom') {
                // Custom date range
                let valid = true;
                
                if (historyFilters.startDate) {
                    const startDate = new Date(historyFilters.startDate);
                    startDate.setHours(0, 0, 0, 0);
                    if (createdAt < startDate) {
                        valid = false;
                    }
                }
                
                if (historyFilters.endDate && valid) {
                    const endDate = new Date(historyFilters.endDate);
                    endDate.setHours(23, 59, 59, 999);
                    if (createdAt > endDate) {
                        valid = false;
                    }
                }
                
                return valid;
            }
            
            return true;
        });
    }
    
    // Apply sort
    filteredDesigns.sort((a, b) => {
        const dateA = new Date(a.created_at);
        const dateB = new Date(b.created_at);
        
        if (historyFilters.sortOrder === 'newest') {
            return dateB - dateA;
        } else {
            return dateA - dateB;
        }
    });
    
    return filteredDesigns;
}

// Process history data and update UI
function processHistoryData(data) {
    debugLog('Processing history data:', data);
    
    if (!data.success || !data.designs || !Array.isArray(data.designs)) {
        // Show error message
        if (noHistoryMessage) {
            noHistoryMessage.classList.remove('d-none');
        }
        if (historyGrid) {
            historyGrid.classList.add('d-none');
        }
        return;
    }
    
    const designs = data.designs;
    
    // Save to global variable
    historyData = designs;
    
    if (designs.length === 0) {
        // Show "no history" message
        if (noHistoryMessage) {
            noHistoryMessage.classList.remove('d-none');
        }
        if (historyGrid) {
            historyGrid.classList.add('d-none');
        }
        return;
    }
    
    // Hide "no history" message
    if (noHistoryMessage) {
        noHistoryMessage.classList.add('d-none');
    }
    if (historyGrid) {
        historyGrid.classList.remove('d-none');
    }
    
    // Display history items
    displayHistoryItems(designs);
    
    // Update load more button visibility
    if (loadMoreHistoryBtn) {
        if (data.page >= data.pages) {
            loadMoreHistoryBtn.classList.add('d-none');
        } else {
            loadMoreHistoryBtn.classList.remove('d-none');
        }
    }
}

// Display history items
function displayHistoryItems(designs) {
    if (!historyItems) {
        console.error("History items container not found");
        return;
    }
    
    debugLog(`Displaying ${designs.length} history items`);
    
    designs.forEach(design => {
        const col = document.createElement('div');
        col.className = 'col-sm-6 col-md-4 col-lg-3 mb-4';
        
        const item = document.createElement('div');
        item.className = 'history-item';
        item.dataset.designId = design.design_id;
        
        // Add click event to show details modal
        item.addEventListener('click', function() {
            showDesignDetails(design);
        });
        
        const img = document.createElement('img');
        img.src = design.cloudinary_url || design.url || '';
        img.alt = 'Design History';
        img.className = 'history-image';
        img.onerror = function() {
            // If image fails to load, try local URL based on type
            if (design.type === 'text_to_image') {
                const filename = design.local_path ? design.local_path.split('/').pop() : '';
                if (filename) {
                    img.src = `/text_to_image_outputs/${filename}`;
                } else if (design.design_id) {
                    // Try with design_id
                    img.src = `/text_to_image_outputs/text_to_image_${design.design_id}.jpg`;
                }
            } else {
                const filename = design.local_path ? design.local_path.split('/').pop() : '';
                if (filename) {
                    img.src = `/outputs/${filename}`;
                } else if (design.design_id) {
                    // Try with design_id
                    img.src = `/outputs/variation_${design.design_id}.jpg`;
                }
            }
        };
        
        // Status badge
        const statusBadge = document.createElement('div');
        statusBadge.className = design.is_favorite ? 
            'history-status production' : 
            'history-status draft';
        statusBadge.innerHTML = design.is_favorite ? 
            '<i class="fas fa-industry me-1"></i> Production' : 
            '<i class="fas fa-edit me-1"></i> Draft';
        
        const info = document.createElement('div');
        info.className = 'history-info';
        
        // Parse the creation date
        let createdDate;
        try {
            createdDate = new Date(design.created_at);
            if (isNaN(createdDate.getTime())) {
                createdDate = new Date(); // Use current date if invalid
            }
        } catch (e) {
            createdDate = new Date();
        }
        
        const formattedDate = createdDate.toLocaleDateString() + ' ' + 
            createdDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        info.innerHTML = `
            <div class="history-date">${formattedDate}</div>
            <div class="history-type">${design.type === 'text_to_image' ? 'Text-to-Image' : 'Image Variation'}</div>
        `;
        
        item.appendChild(img);
        item.appendChild(statusBadge);
        item.appendChild(info);
        col.appendChild(item);
        historyItems.appendChild(col);
    });
}

// Reset history and load again
function resetHistoryAndLoad() {
    if (historyItems) {
        historyItems.innerHTML = '';
    }
    historyPage = 1;
    loadHistoryData();
}

// Show design details in modal
function showDesignDetails(design) {
    if (!designDetailModal) return;
    
    // Set modal content
    if (modalDesignImage) {
        modalDesignImage.src = design.cloudinary_url || design.url || '';
        modalDesignImage.onerror = function() {
            // If image fails to load, try local URL based on type
            if (design.type === 'text_to_image') {
                const filename = design.local_path ? design.local_path.split('/').pop() : '';
                if (filename) {
                    modalDesignImage.src = `/text_to_image_outputs/${filename}`;
                } else if (design.design_id) {
                    // Try with design_id
                    modalDesignImage.src = `/text_to_image_outputs/text_to_image_${design.design_id}.jpg`;
                }
            } else {
                const filename = design.local_path ? design.local_path.split('/').pop() : '';
                if (filename) {
                    modalDesignImage.src = `/outputs/${filename}`;
                } else if (design.design_id) {
                    // Try with design_id
                    modalDesignImage.src = `/outputs/variation_${design.design_id}.jpg`;
                }
            }
        };
    }
    
    if (modalDesignId) {
        modalDesignId.textContent = design.design_id;
    }
    
    if (modalDesignCreated) {
        // Parse the creation date
        let createdDate;
        try {
            createdDate = new Date(design.created_at);
            if (isNaN(createdDate.getTime())) {
                createdDate = new Date(); // Use current date if invalid
            }
        } catch (e) {
            createdDate = new Date();
        }
        
        modalDesignCreated.textContent = createdDate.toLocaleDateString() + ' ' + 
            createdDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    }
    
    if (modalDesignType) {
        modalDesignType.textContent = design.type === 'text_to_image' ? 'Text-to-Image' : 'Image Variation';
    }
    
    if (modalDesignStatus) {
        modalDesignStatus.textContent = design.is_favorite ? 'Sent to Production' : 'Draft';
        
        // Style the status text
        if (design.is_favorite) {
            modalDesignStatus.classList.add('text-success');
            modalDesignStatus.classList.remove('text-secondary');
        } else {
            modalDesignStatus.classList.add('text-secondary');
            modalDesignStatus.classList.remove('text-success');
        }
    }
    
    // Show prompt if available
    if (modalDesignPromptContainer && modalDesignPrompt) {
        if (design.prompt) {
            modalDesignPromptContainer.classList.remove('d-none');
            modalDesignPrompt.textContent = design.prompt;
        } else {
            modalDesignPromptContainer.classList.add('d-none');
        }
    }
    
    // Show quality score if available
    if (modalDesignQualityContainer && modalDesignQuality) {
        if (design.loss !== undefined) {
            modalDesignQualityContainer.classList.remove('d-none');
            const qualityScore = (1 - design.loss).toFixed(2);
            modalDesignQuality.textContent = qualityScore;
        } else {
            modalDesignQualityContainer.classList.add('d-none');
        }
    }
    
    // Update production button text and state
    if (modalDesignProductionBtn && modalProductionBtnText) {
        if (design.is_favorite) {
            modalProductionBtnText.textContent = 'Remove from Production';
            modalDesignProductionBtn.classList.remove('btn-primary');
            modalDesignProductionBtn.classList.add('btn-danger');
        } else {
            modalProductionBtnText.textContent = 'Send to Production';
            modalDesignProductionBtn.classList.remove('btn-danger');
            modalDesignProductionBtn.classList.add('btn-primary');
        }
    }
    
    // Show the modal
    const modal = new bootstrap.Modal(designDetailModal);
    modal.show();
}

// Find design by ID across all available data
function findDesignById(designId) {
    // Look in history data
    const historyDesign = historyData.find(d => d.design_id === designId);
    if (historyDesign) return historyDesign;
    
    // Look in generated images
    if (typeof generatedImages !== 'undefined' && Array.isArray(generatedImages)) {
        const generatedImage = generatedImages.find(img => img.id === designId);
        if (generatedImage) return {
            design_id: generatedImage.id,
            cloudinary_url: generatedImage.cloudinary_url,
            url: generatedImage.url,
            created_at: new Date().toISOString(),
            type: generatedImage.type || 'variation',
            is_favorite: generatedImage.is_favorite || false
        };
    }
    
    // Look in text generated images
    if (typeof textGeneratedImages !== 'undefined' && Array.isArray(textGeneratedImages)) {
        const textGeneratedImage = textGeneratedImages.find(img => img.id === designId);
        if (textGeneratedImage) return {
            design_id: textGeneratedImage.id,
            cloudinary_url: textGeneratedImage.cloudinary_url,
            url: textGeneratedImage.url,
            created_at: new Date().toISOString(),
            type: 'text_to_image',
            is_favorite: textGeneratedImage.is_favorite || false,
            loss: textGeneratedImage.loss,
            prompt: textGeneratedImage.prompt
        };
    }
    
    return null;
}

// Update production status of history item
function updateHistoryItemProductionStatus(designId) {
    // Find the design in history data
    const designIndex = historyData.findIndex(d => d.design_id === designId);
    if (designIndex !== -1) {
        // Toggle the production status
        historyData[designIndex].is_favorite = !historyData[designIndex].is_favorite;
        
        // Update the UI
        const statusBadge = document.querySelector(`.history-item[data-design-id="${designId}"] .history-status`);
        if (statusBadge) {
            if (historyData[designIndex].is_favorite) {
                statusBadge.className = 'history-status production';
                statusBadge.innerHTML = '<i class="fas fa-industry me-1"></i> Production';
            } else {
                statusBadge.className = 'history-status draft';
                statusBadge.innerHTML = '<i class="fas fa-edit me-1"></i> Draft';
            }
        }
    }
}

// Load production designs (formerly favorites)
function loadProductionDesigns() {
    console.log("Loading production designs...");
    
    fetch('/favorites')
        .then(response => {
            if (!response.ok) {
                if (response.status === 401) {
                    return { favorites: [] };
                }
                throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Production designs data:", data);
            const productionDesigns = data.favorites || [];
            
            if (productionDesigns.length === 0) {
                // Show "no production designs" message
                if (noProductionMessage) {
                    noProductionMessage.classList.remove('d-none');
                }
                if (productionGrid) {
                    productionGrid.classList.add('d-none');
                }
                if (productionActions) {
                    productionActions.classList.add('d-none');
                }
                return;
            }
            
            // Hide "no production designs" message
            if (noProductionMessage) {
                noProductionMessage.classList.add('d-none');
            }
            if (productionGrid) {
                productionGrid.classList.remove('d-none');
            }
            if (productionActions) {
                productionActions.classList.remove('d-none');
            }
            
            // Sort production designs by created_at (newest first)
            productionDesigns.sort((a, b) => {
                return new Date(b.created_at) - new Date(a.created_at);
            });
            
            // Display production designs in the grid
            displayProductionDesigns(productionDesigns);
        })
        .catch(error => {
            console.error('Error loading production designs:', error);
            if (!error.message.includes('401')) {
                showNotification('Error', 'Failed to load production designs. Please refresh the page.');
            }
        });
}

// Display production designs in the grid
function displayProductionDesigns(designs) {
    if (!productionItems) return;
    
    // Clear previous items
    productionItems.innerHTML = '';
    
    // Add each design to the grid
    designs.forEach((design, index) => {
        const col = document.createElement('div');
        col.className = 'col-sm-6 col-md-4 col-lg-3 mb-4';
        
        const item = document.createElement('div');
        item.className = 'production-item';
        item.dataset.designId = design.design_id;
        
        // Add click event to show details modal
        item.addEventListener('click', function() {
            showDesignDetails(design);
        });
        
        const img = document.createElement('img');
        img.src = design.cloudinary_url || design.url || '';
        img.alt = 'Production Design';
        img.className = 'production-image';
        img.onerror = function() {
            // If image fails to load, try local URL based on type
            if (design.type === 'text_to_image') {
                const filename = design.local_path ? design.local_path.split('/').pop() : '';
                if (filename) {
                    img.src = `/text_to_image_outputs/${filename}`;
                } else if (design.design_id) {
                    // Try with design_id
                    img.src = `/text_to_image_outputs/text_to_image_${design.design_id}.jpg`;
                }
            } else {
                const filename = design.local_path ? design.local_path.split('/').pop() : '';
                if (filename) {
                    img.src = `/outputs/${filename}`;
                } else if (design.design_id) {
                    // Try with design_id
                    img.src = `/outputs/variation_${design.design_id}.jpg`;
                }
            }
        };
        
        const badge = document.createElement('div');
        badge.className = 'production-badge';
        badge.innerHTML = '<i class="fas fa-industry me-1"></i> Production Ready';
        
        const info = document.createElement('div');
        info.className = 'production-info';
        
        // Parse the creation date
        let createdDate;
        try {
            createdDate = new Date(design.created_at);
            if (isNaN(createdDate.getTime())) {
                createdDate = new Date(); // Use current date if invalid
            }
        } catch (e) {
            createdDate = new Date();
        }
        
        const formattedDate = createdDate.toLocaleDateString();
        
        info.innerHTML = `
            <div class="small">${formattedDate}</div>
            <div class="fw-bold">${design.type === 'text_to_image' ? 'Text-to-Image' : 'Image Variation'}</div>
        `;
        
        const controls = document.createElement('div');
        controls.className = 'production-controls';
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-production-btn';
        removeBtn.innerHTML = '<i class="fas fa-times"></i>';
        removeBtn.title = 'Remove from Production';
        removeBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            toggleProduction(design.design_id, false);
        });
        
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'download-btn';
        downloadBtn.innerHTML = '<i class="fas fa-download"></i>';
        downloadBtn.title = 'Download image';
        downloadBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            downloadImage(design.cloudinary_url || design.url, `design-${index + 1}`);
        });
        
        controls.appendChild(removeBtn);
        controls.appendChild(downloadBtn);
        
        item.appendChild(img);
        item.appendChild(badge);
        item.appendChild(info);
        item.appendChild(controls);
        col.appendChild(item);
        productionItems.appendChild(col);
    });
}

// // Download all production designs
// function downloadAllProductionDesigns() {
//     // Fetch all production designs
//     fetch('/favorites')
//         .then(response => {
//             if (!response.ok) {
//                 throw new Error(`Failed to fetch production designs: ${response.status} ${response.statusText}`);
//             }
//             return response.json();
//         })
//         .then(data => {
//             const productionDesigns = data.favorites || [];
            
//             if (productionDesigns.length === 0) {
//                 showNotification('Error', 'No designs to download');
//                 return;
//             }
            
//             // Extract all image paths from designs with thorough logging
//             console.log('Production designs to process:', productionDesigns);
            
//             const imagePaths = [];
            
//             productionDesigns.forEach(design => {
//                 console.log('Processing design for download:', design);
                
//                 // Check for path directly
//                 if (design.path) {
//                     console.log('Using direct path:', design.path);
//                     imagePaths.push(design.path);
//                 }
//                 // Check for local_path
//                 else if (design.local_path) {
//                     console.log('Using local_path:', design.local_path);
//                     imagePaths.push(design.local_path);
//                 } 
//                 // If no direct path, construct from design_id and type
//                 else if (design.design_id) {
//                     let path;
//                     if (design.type === 'text_to_image') {
//                         path = `${TEXT_TO_IMAGE_OUTPUT_FOLDER}/text_to_image_${design.design_id}.jpg`;
//                     } else {
//                         path = `${OUTPUT_FOLDER}/variation_${design.design_id}.jpg`;
//                     }
//                     console.log('Constructed path from ID:', path);
//                     imagePaths.push(path);
//                 } else {
//                     console.warn('Design has no path or ID information:', design);
//                 }
//             });
            
//             if (imagePaths.length === 0) {
//                 showNotification('Error', 'No valid design paths found');
//                 return;
//             }
            
//             console.log('Sending production image paths for download:', imagePaths);
            
//             // Send request to download all images as zip
//             fetch('/download-all', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json',
//                 },
//                 body: JSON.stringify({ image_paths: imagePaths })
//             })
//             .then(response => {
//                 if (!response.ok) {
//                     throw new Error(`Failed to create zip file: ${response.status} ${response.statusText}`);
//                 }
//                 return response.blob();
//             })
//             .then(blob => {
//                 // Check if the blob is empty or too small (possibly an empty zip)
//                 if (blob.size < 100) { // Arbitrary small size that might indicate an empty file
//                     console.warn('Warning: Downloaded zip appears to be very small or empty', blob.size);
//                 }
                
//                 // Create a link to download the zip
//                 const link = document.createElement('a');
//                 link.href = URL.createObjectURL(blob);
//                 link.download = 'production-designs.zip';
//                 document.body.appendChild(link);
//                 link.click();
//                 document.body.removeChild(link);
                
//                 // Show notification
//                 showNotification('Success', 'All production designs downloaded successfully');
//             })
//             .catch(error => {
//                 console.error('Error downloading production designs:', error);
//                 showNotification('Error', 'Failed to download production designs: ' + error.message);
//             });
//         })
//         .catch(error => {
//             console.error('Error fetching production designs:', error);
//             showNotification('Error', 'Failed to fetch production designs: ' + error.message);
//         });
// }
// Client-side ZIP creation for downloadAllProductionDesigns
function downloadAllProductionDesigns() {
    // Fetch all production designs
    fetch('/favorites')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to fetch production designs: ${response.status} ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            const productionDesigns = data.favorites || [];
            
            if (productionDesigns.length === 0) {
                showNotification('Error', 'No designs to download');
                return;
            }
            
            // Show notification while preparing
            showNotification('Processing', 'Preparing production designs for download...');
            
            // Create a new JSZip instance
            const zip = new JSZip();
            const promises = [];
            
            // Add each design to the zip
            productionDesigns.forEach((design, index) => {
                // Determine image URL
                let imageUrl = design.cloudinary_url || design.url;
                
                // If no URL, try to construct one
                if (!imageUrl) {
                    if (design.type === 'text_to_image') {
                        imageUrl = `/text_to_image_outputs/text_to_image_${design.design_id}.jpg`;
                    } else {
                        imageUrl = `/outputs/variation_${design.design_id}.jpg`;
                    }
                }
                
                if (imageUrl) {
                    const promise = fetch(imageUrl)
                        .then(response => {
                            if (!response.ok) throw new Error(`Failed to fetch image: ${imageUrl}`);
                            return response.blob();
                        })
                        .then(blob => {
                            zip.file(`design-${index + 1}.jpg`, blob);
                            return true;
                        })
                        .catch(error => {
                            console.error(`Error adding image to zip: ${imageUrl}`, error);
                            return false;
                        });
                    
                    promises.push(promise);
                }
            });
            
            // Wait for all fetches to complete
            return Promise.all(promises).then(() => zip.generateAsync({ type: 'blob' }));
        })
        .then(blob => {
            if (!blob) {
                throw new Error('Failed to create ZIP file');
            }
            
            // Create a download link
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'production-designs.zip';
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            setTimeout(() => {
                URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }, 100);
            
            showNotification('Success', 'Production designs downloaded successfully!');
        })
        .catch(error => {
            console.error('Error creating ZIP file:', error);
            showNotification('Error', 'Failed to download designs: ' + error.message);
        });
}

// Send all designs to manufacturing
function sendAllToManufacturing() {
    // This is a placeholder for what would be an integration with a manufacturing system
    // In a real implementation, this would send the designs to a manufacturing partner
    
    showNotification('Success', 'All designs sent to manufacturing partner');
    
    // Create a visual confirmation for the user
    if (productionItems) {
        const items = productionItems.querySelectorAll('.production-item');
        items.forEach(item => {
            item.classList.add('pulse-animation');
            
            // Remove the animation after a few seconds
            setTimeout(() => {
                item.classList.remove('pulse-animation');
            }, 3000);
        });
    }
}

// Load favorites from server (for carousel in dashboard)
function loadFavorites() {
    console.log("Loading favorites for carousel...");
    
    // Add a slight delay to allow the page to fully load first
    setTimeout(() => {
        fetch('/favorites')
            .then(response => {
                console.log("Favorites response status:", response.status);
                if (!response.ok) {
                    if (response.status === 401) {
                        // Not logged in - don't show error
                        return { favorites: [] };
                    }
                    throw new Error(`Server responded with status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Favorites data:", data);
                const favorites = data.favorites || [];
                
                // PROPERLY clear the favorites slider first
                if (favoritesSlider) {
                    // Destroy existing slick carousel if it exists
                    if (typeof $ !== 'undefined' && typeof $.fn.slick !== 'undefined') {
                        if ($('.favorites-slider').hasClass('slick-initialized')) {
                            $('.favorites-slider').slick('unslick');
                        }
                    }
                    // Clear all content
                    favoritesSlider.innerHTML = '';
                }
                
                if (favorites.length === 0) {
                    // Show "no favorites" message
                    if (noFavoritesMessage) {
                        noFavoritesMessage.classList.remove('d-none');
                    }
                    if (favoritesSlider) {
                        favoritesSlider.classList.add('d-none');
                    }
                    return;
                }
                
                // Hide "no favorites" message
                if (noFavoritesMessage) {
                    noFavoritesMessage.classList.add('d-none');
                }
                if (favoritesSlider) {
                    favoritesSlider.classList.remove('d-none');
                }
                
                // Sort favorites by created_at (newest first)
                favorites.sort((a, b) => {
                    return new Date(b.created_at) - new Date(a.created_at);
                });
                
                // Add each favorite to the slider
                favorites.forEach(favorite => {
                    if (!favoritesSlider) return;
                    
                    const slide = document.createElement('div');
                    slide.className = 'favorite-slide';
                    
                    const img = document.createElement('img');
                    img.src = favorite.cloudinary_url || favorite.url || '';
                    img.alt = 'Favorite Design';
                    img.className = 'favorite-image';
                    img.onerror = function() {
                        // If image fails to load, try local URL based on type
                        if (favorite.type === 'text_to_image') {
                            const filename = favorite.local_path ? favorite.local_path.split('/').pop() : '';
                            img.src = `/text_to_image_outputs/${filename}`;
                        } else {
                            const filename = favorite.local_path ? favorite.local_path.split('/').pop() : '';
                            img.src = `/outputs/${filename}`;
                        }
                    };
                    
                    const controls = document.createElement('div');
                    controls.className = 'favorite-controls';
                    
                    const removeBtn = document.createElement('button');
                    removeBtn.className = 'remove-favorite-btn';
                    removeBtn.innerHTML = '<i class="fas fa-trash"></i>';
                    removeBtn.title = 'Remove from production';
                    removeBtn.addEventListener('click', function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        // Use the design_id for unfavoriting
                        toggleFavorite(favorite.design_id, null);
                    });
                    
                    // Add type badge
                    const typeBadge = document.createElement('span');
                    typeBadge.className = 'badge bg-secondary position-absolute top-0 end-0 m-2';
                    typeBadge.textContent = favorite.type === 'text_to_image' ? 'Text' : 'Variation';
                    
                    controls.appendChild(removeBtn);
                    slide.appendChild(img);
                    slide.appendChild(typeBadge);
                    slide.appendChild(controls);
                    favoritesSlider.appendChild(slide);
                });
                
                // Initialize the slick carousel AFTER adding all slides
                initializeSlider();
            })
            .catch(error => {
                console.error('Error loading favorites:', error);
                // Only show the error if it's not a 401 authentication error
                if (!error.message.includes('401')) {
                    showNotification('Error', 'Failed to load favorites. Please refresh the page.');
                }
            });
    }, 500); // Add a 500ms delay
}

// Initialize the slider with Slick
function initializeSlider() {
    // Check if jQuery and slick are available
    if (typeof $ === 'undefined' || typeof $.fn.slick === 'undefined') {
        console.error('jQuery or Slick Carousel not available');
        return;
    }
    
    // Make sure we have slides to show
    if (!favoritesSlider || favoritesSlider.children.length === 0) {
        console.log('No slides to initialize slider with');
        return;
    }
    
    // Double-check that slick is not already initialized
    if ($('.favorites-slider').hasClass('slick-initialized')) {
        console.log('Slider already initialized, skipping');
        return;
    }
    
    // Initialize slick carousel
    try {
        $('.favorites-slider').slick({
            dots: true,
            infinite: false,
            speed: 500,
            slidesToShow: 4,
            slidesToScroll: 1,
            responsive: [
                {
                    breakpoint: 1024,
                    settings: {
                        slidesToShow: 3,
                        slidesToScroll: 1
                    }
                },
                {
                    breakpoint: 768,
                    settings: {
                        slidesToShow: 2,
                        slidesToScroll: 1
                    }
                },
                {
                    breakpoint: 576,
                    settings: {
                        slidesToShow: 1,
                        slidesToScroll: 1
                    }
                }
            ]
        });
        console.log('Slider initialized successfully');
    } catch (error) {
        console.error('Error initializing slider:', error);
    }
}

// Helper function for downloading image
function downloadImage(url, filename) {
    fetch(url)
        .then(response => response.blob())
        .then(blob => {
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = `${filename}.jpg`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            // Show notification
            showNotification('Success', 'Image downloaded successfully');
        })
        .catch(error => {
            console.error('Error downloading image:', error);
            showNotification('Error', 'Failed to download image');
        });
}

// Show notification toast
function showNotification(title, message) {
    console.log(`${title}: ${message}`); // Always log for debugging
    
    // Set toast content
    if (!toastTitle || !toastMessage || !notificationToast) {
        // Elements not found, just log to console
        console.warn('Toast elements not found, could not show notification UI');
        return;
    }
    
    toastTitle.textContent = title;
    toastMessage.textContent = message;
    
    // Create a Bootstrap toast instance if it doesn't exist
    if (typeof bootstrap !== 'undefined' && notificationToast) {
        const toast = new bootstrap.Toast(notificationToast);
        toast.show();
    } else {
        // Fallback if Bootstrap is not available
        alert(`${title}: ${message}`);
    }
}