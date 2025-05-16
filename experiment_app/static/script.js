document.addEventListener("DOMContentLoaded", () => {
    const groupId = window.location.pathname.split("/")[1];
    const memeContainer = document.getElementById("meme-container");
    const searchButton = document.getElementById("search-button");
    const loadingMessage = document.getElementById("loading-message");
    let selectedMemes = [];
    let currentFeedback = null;

    // Load initial memes
    loadRandomMemes();

    function loadRandomMemes() {
        loadingMessage.style.display = "block";
        memeContainer.innerHTML = "";

        fetch(`/get_random_memes/${groupId}`)
            .then(response => response.json())
            .then(data => {
                memeContainer.innerHTML = "";
                loadingMessage.style.display = "none";
                
                data.forEach(meme => {
                    const img = document.createElement("img");
                    img.src = `data:image/png;base64,${meme.image_base64}`;
                    img.classList.add("meme");
                    img.dataset.name = meme.image_name;

                    img.addEventListener("click", () => {
                        if (selectedMemes.includes(meme.image_name)) {
                            selectedMemes = selectedMemes.filter(name => name !== meme.image_name);
                            img.classList.remove("selected");
                        } else if (selectedMemes.length < 5) {
                            selectedMemes.push(meme.image_name);
                            img.classList.add("selected");
                        }
                        searchButton.disabled = selectedMemes.length !== 5;
                    });

                    memeContainer.appendChild(img);
                });
            });
    }

    searchButton.addEventListener("click", () => {
        if (!currentFeedback) {
            // First search (no feedback yet)
            fetchRecommendations({ selected_images: selectedMemes });
        } else {
            // Subsequent searches with feedback
            fetchRecommendations({
                selected_images: selectedMemes,
                feedback_scores: getFeedbackScores()
            });
        }
    });

let feedbackHistory = {};

function displayResults(results, containerId, showFeedback) {
    // Proper container declaration
    const container = document.getElementById(containerId);
    container.innerHTML = "";

    if (showFeedback) {
        const feedbackHeader = document.createElement("h3");
        feedbackHeader.textContent = "Rate these recommendations (-5 = hate, 5 = love):";
        container.appendChild(feedbackHeader);
    }

    // Create meme elements
    results.forEach(result => {
        const wrapper = document.createElement("div");
        wrapper.className = "meme-wrapper";

        const img = document.createElement("img");
        img.src = `data:image/png;base64,${result.base64}`;
        img.className = "meme";
        wrapper.appendChild(img);

        if (showFeedback) {
            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = -5;
            slider.max = 5;
            slider.value = 0;
            slider.dataset.name = result.image_name;
            slider.className = "feedback-slider";
            wrapper.appendChild(slider);

            const scoreDisplay = document.createElement("span");
            scoreDisplay.className = "score-display";
            scoreDisplay.textContent = "0";
            wrapper.appendChild(scoreDisplay);

            slider.addEventListener("input", () => {
                scoreDisplay.textContent = slider.value;
            });
        }

        container.appendChild(wrapper);
    });

    // Add buttons AFTER creating all memes
    if (showFeedback) {
        const buttonContainer = document.createElement("div");
        buttonContainer.style.margin = "20px 0";

        // Refine Button
        const refineButton = document.createElement("button");
        refineButton.textContent = "Refine Recommendations";
        refineButton.className = "refine-button";
        refineButton.addEventListener("click", () => {
            fetchRecommendations({
                selected_images: selectedMemes,
                feedback_scores: getFeedbackScores()
            });
        });

        // Reset Button
        const resetButton = document.createElement("button");
        resetButton.textContent = "Reset Feedback";
        resetButton.className = "reset-button";
        resetButton.style.marginLeft = "10px";
        resetButton.addEventListener("click", () => {
            feedbackHistory = {};
            document.querySelectorAll(".feedback-slider").forEach(slider => {
                slider.value = 0;
                slider.nextElementSibling.textContent = "0";
            });
        });

        buttonContainer.appendChild(refineButton);
        buttonContainer.appendChild(resetButton);
        container.appendChild(buttonContainer);
    }
}

// Updated fetchRecommendations
function fetchRecommendations(baseRequest) {
    const currentScores = getFeedbackScores();
    feedbackHistory = {...feedbackHistory, ...currentScores};

    fetch(`/find_similar/${groupId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            ...baseRequest,
            feedback_scores: feedbackHistory
        })
    })
    .then(response => response.json())
    .then(data => {
        displayResults(data.method1, "method1-results", false);
        displayResults(data.method2, "method2-results", true);
    });
}


    function getFeedbackScores() {
        const feedback = {};
        document.querySelectorAll(".feedback-slider").forEach(slider => {
            feedback[slider.dataset.name] = parseInt(slider.value);
        });
        return feedback;
    }
});