let interval = null;

function register() {
  const name = document.getElementById("name").value;
  const email = document.getElementById("email").value;
  const consent = document.getElementById("consent").checked;

  if (!name || !email) {
    alert("Please enter name and email");
    return;
  }

  if (!consent) {
    alert("Consent is required");
    return;
  }

  fetch("/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email, consent })
  })
  .then(() => {
    document.getElementById("registerBox").classList.add("hidden");
    document.getElementById("emotionBox").classList.remove("hidden");
  });
}

function startDetection() {
  fetch("/start", { method: "POST" });

  if (interval) return;

  interval = setInterval(() => {
    fetch("/rating")
      .then(res => res.json())
      .then(data => {
        document.getElementById("stars").innerText = data.stars;
        document.getElementById("avg").innerText =
          "Average: " + data.average;
      });
  }, 1000);
}

function stopDetection() {
  fetch("/stop", { method: "POST" })
    .then(res => res.json())
    .then(data => {
      document.getElementById("stars").innerText =
        "★".repeat(data.stars) + "☆".repeat(5 - data.stars);
      document.getElementById("avg").innerText =
        "Final Average: " + data.average;
    });

  clearInterval(interval);
  interval = null;
}

function submitReview() {
  fetch("/submit", { method: "POST" })
    .then(res => res.json())
    .then(() => {
      alert("Review submitted successfully!");
    });
}
