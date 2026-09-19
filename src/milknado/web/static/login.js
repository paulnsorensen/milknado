function tokenFromLaunchUrl(launchUrl) {
  return new URL(launchUrl).searchParams.get("token") ?? "";
}

function submitLaunchUrl(form, launchUrl, token) {
  token.value = tokenFromLaunchUrl(launchUrl.value);
  if (token.value) form.submit();
}

if (typeof module !== "undefined") {
  module.exports = { submitLaunchUrl, tokenFromLaunchUrl };
}

if (typeof document !== "undefined") {
  const form = document.querySelector("form");
  if (form) {
    const launchUrl = document.querySelector("#launch-url");
    const token = document.querySelector("#token");
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      submitLaunchUrl(form, launchUrl, token);
    });
  }
}
