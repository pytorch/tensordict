// Point the "View on GitHub" button of a tutorial to its source file.
//
// pytorch_sphinx_theme2 sets the button to "<folder>_source/<name>.py", the
// layout of the pytorch/tutorials repository. The tensordict tutorials are
// in tutorials/sphinx_tuto/. The "load" event fires after the theme has set
// the link.
window.addEventListener("load", function () {
  var typeEl = document.getElementById("tutorial-type");
  var githubLink = document.getElementById("github-link");
  if (!typeEl || !githubLink) {
    return;
  }
  var name = typeEl.textContent.trim().split("/").pop();
  if (name) {
    githubLink.setAttribute(
      "href",
      "https://github.com/pytorch/tensordict/blob/main/tutorials/sphinx_tuto/" +
        name +
        ".py"
    );
  }
});
