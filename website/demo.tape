# chops-search hero demo: zero to measured search on a Zola site.
#
# Assumptions:
#   - cwd is a small demo Zola site (a handful of posts, fixtures/queries.toml present)
#   - chops-search is installed
#   - the model has NOT been fetched yet in this directory (the fetch runs hidden)
#   - replace <hash> in the curl beat with the real hash from manifest.json
#
# Render: vhs demo.tape

Output demo.gif

Set Shell "bash"
Set FontSize 17
Set Width 1200
Set Height 680
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 55ms

# ---- beat 1: scaffold ------------------------------------------------
Type "chops-search init"
Enter
Sleep 3s

# ---- beat 2: the two template lines, in a clean nvim -----------------
# CUT MARKER: drop this beat for the dev.to 500-frame version
# Before recording: make sure base.html does NOT already contain the
# chops-search lines, or this beat inserts duplicates.
Type "nvim --clean templates/base.html"
Enter
Sleep 1.2s
# Jump to the content block; the runtime lines go right after it,
# matching where the real site loads them (end of <body>).
Type "/endblock body"
Enter
Sleep 600ms
Type "o"
Sleep 300ms
Type@40ms '<link rel="stylesheet" href="/search/chops-search.css">'
Enter
Type@40ms '<script defer src="/search/chops-search.js"></script>'
Sleep 500ms
Escape
Type ":wq"
Enter
Sleep 1s

# ---- beat 3: model, lockfile story -----------------------------------
# The 30 MB fetch happens off camera; verify is instant and tells the
# reproducibility story better than a progress bar would.
Hide
Type "chops-search model fetch"
Enter
Sleep 60s
Type "clear"
Enter
Show

Type "chops-search model verify"
Enter
Sleep 2.5s

# ---- beat 4: build ---------------------------------------------------
Type "chops-search build"
Enter
Sleep 4s

# ---- beat 5: the explain output (money shot) -------------------------
Type "clear"
Enter
Type 'chops-search query "what happens to queued jobs when the process dies"'
Enter
Sleep 6s

# ---- beat 6: eval ----------------------------------------------------
Type "clear"
Enter
Type "chops-search eval"
Enter
Sleep 6s

# ---- beat 7: closer, proves the range-fetch claim over HTTP ----------
# CUT MARKER: also droppable for the short version
Type "clear"
Enter
Type 'curl -sI -H "Range: bytes=0-127" https://chops-search.gitbadger.com/search/model.rows.<hash>.i8 | head -4'
Enter
Sleep 5s
