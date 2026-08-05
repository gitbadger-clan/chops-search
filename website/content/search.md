+++
title = "Search"
date = 1970-01-01
weight = 999
in_search_index = false
+++

<div class="chops">
  <div class="chops-bar">
    <span class="chops-icon"></span>
    <input id="chops-input" type="search" placeholder="Search…"
           autocomplete="off" spellcheck="false"
           role="combobox" aria-expanded="false" aria-controls="chops-results"
           aria-autocomplete="list">
    <button type="button" class="chops-clear" data-chops-clear></button>
  </div>
  <span id="chops-mode" class="chops-mode" aria-live="polite"></span>
  <ul id="chops-results" class="chops-results" role="listbox"></ul>
</div>

<link rel="stylesheet" href="/search/chops-search.css">
<script defer src="/search/chops-search.js"></script>
