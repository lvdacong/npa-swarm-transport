# Offline Chinese UI font

`swarm-cjk.woff` is a subset of Noto Sans SC, distributed under the accompanying
SIL Open Font License (`OFL.txt`). Source: the Google Fonts repository,
`ofl/notosanssc/NotoSansSC[wght].ttf`, downloaded 2026-10-23.

The subset includes Chinese and symbol characters present in `web/index.html`,
`web/app.js`, and `src/npa_swarm/server.py`; it was produced with FontTools.
System fonts take precedence. This small local fallback prevents empty squares
in Linux/browser CI without requiring any font-network requests at runtime.
Regenerate the subset when adding new Chinese interface copy.
