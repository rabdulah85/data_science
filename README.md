---
title: Gdp Indonesia
emoji: 🇮🇩
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
license: mit
short_description: Indonesia GDP dashboard — districts, grid GDP, convergence & ESDA
---

# GDP Indonesia Dashboard

Interactive Streamlit dashboard for Indonesia's regional GDP:

- **514 districts** — choropleth map, time series, ranking, σ/β-convergence, descriptive statistics
- **0.25° grid GDP** (2012–2022, PWT 11.0-rescaled, GADM 4.10) — grid map, complete descriptive
  statistics, and **ESDA** (Global Moran's I, LISA cluster map, Getis-Ord Gi\* hot/cold spots)

## Deployment

This single repository deploys to two targets from one `git push`:

- **Hugging Face Space** (Docker) — builds from this `Dockerfile`, runs `app.py`.
- **Streamlit Community Cloud** — runs `app.py` directly.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

- District GDP + GeoJSON: loaded from remote [quarcs-lab/indonesia514](https://github.com/quarcs-lab/indonesia514).
- Grid GDP: slim CSVs in `data/idn_grid/` (rescaled to Penn World Table 11.0, GADM 4.10 geography;
  source collection *Local GDP Estimates Around the World*, Rossi-Hansberg & Zhang).
