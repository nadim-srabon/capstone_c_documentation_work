"""Screenshots of the running app for the report (needs `streamlit run app.py` on BASE and Playwright).

    .venv/Scripts/python scripts/make_app_screenshots.py <assets_dir> <sample_video> [base_url]

Dataset signers are blurred (CSS filter) before the recognition screenshots are taken.
"""
import csv
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1]) / "C_app_screenshots"
VIDEO = sys.argv[2]
BASE = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8599"
OUT.mkdir(parents=True, exist_ok=True)
BLUR = "video, [data-testid='stVideo'], [data-testid='stImage'] img { filter: blur(14px) !important; }"
rows = []


def ready(pg, sel="[data-testid=stTopNavLink]"):
    pg.wait_for_selector(sel, timeout=180000)
    pg.wait_for_timeout(2500)
    pg.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=180000)


def shot(pg, fname, title, caption):
    pg.screenshot(path=str(OUT / f"{fname}.png"))
    rows.append({"file": f"C_app_screenshots/{fname}.png", "pdf_vector": "no", "title": title, "caption": caption,
                 "source": "Screenshot of the deployed Streamlit application (1440x900)",
                 "contains_people": "no (signer blurred)" if "blurred" in caption else "no"})
    print("shot", fname)


def scroll_to(pg, selector, nth=0):
    pg.locator(selector).nth(nth).scroll_into_view_if_needed()
    pg.wait_for_timeout(900)


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 900})

    pg.goto(BASE + "/"); ready(pg, ".home-hero")
    shot(pg, "app01_home_hero", "Web app: home page",
         "Home page with the thesis title, team, calls to action and key results.")
    for i, (sel, t) in enumerate([(".sec-head", "objectives"), (".sec-head", "architectures"), (".sec-head", "findings")]):
        scroll_to(pg, sel, i + 1)
        shot(pg, f"app0{2 + i}_home_{t}", f"Web app: home page, {t}", f"Home page section: {t}.")

    pg.goto(BASE + "/recognition"); ready(pg, ".figcaption")
    shot(pg, "app05_recognize_empty", "Web app: recognition page",
         "Recognition workspace before an upload: Source and Result panels with video, model and option controls.")
    pg.set_input_files('[data-testid="stFileUploaderDropzoneInput"]', VIDEO)
    pg.wait_for_timeout(2500)
    pg.get_by_role("button", name="Recognize sign").click()
    pg.wait_for_selector(".result .sign", timeout=300000)
    ready(pg, ".result .sign")
    pg.add_style_tag(content=BLUR)
    pg.wait_for_timeout(500)
    sign = pg.inner_text(".result .sign")
    shot(pg, "app06_recognize_result", "Web app: recognition result",
         f"Prediction for an ASL Citizen test clip of the sign {sign} with Video Swin-T (signer blurred for privacy).")
    pg.locator("[data-testid=stPlotlyChart]").first.scroll_into_view_if_needed()
    pg.wait_for_selector("[data-testid=stPlotlyChart] .main-svg .bars, [data-testid=stPlotlyChart] g.trace",
                         timeout=60000)
    pg.wait_for_timeout(2000)
    shot(pg, "app07_recognize_top5", "Web app: top-5 predictions",
         "Top-5 predicted signs with softmax confidence for the uploaded clip (signer blurred).")
    pg.get_by_text("What the model sees").scroll_into_view_if_needed()
    pg.mouse.wheel(0, 250)
    pg.wait_for_timeout(1500)
    shot(pg, "app07b_recognize_model_frames", "Web app: frames seen by the model",
         "The 16 sampled 112x112 signer crops shown to the RGB models (signer blurred).")

    pg.goto(BASE + "/comparison"); ready(pg, "[data-testid=stDataFrame]")
    shot(pg, "app08_results_table", "Web app: results page", "Model comparison page: key metrics and the 1,000-sign table.")
    for i, sel in enumerate(["[data-testid=stPlotlyChart]", "[data-testid=stPlotlyChart]"]):
        scroll_to(pg, sel, [0, 2][i])
        pg.wait_for_timeout(1500)
        shot(pg, f"app{9 + i:02d}_results_charts_{i + 1}", f"Web app: results charts {i + 1}",
             "Interactive Plotly charts on the results page.")
    pg.goto(BASE + "/methodology"); ready(pg, ".flow")
    shot(pg, "app11_methodology", "Web app: methodology page", "Methodology page with the processing pipeline.")
    pg.goto(BASE + "/about"); ready(pg, ".page-title")
    shot(pg, "app12_about", "Web app: about page", "About page with the thesis title, background and dataset licence.")
    b.close()

idx = Path(sys.argv[1]) / "figure_index.csv"
with open(idx, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writerows(rows)
print(len(rows), "screenshots indexed")
