"""Capture screenshots of every page of the running app (used for the report).
Usage: streamlit run app.py   (in another terminal)
       python tools/screenshots.py out_dir
"""
import sys
import time

from playwright.sync_api import sync_playwright

OUT = sys.argv[1] if len(sys.argv) > 1 else "screenshots"
URL = "http://localhost:8502"


def wait_idle(page, extra=1.5):
    time.sleep(1.0)
    for _ in range(240):
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        time.sleep(0.5)
    time.sleep(extra)


def errors(page):
    exc = page.locator('[data-testid="stException"]')
    return [exc.nth(i).inner_text()[:500] for i in range(exc.count())]


def go(page, label):
    page.locator('[data-testid="stSidebar"] label').filter(has_text=label).first.click()
    wait_idle(page)


def shot(page, name, full=True):
    # Streamlit scrolls inside its main container, so grow the viewport to fit it
    h = page.evaluate("""() => {
        const m = document.querySelector('[data-testid="stMain"]') || document.querySelector('section.main');
        return m ? m.scrollHeight : document.body.scrollHeight; }""")
    page.set_viewport_size({"width": 1500, "height": min(int(h) + 40, 12000)})
    time.sleep(2.5)
    page.screenshot(path=f"{OUT}/{name}.png", full_page=full)
    page.set_viewport_size({"width": 1500, "height": 1000})
    time.sleep(0.8)
    errs = errors(page)
    print(name, "ERRORS:" if errs else "ok", errs)


def tab(page, label):
    page.get_by_role("tab", name=label).click()
    wait_idle(page, 1.0)


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1)
    page.goto(URL)
    wait_idle(page, 4)
    shot(page, "01_dashboard")
    go(page, "Data ingestion"); shot(page, "02_ingestion")
    go(page, "Preprocessing"); shot(page, "03_preprocessing")
    go(page, "Text mining hub")
    shot(page, "04_classification")
    tab(page, "🧩 K-Means clustering"); shot(page, "05_clustering")
    tab(page, "⚖️ Evaluation & comparison"); shot(page, "06_evaluation")
    tab(page, "♊ Duplicates"); shot(page, "07_duplicates")
    go(page, "Recommendations"); shot(page, "08_recommendation")
    page.get_by_text("Enter free text").click(); wait_idle(page); shot(page, "09_recommendation_query")
    go(page, "Performance analytics")
    page.get_by_role("button", name="Run preprocessing experiment").click(); wait_idle(page, 3)
    shot(page, "10_analytics_prep")
    tab(page, "Duplicates"); page.get_by_role("button", name="Run duplicate experiments").click(); wait_idle(page, 3)
    shot(page, "11_analytics_dup")
    tab(page, "Choice of K"); page.get_by_role("button", name="Run K sweep").click(); wait_idle(page, 3)
    shot(page, "12_analytics_k")
    b.close()
