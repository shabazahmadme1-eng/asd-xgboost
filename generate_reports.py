"""
Batch report generator.

Runs every CSV in generated_csvs/ through the EXACT validated inference + clinical
logic used by the live API (imported from app.py), renders a standalone, printable
HTML clinical report for each VALID file, and bundles them into reports/asd_reports.zip.

CSVs whose format the model can't accept (wrong/missing columns, too few engineered
features, etc.) are detected, reported, and OMITTED from the zip.

Run from the project root:  python generate_reports.py
"""
import os
import sys
import glob
import html
import json
import zipfile
import datetime

import numpy as np
import pandas as pd

# --- Avoid importing the heavy MediaPipe/cv2 video module ---------------------
# app.py does `from services.pose_extraction import get_skeletal_data`, which pulls
# in mediapipe + cv2. We only need the CSV path, so stub that module before import.
import types
_stub = types.ModuleType("services.pose_extraction")
_stub.get_skeletal_data = lambda *a, **k: None  # never called on the CSV path
_stub.EXPECTED_COLS_3D = None
sys.modules["services.pose_extraction"] = _stub

import app as backend  # reuses model + all validated clinical helpers

CSV_DIR = "generated_csvs"
OUT_DIR = "reports"
ZIP_PATH = os.path.join(OUT_DIR, "asd_reports.zip")


def analyze_csv(path):
    """Mirror app.analyze_patient's CSV branch exactly. Returns a result dict, or
    raises ValueError (wrong format) / other Exception (processing failure)."""
    raw_df = pd.read_csv(path)

    raw_timestamps = []
    time_col = next((c for c in raw_df.columns if "H:M:S:MS" in str(c)), None)
    if time_col:
        raw_timestamps = raw_df[time_col].tolist()

    # Raises ValueError for unrecognized formats -> caught by caller as "wrong format".
    X_windows, window_timestamps, csv_kind, X_full = backend.prepare_csv_data(
        raw_df, return_full=True
    )
    source_kind = f"csv-{csv_kind}"

    if len(X_windows) == 0:
        raise ValueError("Failed to engineer spatiotemporal features (no usable windows).")

    X_matrix = np.asarray(X_windows, dtype=np.float32)
    if X_matrix.ndim == 1:
        X_matrix = X_matrix.reshape(1, -1)
    elif X_matrix.ndim == 3:
        X_matrix = X_matrix.reshape(X_matrix.shape[0], -1)

    if X_matrix.shape[1] != backend.model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: model expects {backend.model.n_features_in_}, "
            f"got {X_matrix.shape[1]}."
        )

    predictions = backend.model.predict_proba(X_matrix)[:, 1]
    n_windows = len(predictions)
    peak_risk = float(np.max(predictions))
    mean_risk = float(np.mean(predictions))
    atypical_count = int(np.sum(predictions >= backend.WINDOW_RISK_THRESHOLD))
    min_windows = max(backend.MIN_ATYPICAL_WINDOWS,
                      int(n_windows * backend.MIN_ATYPICAL_FRACTION))

    if n_windows < backend.MIN_ATYPICAL_WINDOWS:
        is_atypical = bool(mean_risk >= 0.50)
    else:
        is_atypical = bool(atypical_count >= min_windows)

    final_risk_score = peak_risk

    if raw_timestamps:
        step = max(1, len(raw_timestamps) // n_windows)
        window_times = raw_timestamps[::step][:n_windows]
        timeline = [{"timestamp": str(ts), "risk_score": float(p)}
                    for ts, p in zip(window_times, predictions)]
    else:
        timeline = [{"timestamp": f"{ts}s", "risk_score": float(p)}
                    for ts, p in zip(window_timestamps, predictions)]

    regions_pct = backend.regional_drivers(X_matrix)
    clinical_summary = backend.build_clinical_summary(
        is_atypical, predictions, atypical_count, min_windows,
        regions_pct, backend.WINDOW_RISK_THRESHOLD)

    flagged_regions = {r for r in backend.REGION_NAMES
                       if regions_pct.get(r, 0) >= 100.0 / len(backend.REGION_NAMES)}
    regional_breakdown = [{
        "name": r,
        "contribution_pct": round(regions_pct.get(r, 0.0), 1),
        "status": "Elevated" if (is_atypical and r in flagged_regions) else "Typical",
        "description": backend.REGION_DESCRIPTIONS[r],
    } for r in backend.REGION_NAMES]

    ood = backend.ood_score(X_matrix)
    warnings_list = []
    if ood is not None and backend.OOD_STATS and ood > backend.OOD_STATS[2] * 1.35:
        warnings_list.append("Input differs notably from the model's reference data; "
                             "interpret with caution.")
    if n_windows < 5:
        warnings_list.append(f"Short recording ({n_windows} analysis window(s)); "
                             "result is less stable.")

    ref = backend.pick_reference(source_kind)
    markers = backend.kinematic_markers(X_matrix)
    findings = backend.kinematic_findings(X_full, ref)
    symmetry = backend.symmetry_indices(X_full, ref)
    flagged_moments = [
        {"timestamp": window_timestamps[i] if i < len(window_timestamps) else i,
         "risk": round(float(predictions[i]), 3)}
        for i in range(n_windows) if predictions[i] >= backend.WINDOW_RISK_THRESHOLD
    ][:25]
    result_stability = {
        "window_risk_std": round(float(np.std(predictions)), 3),
        "consistency": ("high" if float(np.std(predictions)) < 0.12
                        else "moderate" if float(np.std(predictions)) < 0.22 else "low"),
    }

    return {
        "final_risk_score": final_risk_score,
        "is_atypical": is_atypical,
        "timeline": timeline,
        "clinical_summary": clinical_summary,
        "regional_breakdown": regional_breakdown,
        "kinematic_markers": markers,
        "kinematic_findings": findings,
        "symmetry": symmetry,
        "flagged_moments": flagged_moments,
        "result_stability": result_stability,
        "input_meta": {
            "kind": source_kind,
            "windows_analyzed": n_windows,
            "ood_score": round(ood, 2) if ood is not None else None,
            "ood_baseline": round(backend.OOD_STATS[2], 2) if backend.OOD_STATS else None,
        },
        "quality": {"reliable": len(warnings_list) == 0, "warnings": warnings_list},
        "disclaimer": backend.DISCLAIMER,
    }


# ----------------------------- HTML rendering --------------------------------
def _esc(x):
    return html.escape(str(x))


def _sparkline(timeline, threshold):
    """Inline SVG risk timeline."""
    if not timeline:
        return ""
    vals = [t["risk_score"] for t in timeline]
    w, h, pad = 760, 140, 8
    n = len(vals)
    def x(i): return pad + (w - 2 * pad) * (i / max(n - 1, 1))
    def y(v): return pad + (h - 2 * pad) * (1 - v)
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
    area = f"{pad},{y(0):.1f} " + pts + f" {x(n-1):.1f},{y(0):.1f}"
    thr_y = y(threshold)
    return f"""
    <svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="none" role="img" aria-label="Risk timeline">
      <polygon points="{area}" fill="#6366f1" fill-opacity="0.10"/>
      <line x1="{pad}" y1="{thr_y:.1f}" x2="{w-pad}" y2="{thr_y:.1f}" stroke="#f59e0b" stroke-width="1" stroke-dasharray="4 4"/>
      <polyline points="{pts}" fill="none" stroke="#6366f1" stroke-width="2.5"/>
    </svg>
    <div class="muted small">Per-window atypical risk over the recording · dashed line = {int(threshold*100)}% per-window threshold</div>
    """


def _bar(pct, color="#3b82f6"):
    pct = max(0.0, min(100.0, float(pct)))
    return (f'<div class="track"><div class="fill" style="width:{pct:.1f}%;'
            f'background:{color}"></div></div>')


def render_html(source_name, r, session_id, generated_at):
    cs = r["clinical_summary"]
    atypical = r["is_atypical"]
    banner_cls = "bad" if atypical else "good"
    verdict = "Atypical Kinematics Flagged" if atypical else "Typical Motor Development"

    # Metric cards
    cards = [
        ("Peak window risk", f"{cs['peak_risk']*100:.0f}%"),
        ("Mean window risk", f"{cs['mean_risk']*100:.0f}%"),
        ("Windows analyzed", f"{cs['total_windows']}"),
        ("Atypical windows", f"{cs['atypical_windows']} / {cs['total_windows']}"),
        ("Consistency", r["result_stability"]["consistency"].title()),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="card-label">{_esc(l)}</div>'
        f'<div class="card-value">{_esc(v)}</div></div>' for l, v in cards)

    # Regional breakdown
    regions_html = ""
    for reg in r["regional_breakdown"]:
        st = reg["status"]
        col = "#ef4444" if st == "Elevated" else "#3b82f6"
        regions_html += f"""
        <div class="region">
          <div class="region-head">
            <span class="region-name">{_esc(reg['name'])}</span>
            <span class="pill {'pill-bad' if st=='Elevated' else 'pill-ok'}">{_esc(st)}</span>
            <span class="region-pct">{reg['contribution_pct']:.0f}%</span>
          </div>
          {_bar(reg['contribution_pct'], col)}
          <div class="muted small">{_esc(reg['description'])}</div>
        </div>"""

    # Kinematic drivers
    if r["kinematic_markers"]:
        drivers_html = "".join(
            f'<div class="driver"><div class="driver-row"><span>{_esc(m["marker"])}</span>'
            f'<span class="muted">{m["contribution_pct"]:.0f}%</span></div>'
            f'{_bar(m["contribution_pct"], "#6366f1")}</div>'
            for m in r["kinematic_markers"])
    else:
        drivers_html = '<div class="muted small">No dominant kinematic drivers identified.</div>'

    # Findings table
    if r["kinematic_findings"]:
        rows = "".join(
            f"<tr><td>{_esc(f['marker'])}</td><td>{_esc(f['aspect'])}</td>"
            f"<td>{_esc(f['region'])}</td><td>{f['patient_value']}</td>"
            f"<td>{f['typical_value']}</td><td class='{'neg' if f['z_score']<0 else 'pos'}'>"
            f"{f['z_score']:+.1f}σ {_esc(f['direction'])}</td></tr>"
            for f in r["kinematic_findings"])
        findings_html = (f"<table><thead><tr><th>Metric</th><th>Aspect</th><th>Region</th>"
                         f"<th>Patient</th><th>Typical</th><th>Deviation</th></tr></thead>"
                         f"<tbody>{rows}</tbody></table>")
    else:
        findings_html = '<div class="muted small">No metric deviated notably from the typical baseline.</div>'

    # Symmetry
    if r["symmetry"]:
        sym_rows = "".join(
            f"<tr><td>{_esc(s['pair'])}</td><td>{s['coordination']}</td>"
            f"<td>{s['typical']}</td><td class='{'neg' if s['status']=='reduced' else ''}'>"
            f"{_esc(s['status'])}</td></tr>" for s in r["symmetry"])
        symmetry_html = (f"<table><thead><tr><th>Limb pair</th><th>Coordination</th>"
                         f"<th>Typical</th><th>Status</th></tr></thead><tbody>{sym_rows}</tbody></table>")
    else:
        symmetry_html = '<div class="muted small">Symmetry baseline unavailable.</div>'

    # Warnings
    warns = r["quality"]["warnings"]
    warn_html = ""
    if warns:
        items = "".join(f"<li>{_esc(w)}</li>" for w in warns)
        warn_html = f'<div class="warn"><strong>Quality notes</strong><ul>{items}</ul></div>'

    meta = r["input_meta"]
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Kinematic Report — {_esc(source_name)}</title>
<style>
  :root {{ --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --bg:#f8fafc; --blue:#2563eb; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  .wrap {{ max-width:880px; margin:0 auto; padding:32px 24px 64px; }}
  .topbar {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:24px; }}
  .btn {{ display:inline-flex; align-items:center; gap:8px; height:42px; padding:0 18px;
          border-radius:12px; border:1px solid var(--blue); background:var(--blue); color:#fff;
          font-weight:600; cursor:pointer; }}
  h1 {{ font-size:28px; margin:0; letter-spacing:-.02em; }}
  .sub {{ color:var(--muted); margin-top:4px; }}
  .banner {{ display:flex; align-items:center; gap:16px; padding:22px 26px; border-radius:18px;
             color:#fff; margin:22px 0; }}
  .banner.good {{ background:#0f766e; }}
  .banner.bad  {{ background:#9f1239; }}
  .banner .tag {{ font-size:11px; text-transform:uppercase; letter-spacing:.12em; opacity:.8; }}
  .banner h2 {{ margin:2px 0 0; font-size:22px; }}
  .banner .sev {{ margin-left:auto; text-align:right; font-size:13px; opacity:.9; }}
  .grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin:20px 0; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:14px; }}
  .card-label {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }}
  .card-value {{ font-size:22px; font-weight:700; margin-top:6px; }}
  section {{ background:#fff; border:1px solid var(--line); border-radius:18px;
             padding:22px 24px; margin:16px 0; }}
  section h3 {{ margin:0 0 14px; font-size:16px; }}
  .muted {{ color:var(--muted); }}
  .small {{ font-size:12.5px; }}
  .track {{ height:8px; background:#eef2f7; border-radius:99px; overflow:hidden; margin:6px 0; }}
  .fill {{ height:100%; border-radius:99px; }}
  .region {{ margin-bottom:16px; }}
  .region-head {{ display:flex; align-items:center; gap:10px; }}
  .region-name {{ font-weight:600; }}
  .region-pct {{ margin-left:auto; font-weight:700; }}
  .pill {{ font-size:11px; padding:2px 9px; border-radius:99px; font-weight:600; }}
  .pill-ok {{ background:#e0f2fe; color:#0369a1; }}
  .pill-bad {{ background:#fee2e2; color:#b91c1c; }}
  .driver {{ margin-bottom:12px; }}
  .driver-row {{ display:flex; justify-content:space-between; font-size:14px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  td.pos {{ color:#b45309; }} td.neg {{ color:#1d4ed8; }}
  .narrative {{ font-size:15px; line-height:1.7; }}
  .rule {{ background:var(--bg); border:1px dashed var(--line); border-radius:12px;
           padding:12px 14px; margin-top:12px; font-size:13px; color:var(--muted); }}
  .warn {{ background:#fffbeb; border:1px solid #fde68a; border-radius:12px; padding:14px 16px; }}
  .warn ul {{ margin:8px 0 0; padding-left:18px; }}
  .disclaimer {{ font-size:12px; color:var(--muted); margin-top:24px; border-top:1px solid var(--line); padding-top:16px; }}
  @media print {{
    body {{ background:#fff; }}
    .topbar .btn {{ display:none; }}
    section, .card {{ box-shadow:none; }}
    .wrap {{ padding:0; max-width:100%; }}
  }}
</style></head>
<body><div class="wrap">

  <div class="topbar">
    <div>
      <h1>Clinical Kinematic Screening Report</h1>
      <div class="sub">Source: {_esc(source_name)} · Session {_esc(session_id)} · Generated {_esc(generated_at)}</div>
    </div>
    <button class="btn" onclick="window.print()">Save as PDF</button>
  </div>

  <div class="banner {banner_cls}">
    <div>
      <div class="tag">Diagnostic Classification</div>
      <h2>{_esc(verdict)}</h2>
    </div>
    <div class="sev">
      <div class="tag">Severity</div>
      <div>{_esc(cs['severity'])}</div>
    </div>
  </div>

  <div class="grid">{cards_html}</div>

  <section>
    <h3>Clinical Summary</h3>
    <div class="narrative">{_esc(cs['narrative'])}</div>
    <div class="rule">{_esc(cs['decision_rule'])}</div>
  </section>

  <section>
    <h3>Risk Timeline</h3>
    {_sparkline(r['timeline'], backend.WINDOW_RISK_THRESHOLD)}
  </section>

  <section>
    <h3>Regional Breakdown</h3>
    {regions_html}
  </section>

  <section>
    <h3>Primary Kinematic Drivers</h3>
    {drivers_html}
  </section>

  <section>
    <h3>Kinematic Findings vs Typical Population</h3>
    {findings_html}
  </section>

  <section>
    <h3>Bilateral Symmetry</h3>
    {symmetry_html}
  </section>

  {warn_html}

  <div class="disclaimer">
    Analysis windows: {meta['windows_analyzed']} · Source kind: {_esc(meta['kind'])}
    {(' · OOD ' + str(meta['ood_score']) + ' (baseline ' + str(meta['ood_baseline']) + ')') if meta.get('ood_score') is not None else ''}
    <br/>{_esc(r['disclaimer'])}
  </div>

</div></body></html>"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    csvs = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    if not csvs:
        print(f"No CSVs found in {CSV_DIR}/")
        return

    generated_at = datetime.datetime.now().strftime("%b %d, %Y · %H:%M")
    valid, invalid = [], []

    for path in csvs:
        name = os.path.basename(path)
        try:
            result = analyze_csv(path)
        except ValueError as e:
            invalid.append((name, str(e)))
            print(f"  ⛔ WRONG FORMAT — {name}: {e}")
            continue
        except Exception as e:
            invalid.append((name, f"processing error: {e}"))
            print(f"  ❌ FAILED — {name}: {e}")
            continue

        session_id = f"{os.path.splitext(name)[0][:6].upper()}-{abs(hash(name)) % 100000:05d}"
        out_name = os.path.splitext(name)[0] + "_report.html"
        out_path = os.path.join(OUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(render_html(name, result, session_id, generated_at))
        valid.append((name, out_name, result))
        verdict = "ATYPICAL" if result["is_atypical"] else "typical"
        print(f"  ✅ {name}: {verdict} "
              f"(peak {result['clinical_summary']['peak_risk']*100:.0f}%) -> {out_name}")

    # Machine-readable summary alongside the reports
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "generated_at": generated_at,
            "valid": [{"csv": n, "report": o,
                       "verdict": "atypical" if r["is_atypical"] else "typical",
                       "peak_risk": r["clinical_summary"]["peak_risk"]}
                      for n, o, r in valid],
            "omitted_wrong_format": [{"csv": n, "reason": why} for n, why in invalid],
        }, fh, indent=2)

    # Zip ONLY the valid reports (+ summary); omit wrong-format CSVs entirely.
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, out_name, _ in valid:
            zf.write(os.path.join(OUT_DIR, out_name), out_name)
        zf.write(os.path.join(OUT_DIR, "summary.json"), "summary.json")

    print("\n" + "=" * 60)
    print(f"Reports generated : {len(valid)}")
    print(f"Omitted (bad fmt) : {len(invalid)}")
    if invalid:
        for n, why in invalid:
            print(f"   - {n}: {why}")
    print(f"Zip ready         : {ZIP_PATH}")


if __name__ == "__main__":
    main()
