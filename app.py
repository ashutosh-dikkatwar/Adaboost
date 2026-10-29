import os
import pickle
import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Load model once (cold start) -- file must sit next to app.py
# ----------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaboost.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Exact feature order the model was trained with
FEATURES = list(model.feature_names_in_)

# ----------------------------------------------------------------------------
# Text shown for each predicted class. Change if your labels are reversed.
# ----------------------------------------------------------------------------
CLASS_LABELS = {0: "Not Admitted", 1: "Admitted"}

# These two features exist in the model but have ZERO influence on its
# predictions (feature importance = 0), so they are not shown in the form.
# A neutral default is sent instead.
HIDDEN_DEFAULTS = {"gender": 0, "state": 0}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Admission Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#070b17; --card:rgba(255,255,255,.05); --border:rgba(255,255,255,.10);
    --text:#eef1fb; --muted:#9aa3bf; --accent:#7c5cff; --accent2:#22d3ee;
    --yes:#22c55e; --no:#ef4444;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Inter',system-ui,sans-serif;color:var(--text);min-height:100vh;
    background:
      radial-gradient(900px 500px at 10% -10%,rgba(124,92,255,.35),transparent 60%),
      radial-gradient(800px 500px at 100% 0%,rgba(34,211,238,.22),transparent 60%),
      var(--bg);
    padding:40px 18px 70px;
  }
  .wrap{max-width:1080px;margin:0 auto}
  header{text-align:center;margin-bottom:34px}
  .badge{display:inline-block;padding:6px 14px;border-radius:999px;font-size:12px;font-weight:600;
    letter-spacing:.08em;text-transform:uppercase;color:var(--accent2);
    background:rgba(34,211,238,.10);border:1px solid rgba(34,211,238,.25)}
  h1{font-size:clamp(30px,5vw,48px);font-weight:800;margin:16px 0 10px;letter-spacing:-.02em;
    background:linear-gradient(90deg,#fff,#b9a8ff 50%,#67e8f9);-webkit-background-clip:text;background-clip:text;color:transparent}
  header p{color:var(--muted);max-width:620px;margin:0 auto;line-height:1.6}

  .grid{display:grid;grid-template-columns:1fr;gap:20px}
  @media(min-width:900px){.grid{grid-template-columns:1.6fr 1fr;align-items:start}}

  .card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:24px;
    backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px)}
  .section{margin-bottom:26px}
  .section h3{font-size:13px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#c8bfff;
    margin-bottom:14px;display:flex;align-items:center;gap:8px}
  .section h3 span{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;font-size:13px;
    background:linear-gradient(135deg,var(--accent),var(--accent2))}
  .fields{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px}
  label{display:block;font-size:12.5px;color:var(--muted);margin-bottom:6px;font-weight:500}
  input{
    width:100%;padding:12px 14px;border-radius:12px;font:inherit;font-size:14.5px;color:var(--text);
    background:rgba(8,12,26,.7);border:1px solid var(--border);outline:none;transition:.2s;
  }
  input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(124,92,255,.25)}

  details{border:1px solid var(--border);border-radius:14px;padding:14px 16px;background:rgba(255,255,255,.03)}
  summary{cursor:pointer;font-weight:600;font-size:13.5px;color:#c8bfff;list-style:none;display:flex;justify-content:space-between}
  summary::after{content:"+";font-size:18px;line-height:1}
  details[open] summary::after{content:"–"}
  details .fields{margin-top:16px}

  button{
    width:100%;margin-top:22px;padding:15px;border:0;border-radius:14px;cursor:pointer;
    font:inherit;font-weight:700;font-size:15.5px;color:#fff;letter-spacing:.02em;
    background:linear-gradient(135deg,var(--accent),#4f8cff 55%,var(--accent2));
    box-shadow:0 10px 30px -10px rgba(124,92,255,.8);transition:.2s;
  }
  button:hover{transform:translateY(-2px);filter:brightness(1.08)}
  button:disabled{opacity:.6;cursor:wait;transform:none}

  .result{position:sticky;top:20px}
  .result h2{font-size:15px;color:var(--muted);font-weight:600;margin-bottom:16px}
  .placeholder{text-align:center;padding:36px 10px;color:var(--muted);line-height:1.6;font-size:14px}
  .placeholder .ico{font-size:42px;margin-bottom:10px}
  .verdict{text-align:center;padding:22px 12px;border-radius:16px;margin-bottom:20px;border:1px solid var(--border)}
  .verdict small{display:block;color:var(--muted);font-size:12px;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px}
  .verdict strong{font-size:30px;font-weight:800;letter-spacing:-.01em}
  .verdict .conf{margin-top:6px;color:var(--muted);font-size:13px}
  .yes{background:rgba(34,197,94,.12);border-color:rgba(34,197,94,.4)} .yes strong{color:var(--yes)}
  .no{background:rgba(239,68,68,.12);border-color:rgba(239,68,68,.4)} .no strong{color:var(--no)}

  .bar{margin-bottom:14px}
  .bar .row{display:flex;justify-content:space-between;font-size:13.5px;margin-bottom:6px;font-weight:500}
  .track{height:10px;border-radius:99px;background:rgba(255,255,255,.08);overflow:hidden}
  .fill{height:100%;width:0;border-radius:99px;transition:width .9s cubic-bezier(.2,.8,.2,1)}
  .err{color:#fca5a5;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);padding:12px;border-radius:12px;font-size:13.5px}
  .spin{display:inline-block;width:16px;height:16px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;
    border-radius:50%;animation:s .7s linear infinite;vertical-align:-3px;margin-right:8px}
  @keyframes s{to{transform:rotate(360deg)}}
  footer{text-align:center;color:var(--muted);font-size:12.5px;margin-top:34px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="badge">AdaBoost · ML Model</div>
    <h1>Admission Predictor</h1>
    <p>Enter an applicant's academic record and evaluation scores to estimate the chance of admission.</p>
  </header>

  <div class="grid">
    <form id="form" class="card" autocomplete="off">

      <div class="section">
        <h3><span>🎓</span> Academics</h3>
        <div class="fields">
          <div><label>High school GPA (0–4)</label><input type="number" name="high_school_gpa" value="3.4" min="0" max="4" step="0.01" required></div>
          <div><label>SAT score (400–1600)</label><input type="number" name="sat_score" value="1150" min="400" max="1600" step="1" required></div>
          <div><label>ACT score (1–36)</label><input type="number" name="act_score" value="25" min="1" max="36" step="1" required></div>
          <div><label>Attendance rate (%)</label><input type="number" name="attendance_rate" value="92" min="0" max="100" step="any" required></div>
          <div><label>AP courses taken</label><input type="number" name="ap_courses" value="3" min="0" step="1" required></div>
        </div>
      </div>

      <div class="section">
        <h3><span>📝</span> Evaluation Scores (0–100)</h3>
        <div class="fields">
          <div><label>Essay score</label><input type="number" name="essay_score" value="75" min="0" max="100" step="any" required></div>
          <div><label>Recommendation score</label><input type="number" name="recommendation_score" value="78" min="0" max="100" step="any" required></div>
          <div><label>Interview score</label><input type="number" name="interview_score" value="73" min="0" max="100" step="any" required></div>
        </div>
      </div>

      <div class="section">
        <details>
          <summary>Additional profile details</summary>
          <div class="fields">
            <div><label>Age</label><input type="number" name="age" value="18" min="14" max="40" step="1" required></div>
            <div><label>Family income ($/year)</label><input type="number" name="family_income" value="60000" min="0" step="any" required></div>
            <div><label>Extracurricular activities</label><input type="number" name="extracurricular_count" value="3" min="0" step="1" required></div>
            <div><label>Volunteer hours</label><input type="number" name="volunteer_hours" value="50" min="0" step="any" required></div>
            <div><label>Leadership positions</label><input type="number" name="leadership_positions" value="1" min="0" step="1" required></div>
            <div><label>Coding projects</label><input type="number" name="coding_projects" value="2" min="0" step="1" required></div>
            <div><label>Social media (hours/day)</label><input type="number" name="social_media_hours" value="3" min="0" max="24" step="any" required></div>
            <div><label>Online certifications</label><input type="number" name="online_certifications" value="2" min="0" step="1" required></div>
          </div>
        </details>
      </div>

      <button type="submit" id="btn">Predict Admission</button>
    </form>

    <aside class="card result" id="result">
      <h2>Prediction result</h2>
      <div class="placeholder"><div class="ico">🎯</div>Fill in the form and hit <b>Predict Admission</b> to see the model's estimate.</div>
    </aside>
  </div>

  <footer>Built with Flask · scikit-learn · Deployed on Vercel</footer>
</div>

<script>
const CLS    = {"Admitted":"yes","Not Admitted":"no"};
const COLORS = {"Admitted":"var(--yes)","Not Admitted":"var(--no)"};
const ICONS  = {"Admitted":"✅","Not Admitted":"❌"};

document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn');
  const box = document.getElementById('result');
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span>Predicting...';

  const data = Object.fromEntries(new FormData(e.target).entries());
  try {
    const res = await fetch('/predict', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)
    });
    const out = await res.json();
    if(!res.ok) throw new Error(out.error || 'Prediction failed');

    const top = Object.entries(out.probabilities).sort((a,b)=>b[1]-a[1]);
    box.innerHTML = `
      <h2>Prediction result</h2>
      <div class="verdict ${CLS[out.prediction]||''}">
        <small>Predicted outcome</small>
        <strong>${ICONS[out.prediction]||''} ${out.prediction}</strong>
        <div class="conf">Confidence: ${(top[0][1]*100).toFixed(1)}%</div>
      </div>
      ${top.map(([k,v]) => `
        <div class="bar">
          <div class="row"><span>${k}</span><span>${(v*100).toFixed(1)}%</span></div>
          <div class="track"><div class="fill" data-w="${(v*100).toFixed(1)}" style="background:${COLORS[k]||'var(--accent)'}"></div></div>
        </div>`).join('')}`;
    requestAnimationFrame(()=>requestAnimationFrame(()=>{
      box.querySelectorAll('.fill').forEach(f => f.style.width = f.dataset.w + '%');
    }));
    if (window.innerWidth < 900) box.scrollIntoView({behavior:'smooth'});
  } catch(err){
    box.innerHTML = `<h2>Prediction result</h2><div class="err">⚠️ ${err.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = 'Predict Admission';
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        payload = {**HIDDEN_DEFAULTS, **payload}
        row = [float(payload[name]) for name in FEATURES]
        X = np.array([row], dtype=float)

        pred = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]

        return jsonify({
            "prediction": CLASS_LABELS.get(pred, str(pred)),
            "probabilities": {
                CLASS_LABELS.get(int(c), str(c)): float(p)
                for c, p in zip(model.classes_, probs)
            },
        })
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# Local run: python app.py
if __name__ == "__main__":
    app.run(debug=True)