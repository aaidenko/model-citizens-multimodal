from html import escape
from datetime import datetime
import json
from zoneinfo import ZoneInfo


def render(root, data):
    counts = data['counts']
    def probability(p):
        if p < .001:
            return '&lt;0.1%'
        if p > .999:
            return '&gt;99.9%'
        return f'{p:.1%}'
    method_names = {
        'polls_plus_fundamentals':'Polls + historical baseline',
        'polls_with_neutral_independent_prior':'Independent race; extra review',
        'market_proxy':'Market proxy; no recent poll model',
        'unavailable':'Needs source check',
    }
    rows = []
    for r in data['races']:
        chosen = next((o for o in r['outcomes'] if o['selected']),None)
        favorite = max(r['outcomes'],key=lambda o:o['q']) if r['outcomes'] else None
        sources = f'<a href="{escape(r["market_url"],quote=True)}">Market source</a>'
        if r['poll_url']:
            sources += f' · <a href="{escape(r["poll_url"],quote=True)}">Poll source</a>'
        detail = ''.join(f'<li>{escape(issue.replace("_"," "))}</li>' for issue in r['issues'])
        outcome_rows = ''.join('<tr>'+''.join(f'<td>{cell}</td>' for cell in [
            escape(o['candidate']),escape(o['party']),probability(o['benchmark_proxy']),
            probability(o['q']),f'{o["expected_points"]:.3f}',
        ])+'</tr>' for o in r['outcomes'])
        name = escape(chosen['candidate']) if chosen else 'Needs market mapping'
        poll = r['poll_summary']
        coverage = f'{poll["pollsters"]} pollsters · latest {poll["latest_date"]}' if poll else 'Fundamentals / source review'
        if chosen:
            value = probability(chosen['q'])
            price = probability(chosen['benchmark_proxy'])
            points = f'{chosen["expected_points"]:.3f}'
        else:
            value=price=points='—'
        rows.append(f'''<tr class="race" data-state="{r['state']}">
          <td><strong>{r['state']}</strong><br><span class="muted">{r['state_name']}</span></td>
          <td>{name}<br><span class="muted">{escape(method_names[r['model_method']])}</span></td>
          <td>{price}</td><td>{value}</td><td>{points}</td>
          <td><details><summary>Sources &amp; checks</summary><p>{coverage}</p>
          <p>{sources}</p><p>Model favorite: {escape(favorite['candidate']) if favorite else 'unavailable'}</p>
          <div class="scroll"><table><thead><tr><th>Outcome</th><th>Party</th><th>Market proxy</th><th>Forecast</th><th>Expected points</th></tr></thead>
          <tbody>{outcome_rows}</tbody></table></div><ul>{detail or '<li>No race-specific source exception found.</li>'}</ul>
          <p class="muted">All probabilities and strategies remain exploratory. Benchmark prices are not official contest weights.</p>
          </details></td></tr>''')
    validation = data['validation']
    blockers = ''.join(f'<li>{escape(x)}</li>' for x in data['blockers'])
    sim = data['scenario_comparison']
    simrows = []
    for name in dict.fromkeys(x['policy'] for x in sim['results']):
        rr = [x for x in sim['results'] if x['policy']==name]
        simrows.append('<tr><td>'+escape(name)+'</td>'+''.join(
            f'<td>{x["first_place_share"]:.1%}</td>' for x in rr)+f'<td>{rr[0]["mean_points"]:.2f}</td></tr>')
    evidence = data['document_evidence']
    source_rows=[]
    records = json.loads((root/'data/sources.json').read_text())
    latest = {r['id']:r for r in records if datetime.fromisoformat(r['retrieved_at']).astimezone(
        ZoneInfo('America/Los_Angeles')).date().isoformat() <= data['as_of']}
    for source_id,r in sorted(latest.items()):
        cached = (f' · <a href="../{escape(r["path"],quote=True)}">Numeric snapshot</a>'
                  if r['path'].endswith(('.json', '.csv')) else ' · publisher original omitted from Git')
        source_rows.append(f'<tr><td>{escape(source_id)}</td><td><a href="{escape(r["url"],quote=True)}">Original</a>{cached}</td><td>{escape(r["retrieved_at"])}</td><td><code>{r["sha256"][:16]}…</code></td></tr>')
    html = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Citizens · Senate research desk</title>
<style>
:root{color-scheme:light dark;--bg:light-dark(#f5f5f0,#14191a);--paper:light-dark(#fff,#1c2223);--ink:light-dark(#182d2b,#e6efeb);--soft:light-dark(#566966,#a6bab5);--line:light-dark(#d9e1dc,#354744);--accent:light-dark(#096c5b,#7fdbbe);--warn:light-dark(#fff0cd,#493c21)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,sans-serif}main{max-width:1180px;margin:auto;padding:36px 28px 70px}a{color:var(--accent);text-underline-offset:3px}header{border-bottom:1px solid var(--line);padding-bottom:28px}.eyebrow{color:var(--accent);font-weight:650;font-size:13px;letter-spacing:.12em;text-transform:uppercase}h1{font-size:clamp(32px,5vw,52px);line-height:1.1;letter-spacing:-.04em;margin:12px 0}h2{margin:0 0 12px;font-size:25px;letter-spacing:-.02em}h3{font-size:18px}.muted{color:var(--soft);font-size:13px}.sub{max-width:760px;color:var(--soft)}.badge{display:inline-block;background:var(--warn);padding:5px 11px;border-radius:6px;font-size:13px;font-weight:600}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}.stat{background:var(--paper);padding:18px 22px;border:1px solid var(--line);border-radius:10px}.stat strong{display:block;font-size:31px;font-variant-numeric:tabular-nums}.stat span{color:var(--soft);font-size:13px}section{margin-top:36px}.notice{background:var(--warn);padding:16px 20px;border-radius:8px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;text-align:left}th{font-weight:600;color:var(--soft);font-size:12px;text-transform:uppercase;letter-spacing:.025em}td,th{padding:13px 12px;border-bottom:1px solid var(--line);vertical-align:top}td:not(:first-child){font-variant-numeric:tabular-nums}details{max-width:440px}summary{color:var(--accent);cursor:pointer}details p{font-size:13px}details table{font-size:12px}ul{padding-left:22px}li{margin:5px 0}input[type=search]{font:inherit;background:var(--paper);border:1px solid var(--line);border-radius:7px;padding:10px 14px;color:var(--ink);width:100%;max-width:360px}input[type=range]{width:100%;accent-color:var(--accent)}.controls{display:grid;grid-template-columns:1fr 1fr;gap:24px;max-width:700px}.controls label{display:block;font-size:14px}.barrow{display:grid;grid-template-columns:130px 1fr 70px;align-items:center;gap:12px;margin:14px 0}.track{height:22px;background:var(--line);border-radius:4px;overflow:hidden}.bar{height:100%;background:var(--accent);min-width:0}.bar.secondary{opacity:.5}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}.pair img{width:100%;display:block;border:1px solid var(--line)}figcaption{font-size:13px;color:var(--soft);margin-top:8px}figure{margin:0}code{font-size:12px}footer{margin-top:48px;padding-top:20px;border-top:1px solid var(--line);color:var(--soft);font-size:13px}button{font:inherit}details ul{font-size:13px}@media(max-width:740px){main{padding:24px 16px}.stats{grid-template-columns:1fr 1fr;gap:10px}.stat{padding:14px}.pair,.controls{grid-template-columns:1fr}.barrow{grid-template-columns:100px 1fr 60px}td,th{padding:10px 8px}.race td{min-width:88px}.race td:nth-child(2){min-width:180px}}
</style></head><body><main>
<header><div class="eyebrow">Model Citizens / Senate research desk</div>
<h1>Forecast the race.<br>Understand the reward.</h1>
<p class="sub">A public-data forecasting workspace with an explicit scoring decision, traceable evidence, and a separate record of what still needs verification.</p>
<span class="badge">Research draft · AS_OF · $0 in new services</span></header>
<div class="stats"><div class="stat"><strong>RACE_COUNT</strong><span>races tracked</span></div><div class="stat"><strong>MARKET_COUNT</strong><span>verified market mappings</span></div><div class="stat"><strong>POLL_COUNT</strong><span>public polling feeds</span></div><div class="stat"><strong>DOCUMENT_COUNT</strong><span>visually reviewed poll PDF</span></div></div>
<div class="notice"><strong>Not ready to submit.</strong> Current market midpoints are research proxies. The official November 1 benchmark is still required. Forecast probabilities have not completed modern-cycle calibration.</div>
<section><h2>Why a long shot can be the higher-scoring pick</h2><p class="sub">Move the estimates to see the expected competition points for either pick. This comparison is for one hypothetical two-outcome race.</p>
<div class="controls"><label>Kalshi underdog probability: <strong id="p-label">10%</strong><input id="p" type="range" min="1" max="49" value="10" aria-label="Kalshi underdog probability"></label><label>Your underdog probability: <strong id="q-label">10%</strong><input id="q" type="range" min="0" max="100" value="10" aria-label="Your underdog probability"></label></div>
<div role="img" aria-label="Expected points for underdog and favorite"><div class="barrow"><span>Underdog</span><div class="track"><div id="dog-bar" class="bar"></div></div><strong id="dog-value"></strong></div><div class="barrow"><span>Favorite</span><div class="track"><div id="fav-bar" class="bar secondary"></div></div><strong id="fav-value"></strong></div></div><p id="decision" aria-live="polite"></p></section>
<section><h2>Race-by-race research picks</h2><p class="sub">These exploratory picks maximize expected points under the current model. They are not a finished contest entry. A selected candidate can have less than a 50% chance of winning.</p>
<input id="filter" type="search" aria-label="Filter races" placeholder="Search a state or candidate">
<div class="scroll"><table><thead><tr><th>Race</th><th>Research pick</th><th>Market proxy</th><th>Forecast chance</th><th>Expected points</th><th>Evidence</th></tr></thead><tbody>RACE_ROWS</tbody></table></div></section>
<section><h2>A useful multimodal starting point</h2><p>The University of Houston's Texas poll gives the same 48 / 46 / 3 / 3 figures in its chart and overall table. The original methodology says October 10–14, while the university-hosted tracker lists October 10–15. The original report controls the field dates.</p><p class="sub">We count the topline poll once. The chart is a verification surface, not a second independent observation. Subgroup and visual features are held out of the predictive model until historical testing supports their use.</p>
<div class="pair"><figure><a href="../data/evidence/hobby-06.png"><img src="../data/evidence/hobby-06.png" alt="Original poll chart: Talarico 48, Paxton 46, Brown 3, unsure 3" loading="lazy"></a><figcaption>Original PDF page 6: chart, visually checked for the recorded source hash.</figcaption></figure><figure><a href="../data/evidence/hobby-07.png"><img src="../data/evidence/hobby-07.png" alt="Original poll table with the matching overall row and subgroup rows" loading="lazy"></a><figcaption>Original PDF page 7: overall row and subgroup table.</figcaption></figure></div>
<p><a href="https://www.uh.edu/hobby/texas2026/statewide26.pdf">Original poll PDF</a> · <a href="https://texaspolitics.utexas.edu/blog/texas-2026-u-s-senate-poll-tracker">University-hosted tracker</a></p></section>
<section><h2>What the model has actually been tested on</h2><p>The simple fundamentals baseline uses prior presidential results and the president's party in midterms. Historical predictions train on earlier election cycles only.</p><p><strong>TEST_RACES races, TEST_YEARS.</strong> Brier score: <strong>BRIER</strong>. Winner accuracy: <strong>ACCURACY</strong>. Margin RMSE: <strong>RMSE_VALUE points</strong>.</p><p class="sub">These diagnostics cover only the fundamentals baseline. They do not validate the 2026 polling blend, experimental document features, or contest strategy. The training data currently ends in 2018. Contemporary validation and a more complete candidate model are still required. Data revisions are retrospective, so this is not a real-time data-vintage backtest.</p><p><a href="validation.json">Full historical validation records</a></p></section>
<section><h2>Strategy sensitivity, under explicit assumptions</h2><p class="sub">SCENARIO_COUNT simulated elections for MARKET_COUNT mapped races at each assumed shared-error setting. The provisional model is treated as true. Five hypothetical competitors are compared; ties split first-place credit. These are not actual competition winning probabilities.</p><div class="scroll"><table><thead><tr><th>Policy</th><th>First-place share, no shared error</th><th>First-place share, 0.25 latent correlation</th><th>First-place share, 0.50 latent correlation</th><th>Mean points, no shared error</th></tr></thead><tbody>SIM_ROWS</tbody></table></div></section>
<section><h2>Before this becomes an entry</h2><ul>BLOCKERS</ul><p>The immediate modeling priorities are contemporary historical validation, primary-source poll verification in races that drive the score, and candidate/market mapping checks. Keep document features experimental until they beat the same model without those features.</p><p><a href="submission-readiness.json">Readiness record</a> · <a href="research.json">Complete research snapshot</a> · <a href="../README.md">How to reproduce and continue</a></p></section>
<section><details style="max-width:none"><summary>Source archive and hashes</summary><p>Each raw source is stored with its URL, retrieval timestamp, and SHA-256 hash. Refreshing adds new captures. No paid API, account registration, or external model call is needed by this pipeline.</p><div class="scroll"><table><thead><tr><th>Source</th><th>Links</th><th>Retrieved UTC</th><th>SHA-256 prefix</th></tr></thead><tbody>SOURCE_ROWS</tbody></table></div></details></section>
<footer>Research snapshot AS_OF. Public sources only. The research repository is public on GitHub. No trading, organizer contact, or submission has occurred.</footer>
</main><script>
const p=document.getElementById('p'),q=document.getElementById('q');
function update(){const a=Number(p.value)/100,b=Number(q.value)/100;const dog=b*-Math.log2(Math.max(a,.01)),fav=(1-b)*-Math.log2(Math.max(1-a,.01));const ceiling=Math.max(dog,fav,.01);document.getElementById('p-label').textContent=p.value+'%';document.getElementById('q-label').textContent=q.value+'%';document.getElementById('dog-value').textContent=dog.toFixed(3);document.getElementById('fav-value').textContent=fav.toFixed(3);document.getElementById('dog-bar').style.width=(100*dog/ceiling)+'%';document.getElementById('fav-bar').style.width=(100*fav/ceiling)+'%';document.getElementById('decision').textContent=Math.abs(dog-fav)<1e-9?'Equal expected points.':(dog>fav?'Underdog':'Favorite')+' has the higher expected points.';}
p.addEventListener('input',update);q.addEventListener('input',update);update();
document.getElementById('filter').addEventListener('input',e=>{const text=e.target.value.toLowerCase();document.querySelectorAll('.race').forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(text));});
</script></body></html>'''
    replacements = {
        'AS_OF':escape(data['as_of']), 'RACE_COUNT':str(counts['races']),
        'MARKET_COUNT':str(counts['verified_market_races']), 'POLL_COUNT':str(counts['poll_feeds']),
        'DOCUMENT_COUNT':str(int(evidence['visual_review_passed'])), 'RACE_ROWS':''.join(rows),
        'TEST_RACES':str(validation['test_races']),
        'TEST_YEARS':', '.join(map(str,validation['test_years'])),
        'BRIER':f'{validation["brier"]:.3f}', 'ACCURACY':f'{validation["winner_accuracy"]:.1%}',
        'RMSE_VALUE':f'{validation["rmse_margin_points"]:.1f}',
        'SCENARIO_COUNT':f'{sim["simulations"]:,}', 'SIM_ROWS':''.join(simrows),
        'BLOCKERS':blockers, 'SOURCE_ROWS':''.join(source_rows),
    }
    for token,value in replacements.items():
        html=html.replace(token,value)
    # Link to publisher pages instead of redistributing their page images.
    image_start = html.index('<div class="pair">')
    image_end = html.index('</div>', image_start)+len('</div>')
    html = html[:image_start] + '''<p><a href="https://www.uh.edu/hobby/texas2026/statewide26.pdf#page=6">View original chart (PDF page 6)</a> ·
    <a href="https://www.uh.edu/hobby/texas2026/statewide26.pdf#page=7">View original table (PDF page 7)</a>.</p>
    <p class="sub">Publisher page images are omitted from the public repository. Source hashes and the reviewed numerical extraction are retained.</p>''' + html[image_end:]
    mm = data.get('multimodal_experiment')
    if mm:
        audit = data.get('multimodal_input_audit')
        audit_notice = ''
        if audit:
            audit_notice = f'''<p><strong>Input defect found after the first experiment:</strong>
            the default square crop discards over half the source image area in
            {audit['training_images_losing_over_half_the_area']} of {audit['training_images']} training images.
            In a checked example it cuts off candidate names and topline values. The versioned v0.2 experiment repairs this;
            the original failed experiment is preserved below.</p><p><a href="image-input-audit.json">Original image-input audit</a> ·
            <a href="../docs/IMPROVEMENT_PLAN.md">Ranked improvement plan</a></p>'''
        mm_rows = []
        entries = [('Uncorrected poll', mm['raw_poll'])] + [
            (name.replace('_', ' + '), value) for name, value in mm['models'].items()]
        for name, value in entries:
            scores = value['test']
            mm_rows.append(f'<tr><td>{escape(name)}</td><td>{scores["margin_rmse"]:.2f}</td>'
                           f'<td>{scores["brier"]:.3f}</td><td>{scores["accuracy"]:.0%}</td></tr>')
        mm_html = f'''<section id="multimodal"><h2>The trained multimodal experiment</h2>
        <p>A real neural forecasting network combines numeric poll data with text and images of original poll documents.
        Free January 2021 CLIP encoders stay frozen; our small forecasting network learns its own weights locally.</p>
        <p><strong>{mm['training_races']} training races (2014–2020), 2 validation races (2022), 3 held-out races (2024).</strong>
        The full network has 113 trained parameters per seed; three seeds are averaged. New service spend: $0.</p>
        <div class="notice"><strong>Experimental. Not used in contest picks.</strong> Adding text and images made this
        tiny 2024 holdout worse than the matched numeric-only network. All trained variants missed all three winners.
        Three close races cannot establish general forecasting skill. The 6-point probability scale is an assumption.</div>
        {audit_notice}
        <div class="scroll"><table><thead><tr><th>Inputs</th><th>Margin RMSE ↓</th><th>Brier score ↓</th><th>Winner accuracy</th></tr></thead>
        <tbody>{''.join(mm_rows)}</tbody></table></div>
        <p class="sub">Each model uses the same original poll snapshots. Scaling and dimensionality reduction fit on
        training years only; epochs are selected on 2022 only. The 2024 holdout was then opened once for comparison.</p>
        <p><img src="multimodal-ablation.png" alt="All five comparisons: the full multimodal network has worse held-out Brier score than the numeric network" style="max-width:100%;height:auto" loading="lazy"></p><p><a href="multimodal.json">Full training results and race predictions</a> ·
        <a href="../docs/MODEL_CARD.md">Model card</a> · <a href="../docs/REPRODUCING.md">Reproduce the experiment</a></p></section>'''
        html = html.replace('<section><h2>Strategy sensitivity', mm_html+'<section><h2>Strategy sensitivity')
        html = html.replace('Subgroup and visual features are held out of the predictive model until historical testing supports their use.',
                            'The separate multimodal neural experiment below has now been trained and tested. Its features remain excluded from contest picks because the first holdout did not support them.')
    v2=data.get('v2_document_experiment');numeric=data.get('v2_numeric_experiment')
    if v2 and numeric:
        entries=[('Numeric baseline; no document correction',v2['baseline']['test'],None)]
        entries.extend((name.replace('_',' + '),m['test'],m['selected']['weight']) for name,m in v2['models'].items())
        cells=''.join(f'<tr><td>{escape(name)}</td><td>{scores["margin_rmse"]:.2f}</td><td>{scores["brier"]:.3f}</td><td>{scores["accuracy"]:.0%}</td><td>{"—" if weight is None else f"{weight:.2f}"}</td></tr>' for name,scores,weight in entries)
        num=numeric['test']['selected'];raw=numeric['test']['raw_poll_same_calibration']
        section=f'''<section id="version-two"><h2>Version 0.2: larger data, repaired images, calibrated uncertainty</h2>
        <p><strong>{numeric['data']['races']} numeric race snapshots and 50 reviewed document races.</strong>
        Numeric selection uses 249 training and 45 development races, then refits through 2020. Probability calibration uses 24 races in 2022.
        The document head uses 29 earlier training races, selects on five 2020 races, and refits on 34. Fourteen new 2024 races have reviewed documents.</p>
        <p>The larger selected numeric model reduced margin RMSE from <strong>{raw['margin_rmse']:.2f} to {num['margin_rmse']:.2f} points</strong>
        and Brier score from {raw['brier']:.3f} to {num['brier']:.3f} on 25 new 2024 test races. This test excludes the three consumed v0.1 races (MI, PA, WI), includes many safe seats, and covers only one election cycle. It is not a probability of winning this competition.</p>
        <div class="notice"><strong>Direct image/text correction was not selected.</strong> Earlier-cycle validation selected zero contribution for the full multimodal branch. Its test result therefore equals the numeric baseline.
        A lower score than the separately trained numeric document head does not demonstrate an image benefit. All neural heads were actually trained; unblended diagnostic scores and weights are retained.</div>
        <div class="scroll"><table><thead><tr><th>Matched document arm</th><th>Margin RMSE ↓</th><th>Brier ↓</th><th>Accuracy</th><th>Document weight</th></tr></thead><tbody>{cells}</tbody></table></div>
        <p><img src="v2-ablation.png" alt="Matched 14-race document comparison; validation suppresses the image and full document corrections" style="max-width:100%;height:auto" loading="lazy"></p>
        <p>Full-image padding and overlapping readable tiles replace square center cropping. Original PDFs and publisher table screenshots are hash-pinned and visually reviewed; their images are excluded from public Git. OCR is a separately tested local extraction fallback, not an automatic reviewer or an extra forecast signal.</p>
        <p>All 14 new document-test races use publishers absent from head training. This is confounded with the new election cycle. Historical availability is partly proxied from median field dates and the archives have retrospective revisions. The prespecified Monmouth/Emerson PDF test was unavailable under the collection rule; it is not reported as passed.</p>
        <p><a href="v2-numeric.json">Expanded numeric results</a> · <a href="v2-documents.json">Matched document results</a> · <a href="v2-image-audit.json">Actual-input audit</a> ·
        <a href="v2-ocr.json">Limited OCR diagnostic</a> · <a href="../docs/V2_MODEL_CARD.md">v0.2 model card</a> · <a href="https://github.com/aaidenko/model-citizens-multimodal">Public GitHub repository</a></p></section>'''
        html=html.replace('<section id="multimodal">',section+'<section id="multimodal">')
    if not evidence['visual_review_passed']:
        html=html.replace("The University of Houston's Texas poll gives the same 48 / 46 / 3 / 3 figures in its chart and overall table. The original methodology says October 10–14, while the university-hosted tracker lists October 10–15. The original report controls the field dates.",
                          'The source PDF has changed since its page images were visually verified. The new extraction requires review before it can supply a primary-source correction.')
        html=html.replace('View original chart (PDF page 6)', 'Changed original chart: needs review')
    (root/'reports/index.html').write_text(html)
