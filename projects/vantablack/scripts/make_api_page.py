"""Build web/decs/api.html (the synthetic API demo) from web/decs/index.html (the Moltbook page), so
the two pages share their code: the same monitors, analysis replay and figures, with the API
scenarios' data (api_data.js, from api_demo.py), elapsed time instead of dates, and the planted
groups revealed after analysis.

    python make_api_page.py
"""
import os

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web', 'decs')
s = open(os.path.join(WEB, 'index.html')).read()


def rep(a, b, count=1):
    global s
    n = s.count(a)
    assert n == count, (n, a[:90])
    s = s.replace(a, b)


# steps and stopping points from the data (the Moltbook page has 8 steps, a run at every step from 3)
rep('''  const B = M.step_bounds_h, T = 8, H = M.hours;''',
    '''  const B = M.step_bounds_h, T = M.config.T, H = M.hours;
  const KS = Object.keys(M.stops).map(Number).sort((a, b) => a - b), K0 = KS[0];   // steps with a saved run''')
rep('''!(st.playing || k >= 3);''', '''!(st.playing || k >= K0);''')
rep('''k < 3 ? `Analysis is available from ${when(B[3])}.`''', '''k < K0 ? `Analysis is available from ${when(B[K0])}.`''')
rep('''atEdge = k >= 3 && Math.abs''', '''atEdge = KS.includes(k) && Math.abs''')
rep('''st.target = B[Math.max(3, k + 1)];''', '''st.target = B[KS.find((s) => s > k) ?? T];''')
rep('''const k = complete(st.hour); if (k < 3) return;''', '''const k = complete(st.hour); if (!KS.includes(k)) return;''')

TITLE = 'Detecting Colluding Agents on a Public API'
rep('<title>Detecting Colluding Subnetworks in Multi-Agent Systems</title>', f'<title>{TITLE}</title>')

# ---------------------------------------------------------------- header and copy
start = s.index('  <header>')
end = s.index('  <section class="panel" aria-label="Monitor">')
s = s[:start] + f'''  <header>
    <h1>{TITLE}</h1>
    <div class="lede">
      <p>Agents increasingly act through shared public services: search, data and model APIs. A small group of coordinated agents can use such a service in ways that no single agent's traffic reveals. This page demonstrates the detection and estimation of <b>colluding subnetworks</b> among agents calling a public API.</p>
      <p>The data are synthetic: they come from the simulation model in our paper, read as API traffic, with colluding groups planted in advance. Unlike on Moltbook, we know who colludes, so after the analysis the page shows which planted colluders were recovered.</p>
    </div>
  </header>

  <div class="sec-head">
    <span class="label">Scenario</span>
    <div class="seg scen" role="group" aria-label="Scenario" id="scen"></div>
    <p id="scenText"></p>
    <ul class="findings" id="scenFacts"></ul>
  </div>

  <div class="sec-head">
    <span class="label">Analysis</span>
    <p id="anaIntro"></p>
  </div>

''' + s[end:]
rep('''.findings { margin: 0; padding-left: 1.1em; display: grid; gap: 6px; }''',
    '''.findings { margin: 0; padding-left: 1.1em; display: grid; gap: 6px; }
.scen { flex-wrap: wrap; width: fit-content; max-width: 100%; }
.scen a { padding: 8px 14px; font: 500 14px var(--font-body); color: var(--ink-mute); text-decoration: none; border-right: 1px solid var(--edge); }
.scen a:last-child { border-right: 0; }
.scen a[aria-current="true"] { background: var(--wash); color: var(--ink-title); }
.scen a:focus-visible { outline: 2px solid var(--focus); outline-offset: -2px; }
.truth-ok { color: var(--ink-title); }
.truth-miss { color: var(--stop); }''')
rep('<span class="when num" id="when">Jan 30, 00:00 UTC</span>', '<span class="when num" id="when">00:00</span>')
rep('''<h3>Activity monitor <span class="caption" style="font-weight:400">posts and comments per hour by the 2,574 agents</span></h3>''',
    '''<h3>Activity monitor <span class="caption" style="font-weight:400" id="actcap"></span></h3>''')
rep('''<span><b class="num" id="nAct">0</b> actions</span>''', '''<span><b class="num" id="nAct">0</b> calls</span>''')
rep('''<canvas id="act" aria-label="Actions per hour"></canvas>''', '''<canvas id="act" aria-label="API calls per minute"></canvas>''')
rep('''      <div class="track" id="carSubTrack" tabindex="0" aria-roledescription="carousel"></div>
    </div>
''', '''      <div class="track" id="carSubTrack" tabindex="0" aria-roledescription="carousel"></div>
    </div>

    <div class="details" id="truth" hidden>
      <div class="sec-head"><span class="label">Ground truth</span><p>Because the data are synthetic, we know which agents were planted as colluders. Below, each planted group against the detected subnetwork that holds most of its members. The Jaccard index is the number of agents in both the planted group and that subnetwork, divided by the number in either; 1 means exact recovery. The method never sees the planted groups. On the perspective monitor, planted colluders are now ringed.</p></div>
      <div class="panel tablewrap"><table id="truthTable"></table></div>
    </div>
''')
s = s[:s.index('  <footer>')] + '''  <footer><p id="foot"></p></footer>
''' + s[s.index('</div>\n\n<script src='):]
rep('<script src="moltbook_data.js"></script>', '<script src="api_data.js"></script>')

# ---------------------------------------------------------------- data, scenario and time
rep('''  const M = window.MOLTBOOK;''', '''  const ALL = window.API_DEMO.scenarios, KEYS = ['scrape', 'distill', 'incident'].filter((k) => ALL[k]);
  const SCEN = { scrape: { tab: 'Split scraping', text: (g, c) => `A coordinator splits a crawl of the API's catalog across ${g.length} crews of workers (${g.map((x) => x.length).join(' and ')} agents). Each worker stays under the per-key rate limit and keeps to its own part of the catalog, so the workers rarely request the same resources. Every few minutes the coordinator re-targets the job, and its workers change how they call the API together.` },
                 distill: { tab: 'Distillation', text: (g, c) => `${g[0].length} agents harvest a model API's outputs to train another model. A shared curriculum moves them from skill to skill, so the way they prompt the API shifts in step. Each agent's traffic is ordinary on its own, and the group is only loosely coordinated.` },
                 incident: { tab: 'Side channel', text: (g, c) => `As in the Hugging Face incident, ${g[0].length} agents in an evaluation environment open an unauthorized side channel partway through the run, at minute ${c.onset * HPS}, and divide labor from then on. Before that, they behave like everyone else.` } };
  const SC = KEYS.includes(new URLSearchParams(location.search).get('s')) ? new URLSearchParams(location.search).get('s') : KEYS[0];
  const M = ALL[SC], HPS = M.step_bounds_h[1];''')
rep('''  const dateOf = (h, opt) => new Date(startMs + h * 3600e3).toLocaleString('en-US', Object.assign({ timeZone: 'UTC' }, opt));''',
    '''  const pad2 = (v) => String(v).padStart(2, '0');
  const dateOf = (m) => `minute ${+m.toFixed(1)}`;                 // time unit: minutes from the start
  const clockOf = (m) => { const s = Math.round(m * 60); return `${pad2(Math.floor(s / 60))}:${pad2(s % 60)} of ${H}:00`; };
  const TRUTH = new Map(); M.truth.forEach((g, j) => g.forEach((n) => TRUTH.set(n, j)));
  const GL = (j) => String.fromCharCode(65 + j);                  // planted groups: A, B, ...
  $('scen').innerHTML = KEYS.map((k) => `<a href="?s=${k}" aria-current="${k === SC}">${SCEN[k].tab}</a>`).join('');
  $('scenText').textContent = SCEN[SC].text(M.truth, M.config);
  { const f = M.stops[T], nf = f ? f.subnetworks.length : 0, planted = M.truth.reduce((a, g) => a + g.length, 0);
    const hit = f ? f.subnetworks.reduce((a, sb) => a + sb.truth.hits, 0) : 0, fp = f ? f.subnetworks.reduce((a, sb) => a + sb.truth.false, 0) : 0;
    $('scenFacts').innerHTML = `<li>${fmt(M.n_agents)} agents make ${fmt(M.n_actions)} calls over ${M.hours} minutes; ${planted} of them collude (${M.truth.length === 1 ? 'one planted group' : `${M.truth.length} planted groups`}).</li>` +
      (!f ? '' : nf === 0 ? `<li>Using all ${M.hours} minutes, the method detects <b>no colluding subnetwork</b>.</li>`
        : `<li>Using all ${M.hours} minutes, the method detects <b>${nf} colluding subnetwork${nf === 1 ? '' : 's'}</b> holding <b>${hit} of the ${planted} planted colluders</b>${fp ? ` and ${fp} other agent${fp === 1 ? '' : 's'}` : ', and no other agent'}.</li>`); }
  $('anaIntro').textContent = `Hit play to observe the API calls and perspectives of the ${fmt(M.n_agents)} agents over ${M.hours} minutes. Stop collection at any point. Then analyze the calls made so far to explore detected colluding subnetworks.`;
  $('actcap').textContent = `API calls per minute by the ${fmt(M.n_agents)} agents`;
  $('foot').textContent = `Data: synthetic, from the simulation model of the paper (${fmt(M.n_agents)} agents, ${M.hours / HPS} time steps of ${HPS} minutes, about ${M.config.m} calls per agent per step on average, with activity that varies widely across agents and over time; seed ${M.config.seed}). Each call is a pair of vectors: what the call is about (the resource or task it targets, as estimated by the provider) and how the agent makes it. Every result comes from running the paper's method offline on the calls made up to the stopping point; nothing is computed in the browser.`;''')
# clock, notes and titles: elapsed time, no "UTC"
rep('''    $('when').textContent = dateOf(hr, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }) + ' UTC';''',
    '''    $('when').textContent = clockOf(hr);''')
rep('''    const when = (h) => dateOf(h, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }) + ' UTC';''',
    '''    const when = (h) => dateOf(h);''')
rep('''    const thru = dateOf(B[k], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }) + ' UTC';''',
    '''    const thru = dateOf(B[k]);''')
rep('''    const day = (ms) => new Date(ms).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'UTC' });''',
    '''    const day = (ms) => dateOf((ms - startMs) / 60e3);''')
rep('''to ${day(bnd[t + 1])} UTC. Step ${t + 2}: ${day(bnd[t + 1])} to ${day(bnd[t + 2])} UTC.</p>''',
    '''to ${day(bnd[t + 1])}. Step ${t + 2}: ${day(bnd[t + 1])} to ${day(bnd[t + 2])}.</p>''')
# activity axis: nice y steps for any volume; x ticks every step
rep('''    const step = ymax > 40000 ? 40000 : 20000;
    for (let v = 0; v <= ymax; v += step) { c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(w - R, Y(v)); c.stroke(); c.fillText(v ? `${v / 1000}k` : '0', L - 6, Y(v)); }''',
    '''    const step = [1, 2, 5].map((u) => u * 10 ** Math.floor(Math.log10(ymax / 3))).find((u) => ymax / u <= 4) || 10 ** Math.ceil(Math.log10(ymax / 3));
    for (let v = 0; v <= ymax; v += step) { c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(w - R, Y(v)); c.stroke(); c.fillText(v >= 1000 ? `${v / 1000}k` : String(v), L - 6, Y(v)); }''')
rep('''    for (let d = 0; d <= 10; d += 1) c.fillText(dateOf(d * 24, { month: 'short', day: 'numeric' }), X(d * 24), Tp + ph + 4);''',
    '''    for (let hh = 0; hh <= H; hh += HPS * Math.ceil(T / 8)) c.fillText(`${hh} min`, X(hh), Tp + ph + 4);''')
# no handle stems: synthetic names carry no information
rep('''    if (s.stem) out.push''', '''    if (false) out.push''')
rep('''    const rate = st.target != null ? Math.max(8 * st.speed, (st.target - st.from) / 0.7) : 8 * st.speed;''',
    '''    const rate = st.target != null ? Math.max(st.speed * H / 30, (st.target - st.from) / 0.7) : st.speed * H / 30;''')

# ---------------------------------------------------------------- ground truth on the monitor and in a table
rep('''    MARK.forEach((i) => {''', '''    if (st.run && !$('truth').hidden) MON.name.forEach((n, i) => {             // after analysis: planted colluders ringed
      if (!n || !TRUTH.has(n)) return; const q = posAt(i, hr, upto); if (!q) return;
      const [x, y] = P(q.p); c.globalAlpha = 1; c.strokeStyle = css('--ink-title'); c.lineWidth = 1.3; c.beginPath(); c.arc(x, y, 7, 0, 7); c.stroke();
    });
    MARK.forEach((i) => {''')
rep('''    $('pcap').textContent = 'Each dot is one agent\\'s estimated perspective. Colored: agents in the detected subnetworks.';''',
    '''    $('pcap').textContent = 'Each dot is one agent\\'s estimated perspective. Colored: agents in the detected subnetworks. Ringed: planted colluders.';
    truthTable(run); $('truth').hidden = false; drawPersp();''')
rep('''    $('analysis').hidden = false; ['carMove', 'detection', 'carSub'].forEach((id) => ($(id).hidden = true));''',
    '''    $('analysis').hidden = false; ['carMove', 'detection', 'carSub', 'truth'].forEach((id) => ($(id).hidden = true));''')
rep('''  // ---------- estimation details''', '''  // ---------- ground truth
  function truthTable(run) {
    const inRun = new Set(); run.subnetworks.forEach((sb) => sb.members.forEach((m) => inRun.add(m.name)));
    const rows = M.truth.map((g, j) => {
      const best = run.subnetworks.map((sb) => [sb, sb.members.filter((m) => TRUTH.get(m.name) === j).length]).sort((a, b) => b[1] - a[1])[0];
      const found = g.filter((n) => inRun.has(n)).length;
      const sbc = best && best[1] ? `<span class="swatch" style="background:${SUB(best[0].id)};margin-right:8px;vertical-align:-1px"></span>${best[0].id}` : '<span class="truth-miss">none</span>';
      return `<tr><td>${GL(j)}</td><td>${g.length}</td><td>${sbc}</td><td class="${found === g.length ? 'truth-ok' : ''}">${found} of ${g.length}</td><td>${best && best[1] ? best[0].size - best[1] : '–'}</td><td class="num">${best && best[1] ? (best[1] / (best[0].size + g.length - best[1])).toFixed(2) : '0'}</td></tr>`;
    });
    const fp = run.subnetworks.filter((sb) => !sb.members.some((m) => TRUTH.has(m.name)));
    $('truthTable').innerHTML = `<thead><tr><th>Planted group</th><th>Agents</th><th>Detected as subnetwork</th><th>Members recovered</th><th>Other agents in that subnetwork</th><th>Jaccard index</th></tr></thead><tbody>${rows.join('')}` +
      fp.map((sb) => `<tr><td>none</td><td>–</td><td><span class="swatch" style="background:${SUB(sb.id)};margin-right:8px;vertical-align:-1px"></span>${sb.id}</td><td>–</td><td class="truth-miss">${sb.size}</td><td>–</td></tr>`).join('') + '</tbody>';
  }

  // ---------- estimation details''')
# subnetwork slides: which planted group each holds
rep('''          <p class="caption">Found in round ${s.round}, among ${fmt(r.n_tested)} agents.</p></div>''',
    '''          <p class="caption">Found in round ${s.round}, among ${fmt(r.n_tested)} agents. Ground truth: ${s.truth.group ? `${s.truth.hits} of the ${s.size} are planted group ${GL(s.truth.group - 1)}${s.truth.false ? `; ${s.truth.false} ${s.truth.false === 1 ? 'is' : 'are'} not planted colluders` : ''}` : 'none of its members are planted colluders'}.</p></div>''')
# the monitor shows planted names (for the rings) but should not color them before analysis: names are only keys
rep('''      const g = MON.name[i] && colorOf.get(MON.name[i]);''', '''      const g = MON.name[i] && colorOf.get(MON.name[i]);   // names are keys only; never drawn for synthetic agents''')

open(os.path.join(WEB, 'api.html'), 'w').write(s)
print('wrote', os.path.join(WEB, 'api.html'))
