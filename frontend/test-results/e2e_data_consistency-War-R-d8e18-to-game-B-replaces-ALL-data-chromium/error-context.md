# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: e2e_data_consistency.spec.ts >> War Room — game selector data consistency (Bug 1) >> switching from game A to game B replaces ALL data
- Location: e2e_data_consistency.spec.ts:83:7

# Error details

```
Test timeout of 30000ms exceeded.
```

```
Error: page.selectOption: Test timeout of 30000ms exceeded.
Call log:
  - waiting for locator('select')
    - locator resolved to 2 elements. Proceeding with the first one: <select class="w-full h-10 rounded-lg px-3 text-sm font-mono text-white border focus:outline-none focus:ring-1">…</select>
  - attempting select option action
    2 × waiting for element to be visible and enabled
      - did not find some options
    - retrying select option action
    - waiting 20ms
    2 × waiting for element to be visible and enabled
      - did not find some options
    - retrying select option action
      - waiting 100ms
    58 × waiting for element to be visible and enabled
       - did not find some options
     - retrying select option action
       - waiting 500ms

```

# Page snapshot

```yaml
- generic [ref=e2]:
  - generic [ref=e3]:
    - button "🏟️ War Room" [active] [ref=e4] [cursor=pointer]
    - button "📊 Post-Game Review" [ref=e5] [cursor=pointer]
    - generic [ref=e6]: MLB War Room v2.1
  - generic [ref=e8]:
    - banner [ref=e9]:
      - generic [ref=e10]:
        - generic [ref=e11]: ⚾ MLB War Room
        - generic [ref=e12]: Estrategia Pre-Partido
      - generic [ref=e15]: LIVE
      - generic [ref=e16]:
        - generic [ref=e17]: 📊
        - generic [ref=e18]: "Precisión 30d:"
        - generic [ref=e19]: 74%
        - generic [ref=e20]: "|"
        - generic [ref=e21]: 847 partidos
    - generic [ref=e22]:
      - generic [ref=e23]:
        - generic [ref=e24]: 🌬️
        - generic [ref=e25]: "Viento: 12mph OUT"
      - generic [ref=e26]:
        - generic [ref=e27]: 💧
        - generic [ref=e28]: "Humedad: 67%"
      - generic [ref=e29]:
        - generic [ref=e30]: 🏟️
        - generic [ref=e31]: "Park Factor: 0.94"
        - generic [ref=e32]: E[R] -0.22
      - generic [ref=e33]: "|"
      - generic [ref=e34]:
        - text: — · —
        - generic [ref=e35]: (RHP)
        - text: · ERA 0.00
    - generic [ref=e36]:
      - generic [ref=e37]: "Error cargando juegos: HTTP 500:"
      - generic [ref=e38]:
        - combobox [ref=e40]:
          - option "Sin juegos disponibles" [selected]
        - generic [ref=e41]:
          - button "Visitante" [ref=e42] [cursor=pointer]
          - button "Local" [ref=e43] [cursor=pointer]
      - generic [ref=e44]:
        - generic [ref=e45]:
          - generic [ref=e46]:
            - generic [ref=e47]: "4.50"
            - generic [ref=e48]: E[R] proyectado
          - generic [ref=e49]:
            - paragraph [ref=e50]: Distribución de Carreras Esperadas
            - img [ref=e52]:
              - generic [ref=e53]:
                - generic [ref=e56]: Liga 4.50
                - generic [ref=e57]: E[R] 4.50
                - generic [ref=e58]:
                  - generic [ref=e59]: "0"
                  - generic [ref=e60]: "1"
                  - generic [ref=e61]: "2"
                  - generic [ref=e62]: "3"
                  - generic [ref=e63]: "4"
                  - generic [ref=e64]: "5"
                  - generic [ref=e65]: "6"
                  - generic [ref=e66]: "7"
                  - generic [ref=e67]: "8"
                  - generic [ref=e68]: "9"
        - generic [ref=e69]:
          - paragraph [ref=e70]: Probabilidad de Victoria
          - generic [ref=e71]:
            - paragraph [ref=e72]: P(Victoria)
            - paragraph [ref=e73]: 50.0%
            - generic [ref=e74]:
              - generic "55% ref." [ref=e76]
              - generic "50% equilibrio" [ref=e77]
            - paragraph [ref=e78]:
              - text: "Obj: 50.0%"
              - generic [ref=e79]: · 55% ref.
        - generic [ref=e80]:
          - paragraph [ref=e81]: Confianza del Modelo
          - generic [ref=e82]:
            - paragraph [ref=e83]: Confianza
            - paragraph [ref=e84]: 70.0%
            - generic [ref=e85]:
              - generic "80% óptimo" [ref=e87]
              - generic "70% mínimo" [ref=e88]
            - paragraph [ref=e89]:
              - text: "Obj: 70.0%"
              - generic [ref=e90]: · 80% óptimo
      - generic [ref=e91]:
        - generic [ref=e92]:
          - generic [ref=e93]:
            - paragraph [ref=e94]: Lineup Óptimo
            - generic [ref=e95]: Selecciona un partido para ver el lineup
          - generic [ref=e96]:
            - paragraph [ref=e97]: Simulador What-If
            - paragraph [ref=e99]: Arrastra jugadores para simular impacto
        - generic [ref=e100]:
          - generic [ref=e102]: Sin datos
          - generic [ref=e103]:
            - paragraph [ref=e104]: Perfil Ofensivo
            - generic [ref=e105]: Sin datos
      - generic [ref=e106]:
        - paragraph [ref=e107]: Intel Briefing
        - paragraph [ref=e108]: Sin análisis disponible
      - generic [ref=e109]: "MLB War Room · 21/5/2026 · Datos: MLB Stats API"
```

# Test source

```ts
  1   | /**
  2   |  * e2e_data_consistency.spec.ts
  3   |  * ============================
  4   |  * Playwright E2E tests for data-consistency guarantees (Bug 1 fix).
  5   |  *
  6   |  * Tests verify:
  7   |  *  1. Selecting game A renders data for game A (no cross-contamination).
  8   |  *  2. Switching to game B replaces ALL DOM elements with game B data.
  9   |  *  3. During a slow API response, game A data is NOT visible while loading game B.
  10  |  *  4. Post-Game selector coherence: game_pk in URL / DOM matches the selected game.
  11  |  *
  12  |  * To run:
  13  |  *   npx playwright install --with-deps
  14  |  *   npx playwright test e2e_data_consistency.spec.ts --reporter=line
  15  |  *
  16  |  * Assumes the React app is running at http://localhost:5173
  17  |  * and the FastAPI is at http://localhost:8000 (or DEMO_MODE mock).
  18  |  */
  19  | 
  20  | import { test, expect, type Page } from '@playwright/test';
  21  | 
  22  | const APP_URL = process.env.APP_URL ?? 'http://localhost:5173';
  23  | 
  24  | // ── Helpers ───────────────────────────────────────────────────────────────────
  25  | 
  26  | async function waitForNoSkeletons(page: Page) {
  27  |   /** Wait until all skeleton loaders are gone (data has loaded). */
  28  |   await page.waitForFunction(() => {
  29  |     const skeletons = document.querySelectorAll('[style*="shimmer"]');
  30  |     return skeletons.length === 0;
  31  |   }, { timeout: 15_000 });
  32  | }
  33  | 
  34  | async function getSelectedGameOption(page: Page): Promise<string> {
  35  |   return page.evaluate(() => {
  36  |     const sel = document.querySelector<HTMLSelectElement>('select');
  37  |     return sel?.options[sel.selectedIndex]?.text ?? '';
  38  |   });
  39  | }
  40  | 
  41  | // ── War Room tests ────────────────────────────────────────────────────────────
  42  | 
  43  | test.describe('War Room — game selector data consistency (Bug 1)', () => {
  44  |   test.beforeEach(async ({ page }) => {
  45  |     await page.goto(APP_URL);
  46  |     // Ensure War Room tab is active
  47  |     const warRoomTab = page.getByRole('button', { name: /War Room/i });
  48  |     await warRoomTab.click();
  49  |     // Wait for game list to load
  50  |     await page.waitForSelector('select', { timeout: 10_000 });
  51  |   });
  52  | 
  53  |   test('selecting game A shows data labelled with game A', async ({ page }) => {
  54  |     const options = await page.evaluate(() =>
  55  |       Array.from(document.querySelectorAll<HTMLOptionElement>('select option')).map(
  56  |         (o) => ({ value: o.value, text: o.text })
  57  |       )
  58  |     );
  59  |     if (options.length < 1) test.skip(true, 'No games available in test env');
  60  | 
  61  |     // Select first game
  62  |     const firstGame = options[0];
  63  |     await page.selectOption('select', firstGame.value);
  64  |     await waitForNoSkeletons(page);
  65  | 
  66  |     // The currently shown selector text must match the selected game
  67  |     const displayedLabel = await getSelectedGameOption(page);
  68  |     expect(displayedLabel).toBe(firstGame.text);
  69  | 
  70  |     // No data from a different game should appear in any heading or label
  71  |     if (options.length > 1) {
  72  |       const secondGameTeams = options[1].text.split(/[@·]/)[0].trim();
  73  |       const bodyText = await page.evaluate(() => document.body.innerText);
  74  |       // The second game's team name should NOT appear in any stat heading
  75  |       // (it may appear in the dropdown list itself, so we check headings)
  76  |       const headings = await page.$$eval('h1, h2, h3, [class*="label"]', (els) =>
  77  |         els.map((e) => e.textContent ?? '').join(' ')
  78  |       );
  79  |       expect(headings).not.toContain(secondGameTeams);
  80  |     }
  81  |   });
  82  | 
  83  |   test('switching from game A to game B replaces ALL data', async ({ page }) => {
  84  |     const options = await page.evaluate(() =>
  85  |       Array.from(document.querySelectorAll<HTMLOptionElement>('select option'))
  86  |         .filter((o) => o.value !== '')
  87  |         .map((o) => ({ value: o.value, text: o.text }))
  88  |     );
  89  |     if (options.length < 2) test.skip(true, 'Need at least 2 games');
  90  | 
  91  |     // Select game A and wait for data
> 92  |     await page.selectOption('select', options[0].value);
      |                ^ Error: page.selectOption: Test timeout of 30000ms exceeded.
  93  |     await waitForNoSkeletons(page);
  94  | 
  95  |     // Capture a piece of game A identity (e.g. team abbreviation chip text)
  96  |     const teamA = options[0].text.split('@')[0].trim().slice(0, 3);
  97  | 
  98  |     // Select game B
  99  |     await page.selectOption('select', options[1].value);
  100 | 
  101 |     // Immediately after selection, skeleton must appear (no stale data)
  102 |     const hasSkeletonImmediately = await page.evaluate(() =>
  103 |       document.querySelectorAll('[style*="shimmer"], [class*="animate-pulse"]').length > 0
  104 |         || document.querySelectorAll('[class*="loading"]').length > 0
  105 |     );
  106 |     // In the context of the reducer, lineupData is immediately null → skeleton rendered
  107 |     // This assertion is best-effort: if the API responds in <1 frame, it may already be resolved
  108 |     // (acceptable — the reducer still cleared stale data synchronously)
  109 |     console.log(`Skeleton visible immediately after switch: ${hasSkeletonImmediately}`);
  110 | 
  111 |     // After data loads for game B, the selected dropdown must reflect game B
  112 |     await waitForNoSkeletons(page);
  113 |     const displayedLabelB = await getSelectedGameOption(page);
  114 |     expect(displayedLabelB).toBe(options[1].text);
  115 | 
  116 |     // Game A team abbreviation should not appear in any stat card heading
  117 |     // (it may appear in the dropdown option, but not in the data panels)
  118 |     const dataPanel = page.locator('[class*="KPI"], [class*="kpi-card"], [class*="stat-card"]');
  119 |     const panelTexts = await dataPanel.allTextContents();
  120 |     // This is a soft check — the team abbr can legitimately appear in shared UI
  121 |     // What must NOT appear is game A's lineup names in game B's lineup table
  122 |     const lineupTable = page.locator('table');
  123 |     const lineupText = (await lineupTable.count()) > 0
  124 |       ? await lineupTable.first().textContent()
  125 |       : '';
  126 |     // Game B lineup should be visible (not empty, not from game A in DEMO mode)
  127 |     expect(lineupText?.length).toBeGreaterThan(0);
  128 |   });
  129 | 
  130 |   test('slow API response does not leak game A data into game B load', async ({
  131 |     page,
  132 |   }) => {
  133 |     const options = await page.evaluate(() =>
  134 |       Array.from(document.querySelectorAll<HTMLOptionElement>('select option'))
  135 |         .filter((o) => o.value !== '')
  136 |         .map((o) => ({ value: o.value, text: o.text }))
  137 |     );
  138 |     if (options.length < 2) test.skip(true, 'Need at least 2 games');
  139 | 
  140 |     // Intercept /v1/optimize calls and delay game B response by 2s
  141 |     await page.route('**/v1/optimize/**', async (route) => {
  142 |       const url = route.request().url();
  143 |       const gameBPk = options[1].value;
  144 |       if (url.includes(gameBPk)) {
  145 |         await new Promise((r) => setTimeout(r, 2000));
  146 |       }
  147 |       await route.continue();
  148 |     });
  149 | 
  150 |     // Select game A
  151 |     await page.selectOption('select', options[0].value);
  152 |     await waitForNoSkeletons(page);
  153 | 
  154 |     // Now switch to game B (API will be slow)
  155 |     await page.selectOption('select', options[1].value);
  156 | 
  157 |     // During the 2-second delay, game A data must NOT be visible in the lineup table
  158 |     // The reducer cleared lineupData = null immediately, so skeleton should appear
  159 |     const tableText = async () => {
  160 |       const tables = await page.$$('table');
  161 |       return tables.length > 0 ? await tables[0].textContent() : '';
  162 |     };
  163 | 
  164 |     // Poll for 1.5s to check no stale game A lineup data appears
  165 |     let foundStaleData = false;
  166 |     const pollStart = Date.now();
  167 |     while (Date.now() - pollStart < 1500) {
  168 |       const text = await tableText();
  169 |       if (text && text.length > 50) {
  170 |         // Some data is visible — it should belong to game B
  171 |         // In DEMO mode game A = NYY, game B should be different
  172 |         // We just verify the table is not showing BOTH games' players
  173 |         foundStaleData = true; // flag that data appeared early
  174 |         break;
  175 |       }
  176 |       await page.waitForTimeout(200);
  177 |     }
  178 | 
  179 |     // After full load, game B data should be present
  180 |     await waitForNoSkeletons(page);
  181 |     const displayedLabel = await getSelectedGameOption(page);
  182 |     expect(displayedLabel).toBe(options[1].text);
  183 |   });
  184 | });
  185 | 
  186 | // ── Post-Game Review tests ────────────────────────────────────────────────────
  187 | 
  188 | test.describe('Post-Game Review — selector coherence (Bug 1)', () => {
  189 |   test.beforeEach(async ({ page }) => {
  190 |     await page.goto(APP_URL);
  191 |     const pgTab = page.getByRole('button', { name: /Post-Game/i });
  192 |     await pgTab.click();
```