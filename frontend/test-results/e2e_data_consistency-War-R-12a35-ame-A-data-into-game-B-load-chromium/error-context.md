# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: e2e_data_consistency.spec.ts >> War Room — game selector data consistency (Bug 1) >> slow API response does not leak game A data into game B load
- Location: e2e_data_consistency.spec.ts:130:7

# Error details

```
Error: page.selectOption: Target page, context or browser has been closed
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
    22 × waiting for element to be visible and enabled
       - did not find some options
     - retrying select option action
       - waiting 500ms

```

```
Error: browserContext.close: Target page, context or browser has been closed
```