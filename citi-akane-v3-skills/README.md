# Citi Devin — Akane v3 Skills

This package contains the 10 complete Devin skill prompts.

## Configuration

Create the skills in Citi Devin by entering the `name`, `description`, and instructions from each `SKILL.md` file. For each one, set **Should Devin automatically use this skill? = No**. Invoke the master explicitly with `@skills:akane`.

| Skill | Role | Auto-use |
|---|---|---|
| `akane` | Autonomous engineering lead that creates Managed Devin child sessions, coordinates development, manages integration branches, automatically integrates verified child PRs, and creates a final GitHub PR. | No |
| `design` | Analyzes existing architecture, defines interfaces, identifies dependencies, and creates implementation plans. | No |
| `trace` | Investigates complex failures, reproduces bugs, traces execution paths, and identifies precise root causes. | No |
| `build` | Implements features and bug fixes on isolated branches, writes tests, creates GitHub PRs, and automatically merges authorized child PRs. | No |
| `prove` | Independently verifies application behavior, runs regression tests, and validates the exact integrated code. | No |
| `review` | Independently inspects code for correctness defects, broken contracts, regressions, and dangerous assumptions. | No |
| `guard` | Reviews authentication, authorization, sensitive data flows, trust boundaries, and security vulnerabilities. | No |
| `perf` | Investigates bottlenecks, profiles workloads, and verifies performance improvements with repeatable measurements. | No |
| `pr` | Creates commits, handles child-to-parent pull requests and authorized automatic merges, and prepares Akane's final PR to main. | No |
| `watch` | Checks GitHub CI, review feedback, branch conflicts, and PR readiness, coordinating fixes when checks fail. | No |

## Intended branch structure

- Akane creates `akane/<task-name>` from the repository's approved `main`.
- Each coding child creates its own `work/<task-name>-<component>` branch.
- Child PRs target Akane's branch and are merged automatically **only when Citi permissions, reviews, checks, and approvals allow**.
- Akane verifies the integrated commit and opens the final PR from `akane/<task-name>` to `main`.
- The final PR is never automatically merged to `main`.

## Example prompt

```
@skills:akane

Implement the requested change end to end.
Create the integration branch, delegate independent work to Managed Devin children,
have children open and merge authorized PRs into the integration branch,
run independent QA and review on the integrated commit,
and open the final PR to main. Do not merge the final PR.
```

**Note:** Skill instructions describe behavior; they cannot enable Managed Devin or grant GitHub permissions. Confirm capabilities and Citi policy in an authorized test repository.
