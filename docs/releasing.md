# Releasing rlx-models

The mechanics live in `scripts/publish.sh` (tier-ordered walk, rate-limit
sleeps, sparse-index polling, resume-on-error). This document is the part that
is *not* mechanical: the preconditions, and the traps that have actually broken
releases and that no cargo command reports.

## Preconditions

rlx-models pins upstream `rlx*` at the version it was built against, and
publishes a subset of its own crates. The two lines are **not** lockstep —
upstream has published versions that rlx-models never did.

1. **Every upstream pin must exist on crates.io.** This has been the hard
   blocker twice: `rlx-opscope` (0.2.16 cycle) and `rlx-torch-ckpt`. **Both
   are shipped as of 2026-10-11** (0.2.16 and 0.2.17 on the sparse index), so
   there is currently no upstream blocker — `check-publishable.py --release`
   reports only the clean-tree problems. Re-check rather than assume: a new
   upstream crate reintroduces this class instantly.
   A single unpublished upstream crate makes the *whole workspace*
   unresolvable — `cargo metadata` fails outright once the local path patch is
   removed, so a fresh clone or CI box cannot even load the workspace. A
   dev-dependency is not a lesser problem: cargo resolves dev-deps at lock time
   for the whole workspace.
2. **The tree must be committed.** `cargo publish` refuses a dirty tree, and
   `--allow-dirty` *excludes untracked files* — which is worse than failing,
   because a crate whose `mod`-declared sources are untracked uploads
   successfully and then fails to compile for everyone.

Both are checked by `python3 scripts/check-publishable.py --release`.

### Pin the upstream version deliberately

Upstream is **0.2.19**; this workspace is 0.2.19 and pins every upstream
crate at `^0.2.19` (`rlx-program` included, in the standalone `program/`
manifests). The local `../rlx` checkout is 0.2.19 too, so the patched build
and a registry build resolve to the same thing — there is no drift of the kind
that bit the 0.2.14 cycle. When upstream moves again, bump the pins and the
workspace version together, or pin `=0.2.19` on purpose. Leaving it implicit
is how the two lines drift apart.

## Gates

```sh
just lint-all           # fmt + clippy -D warnings + catalog + publish preflight
just release-preflight  # the above, plus the two preconditions (needs network)
```

- `cargo fmt --all -- --check` and `clippy --workspace --all-targets -- -D
  warnings` drift badly between releases (155 files and 26 warnings in the
  0.2.16 cycle). Most are machine-applicable: `cargo clippy --fix
  --allow-dirty -p <crate> --all-targets --no-deps`, then `cargo fmt --all`.
  **Run clippy's fixer before fmt** — `--fix` reintroduces formatting drift.
- `scripts/check-models-catalog.py` — MODELS.md claims to be generated from
  each crate's `Cargo.toml`; it is not, and it goes stale every release.
- `scripts/check-publishable.py` — the publish-blocking classes below.

## Traps that no cargo command catches

**`include_str!` / `include_bytes!` escaping its own package.** `cargo package`
ships only files under the crate root, so the published crate cannot compile.
Invisible locally, because the sibling crate is right there — and
`cargo package --no-verify` misses it too. Found in 0.2.16 in
`rlx-ten-vad-core`, which included a blob from `../../rlx-ten-vad/`. Fix by
moving the asset *down* to the crate lowest in the dep graph and re-exporting
it. Note the library building is not sufficient evidence: a stale path left in
a `#[cfg(test)]` block still compiles the lib and only fails under `cargo test`.

**An include target caught by a `.gitignore` net.** `.gitignore` has repo-wide
nets for regenerable fixtures (`crates/*/tests/fixtures/*.{bin,json,txt}`) and
weight dirs (`crates/*/weights/`). A compile-time input caught by one of these
does not exist on a fresh clone, so the target does not build at all — for a
reason invisible on the machine that generated the file. The tracked idiom is an
explicit negation under `# Required build assets (baked via include_str!)`.
Where the net is a **directory** rule, a bare file negation is inert: git never
descends into an excluded directory. Use re-include-dir → re-exclude-contents →
allow-one-file, and check a stray blob is still ignored afterwards.

**`publish.sh` TIERS drift.** A crate in no tier is silently never published.
A dependency in the same or a later tier than its dependent fails *partway
through* a real run. `publish.sh --list` validates membership but **never
ordering**. A crate listed in two tiers silently publishes from the later one.
Recompute from `cargo metadata` rather than hand-editing; keep each crate's
current tier and iterate `tier[n] = max(tier[n], tier[dep] + 1)` to a fixpoint —
a from-scratch depth assignment churns roughly twice as many crates for no
benefit. Being in a later tier than necessary is harmless.

**Path deps with no version.** `cargo publish` hard-rejects them; nothing else
does. Self dev-deps (a crate depending on itself to enable test features) are
*not* an instance — cargo strips those entirely.

**Crate size.** crates.io caps a `.crate` at 10 MiB, and float weights barely
compress.

## Verifying before you start

Tier 0 is the only tier that can be verified ahead of time — every later tier
depends on crates that are not on the index yet, so their `cargo package` fails
on dep resolution, which is expected and not a defect. Verify tier 0 with build
verification (not `--no-verify`, which is what misses the include trap):

```sh
for c in $(...tier 0 crates...); do cargo package --allow-dirty -p "$c" || echo "FAIL $c"; done
```

## Running it

```sh
./scripts/publish.sh --list       # tier coverage (membership only — see above)
./scripts/publish.sh --dry-run
./scripts/publish.sh --yes
```

It skips versions already on the index, polls until each upload is resolvable
before starting the next, and retries registry 429/5xx with the server's own
backoff — so it is safe to leave running unattended and safe to re-run after an
interruption.
