# @felipejesus/openloop

Ralph Loop for [OpenCode](https://opencode.ai): start a task once, then let the plugin keep nudging the session forward until it truthfully emits `<promise>DONE</promise>`.

## What it does

- adds `/ralph-loop` to start an auto-continuation loop
- adds `/cancel-ralph` to stop an active loop
- adds `/ralph-help` for a quick reference
- stores loop state in `.opencode/ralph-loop.local.md`
- watches for `session.idle`, checks the last assistant reply, and continues the session if the task is not done yet

## Install

Add the plugin name to your OpenCode config after publishing or installing it as a dependency:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["@felipejesus/openloop"]
}
```

On first load, the plugin copies its commands and skills into your OpenCode config directory so the slash commands are available globally.

For this repository specifically, a project-local loader lives at `.opencode/plugins/ralph-loop.ts`, so opening this repo in OpenCode will also load the plugin directly from source.

## Commands

```text
/ralph-loop Build a REST API with authentication
/cancel-ralph
/ralph-help
```

## How it works

1. `/ralph-loop` writes `.opencode/ralph-loop.local.md`
2. the plugin binds that loop to the current OpenCode session
3. when the session goes idle, the plugin checks the latest assistant text for `<promise>DONE</promise>`
4. if the promise is missing and the iteration cap is not reached, the plugin sends a continuation prompt back into the same session
5. the loop stops when the promise appears, the loop is cancelled, or `maxIterations` is reached

## Development

```bash
npm install
npm run release:check
```

## Release pipeline

GitHub Actions is configured with:

- `/.github/workflows/ci.yml` to run type-checking and package validation on pushes and pull requests
- `/.github/workflows/release.yml` to publish to npm and create a GitHub Release when a `v*` tag is pushed

Release migration plan:

1. Publish the first version of `@felipejesus/openloop` with the existing `NPM_TOKEN`
2. Open the `@felipejesus/openloop` package settings on npm
3. Add a GitHub Actions trusted publisher
4. Use `felipepiresdejesus` as the owner, `openloop` as the repository, and `release.yml` as the workflow filename
5. After trusted publishing is working, delete the old `NPM_TOKEN` GitHub secret

The release workflow is compatible with both stages: npm will use the token for the first publish, then switch to GitHub OIDC once trusted publishing is configured.

To cut a release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Repo layout

```text
src/index.ts              Main plugin implementation
commands/                 Slash commands installed into OpenCode
skills/                   Lazy-loaded skills installed into OpenCode
```
