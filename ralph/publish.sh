#!/usr/bin/env bash
# Upload exact Git objects through gh; Git HTTPS is unreliable on this machine.
# UTC commit identities make GitHub's canonical timestamps hash identically.
publish_objects() {
  local commit parent file entry mode object uploaded tree author_date committer_date
  while IFS= read -r commit; do
    parent=$(git -C "$ROOT" rev-parse "$commit^")
    : > "$EVIDENCE/tree-entries.jsonl"
    while IFS= read -r -d '' file; do
      entry=$(git -C "$ROOT" ls-tree "$commit" -- "$file")
      if [[ -z $entry ]]; then
        jq -nc --arg path "$file" '{path:$path,mode:"100644",type:"blob",sha:null}' >> "$EVIDENCE/tree-entries.jsonl"
        continue
      fi
      mode=${entry%% *}; object=${entry#*blob }; object=${object%%$'\t'*}
      [[ $mode == 100644 || $mode == 100755 ]] || die 'Only regular files may be published.'
      git -C "$ROOT" cat-file blob "$object" | base64 | jq -Rs '{encoding:"base64",content:.}' > "$EVIDENCE/blob-request.json"
      uploaded=$(gh_call api --method POST "repos/$REPO/git/blobs" --input "$(native_path "$EVIDENCE/blob-request.json")" --jq .sha)
      [[ $uploaded == "$object" ]] || die 'GitHub blob differs from the tested content.'
      jq -nc --arg path "$file" --arg mode "$mode" --arg sha "$object" \
        '{path:$path,mode:$mode,type:"blob",sha:$sha}' >> "$EVIDENCE/tree-entries.jsonl"
    done < <(git -C "$ROOT" diff-tree --no-commit-id --no-renames --name-only -r -z "$parent" "$commit")
    jq -s --arg base "$(git -C "$ROOT" rev-parse "$parent^{tree}")" '{base_tree:$base,tree:.}' \
      "$EVIDENCE/tree-entries.jsonl" > "$EVIDENCE/tree-request.json"
    tree=$(gh_call api --method POST "repos/$REPO/git/trees" --input "$(native_path "$EVIDENCE/tree-request.json")" --jq .sha)
    [[ $tree == "$(git -C "$ROOT" rev-parse "$commit^{tree}")" ]] || die 'GitHub tree differs from the tested commit.'
    git -C "$ROOT" cat-file commit "$commit" | sed '1,/^$/d' > "$EVIDENCE/message.txt"
    author_date=$(git -C "$ROOT" show -s --format=%aI "$commit")
    committer_date=$(git -C "$ROOT" show -s --format=%cI "$commit")
    jq -n --arg tree "$tree" --arg parent "$parent" --rawfile message "$EVIDENCE/message.txt" \
      --arg an "$(git -C "$ROOT" show -s --format=%an "$commit")" --arg ae "$(git -C "$ROOT" show -s --format=%ae "$commit")" \
      --arg cn "$(git -C "$ROOT" show -s --format=%cn "$commit")" --arg ce "$(git -C "$ROOT" show -s --format=%ce "$commit")" \
      --arg ad "$author_date" --arg cd "$committer_date" \
      '{tree:$tree,parents:[$parent],message:$message,author:{name:$an,email:$ae,date:$ad},committer:{name:$cn,email:$ce,date:$cd}}' > "$EVIDENCE/commit-request.json"
    uploaded=$(gh_call api --method POST "repos/$REPO/git/commits" --input "$(native_path "$EVIDENCE/commit-request.json")" --jq .sha)
    [[ $uploaded == "$commit" ]] || die 'GitHub commit identity differs; no remote branch moved.'
  done < <(git -C "$ROOT" rev-list --reverse "$BASE..$HEAD")
}
publish_branch() {
  publish_objects
  gh_call api --method PATCH "repos/$REPO/git/refs/heads/$BRANCH" -f "sha=$HEAD" -F force=false >/dev/null
}
