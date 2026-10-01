def paint($code; $text):
  if $color == "1"
  then "\u001b[" + $code + "m" + $text + "\u001b[0m"
  else $text
  end;

def clean_text:
  tostring
  | gsub("\r\n"; "\n")
  | gsub("\r"; "\n")
  | gsub("’"; "'")
  | gsub("‘"; "'")
  | gsub("“"; "\"")
  | gsub("”"; "\"")
  | gsub("—"; "--")
  | gsub("–"; "-");

def line($text):
  $text + "\n";

def compact_command:
  tostring
  | gsub("[\r\n\t ]+"; " ")
  | if contains(" -Command ")
    then split(" -Command ") | .[1:] | join(" -Command ")
    else .
    end
  | if ((startswith("'") and endswith("'")) or (startswith("\"") and endswith("\"")))
    then .[1:-1]
    else .
    end
  | if length > 118 then .[0:115] + "..." else . end;

def relative_path:
  tostring
  | gsub("\\\\"; "/") as $path
  | ($repo_root | tostring | gsub("\\\\"; "/") | sub("/+$"; "")) as $root
  | if (($root | length) > 0 and ($path | startswith($root + "/")))
    then $path[($root | length) + 1:]
    else $path
    end;

def compact_number:
  (tonumber? // 0)
  | if . >= 1000000
    then ((. / 100000 | floor) / 10 | tostring) + "m"
    elif . >= 1000
    then ((. / 100 | floor) / 10 | tostring) + "k"
    else tostring
    end;

if .type == "turn.started" then
  line(paint("0;34"; "  | START Codex turn"))
elif (.type == "item.completed" and .item.type == "agent_message") then
  "\n" + ((.item.text // "") | clean_text) + "\n\n"
elif (.type == "item.started" and .item.type == "command_execution") then
  line(paint("0;33"; "  | RUN   " + ((.item.command // "command") | compact_command)))
elif (.type == "item.completed" and .item.type == "command_execution"
      and ((.item.status // "") == "failed" or (.item.exit_code // 0) != 0)) then
  line(paint("0;31"; "  | FAIL  "
    + ((.item.command // "command") | compact_command)
    + " (exit " + ((.item.exit_code // "?") | tostring)
    + "; details in diagnostics log)"))
elif (.type == "item.completed" and .item.type == "file_change") then
  ([.item.changes[]?.path | relative_path] | unique | join(", ")) as $files
  | line(paint("0;36"; "  | EDIT  "
    + (if $files == "" then "files changed" else $files end)))
elif (.type == "item.started" and .item.type == "mcp_tool_call") then
  ([.item.server, .item.tool] | map(select(. != null and . != "")) | join("/")) as $tool
  | line(paint("0;33"; "  | TOOL  "
    + (if $tool == "" then "MCP call" else $tool end)))
elif (.type == "item.started" and .item.type == "web_search") then
  line(paint("0;33"; "  | WEB   " + ((.item.query // "search") | compact_command)))
elif (.type == "error" or .type == "turn.failed") then
  line(paint("0;31"; "  | ERROR "
    + ((.message // .error.message // .error // "Codex reported an error") | tostring | compact_command)))
elif .type == "turn.completed" then
  line(paint("0;32"; "  | DONE  turn complete"
    + " | input " + ((.usage.input_tokens // 0) | compact_number)
    + " | cached " + ((.usage.cached_input_tokens // 0) | compact_number)
    + " | output " + ((.usage.output_tokens // 0) | compact_number)
    + " | reasoning " + ((.usage.reasoning_output_tokens // 0) | compact_number)))
else
  empty
end
