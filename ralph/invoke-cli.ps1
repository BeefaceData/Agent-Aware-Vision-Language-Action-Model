# Argument-array bridge for Windows CLI wrappers. Never evaluate shell text.
param([Parameter(Mandatory=$true, Position=0)][string]$Executable,
      [Parameter(ValueFromRemainingArguments=$true)][string[]]$CliArguments)
$ErrorActionPreference = 'Stop'
$input | & $Executable @CliArguments
exit $LASTEXITCODE
