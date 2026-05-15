param(
    [switch]$SkipDataset,
    [switch]$SkipIndex,
    [switch]$SkipAccuracy,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv-win\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

Push-Location $Root
try {
    if ($DryRun) {
        Write-Host "Would run benchmark from $Root"
        Write-Host "Gemini-only final run; clearing local proxy variables and forcing TG_FORCE_LOCAL=1"
        if (-not $SkipDataset) {
            Write-Host "$Python scripts\download_cold_cases.py"
            Write-Host "$Python scripts\chunk_corpus.py"
            Write-Host "$Python scripts\export_tigergraph_csv.py"
        }
        if (-not $SkipIndex) {
            Write-Host "$Python scripts\basic_rag_gemini.py --build"
        }
        Write-Host "$Python scripts\llm_only_baseline.py"
        Write-Host "$Python scripts\basic_rag_gemini.py"
        Write-Host "TG_FORCE_LOCAL=1 $Python scripts\graphrag_tigergraph.py"
        if (-not $SkipAccuracy) {
            Write-Host "$Python scripts\evaluate_accuracy.py --bertscore-model distilbert-base-uncased"
        }
        Write-Host "$Python scripts\compare_pipelines.py --accuracy-report data\reports\accuracy_report.json"
        return
    }

    # Final benchmark path: Gemini generation/judging over the exported TigerGraph graph.
    # Some local shells carry disabled proxy env vars; clear them for outbound LLM API calls.
    Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY -ErrorAction SilentlyContinue
    $env:LLM_PROVIDER = "gemini"
    $env:LLM_FALLBACK_PROVIDER = ""
    $env:TG_FORCE_LOCAL = "1"
    $env:PYTHONIOENCODING = "utf-8"

    if (-not $SkipDataset) {
        & $Python scripts\download_cold_cases.py
        & $Python scripts\chunk_corpus.py
        & $Python scripts\export_tigergraph_csv.py
    }

    if (-not $SkipIndex) {
        & $Python scripts\basic_rag_gemini.py --build
    }

    & $Python scripts\llm_only_baseline.py
    & $Python scripts\basic_rag_gemini.py

    & $Python scripts\graphrag_tigergraph.py

    if (-not $SkipAccuracy) {
        & $Python scripts\evaluate_accuracy.py --bertscore-model distilbert-base-uncased
    }

    & $Python scripts\compare_pipelines.py --accuracy-report data\reports\accuracy_report.json
}
finally {
    Pop-Location
}
