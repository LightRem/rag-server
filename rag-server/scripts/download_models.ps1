$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
$modelsDir = Join-Path $rootDir "models"
if (-not (Test-Path -Path $modelsDir)) {
    Write-Host "Creating models directory: $modelsDir"
    New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
}
$baseUrl = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"
$models = @(
    @{
        Name = "tokenizer.json"
        Url  = "$baseUrl/tokenizer.json"
    },
    @{
        Name = "text_model.onnx"
        Url  = "$baseUrl/onnx/text_model.onnx"
    },
    @{
        Name = "vision_model.onnx"
        Url  = "$baseUrl/onnx/vision_model.onnx"
    }
)
Write-Host "=== Checking and downloading ONNX CLIP models ==="
Write-Host "Target folder: $modelsDir"
foreach ($item in $models) {
    $targetPath = Join-Path $modelsDir $item.Name
    if (Test-Path -Path $targetPath) {
        Write-Host "✓ $($item.Name) already exists, skipping."
    } else {
        Write-Host "-> Downloading $($item.Name)..."
        & curl.exe --create-dirs -L --fail --retry 3 $item.Url -o $targetPath
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to download $($item.Name). Exit code: $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        Write-Host "✓ $($item.Name) successfully downloaded."
    }
}
Write-Host "=== All model files are verified and ready! ==="
