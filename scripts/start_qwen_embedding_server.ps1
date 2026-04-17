param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8001,
    [string]$Model = "Qwen/Qwen3-Embedding-0.6B",
    [string]$Device = "",
    [int]$BatchSize = 32
)

$ErrorActionPreference = "Stop"

conda activate py312pt291cu128

# python -m pip install -r requirements-embedding-server.txt

$env:EMBED_SERVER_HOST = $Host
$env:EMBED_SERVER_PORT = [string]$Port
$env:EMBED_SERVER_MODEL = $Model
$env:EMBED_SERVER_BATCH_SIZE = [string]$BatchSize

if ($Device) {
    $env:EMBED_SERVER_DEVICE = $Device
}

python .\scripts\serve_qwen_embedding.py
