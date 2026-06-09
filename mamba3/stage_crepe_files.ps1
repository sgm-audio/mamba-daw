# Stage only CREPE/pitch .npy files for drag-and-drop to RunPod
# RunPod gets raw MUSDB18-HQ stems; this script packages just the .npy overlays.

$srcRoot  = "C:\Dev\mamba3\mamba3\Mamba-3-training\datasets\musdb18hq"
$destRoot = "C:\Dev\mamba3\mamba3\Mamba-3-training\crepe_overlay_only"

foreach ($split in @('train', 'test')) {
    $srcDir = Join-Path $srcRoot $split
    $tracks = Get-ChildItem $srcDir -Directory
    foreach ($track in $tracks) {
        $destTrack = Join-Path $destRoot $split $track.Name
        New-Item -ItemType Directory -Path $destTrack -Force | Out-Null
        # Copy only .crepe.npy and .pitched.npy files
        Get-ChildItem $track.FullName -Filter *.crepe.npy | Copy-Item -Destination $destTrack
        Get-ChildItem $track.FullName -Filter *.pitched.npy | Copy-Item -Destination $destTrack
    }
}

Write-Host "Done. Staged at: $destRoot"
Write-Host ""
Write-Host "Zip and transfer:"
Write-Host "  Compress-Archive -Path $destRoot\* -DestinationPath crepe_overlay_only.zip"
Write-Host ""
Write-Host "On RunPod, unzip into your musdb18hq root so train/ and test/ merge:"
Write-Host "  unzip crepe_overlay_only.zip -d /path/to/datasets/musdb18hq"
