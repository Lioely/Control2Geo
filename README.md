# Control2Geo

## Abstract
We propose Control2Geo, a diffusion-based model for monocular geometry estimation that recovers 3D geometry from open-domain images with high data efficiency. Control2Geo introduces explicit geometric guidance through a separate pretrained controller, which injects hierarchical cues into the main U-Net without disrupting the pretrained diffusion prior. By formulating estimation as a diffusion bridge from image latent to point-map latent, our framework enables structurally aligned and efficient deterministic inference with only a few sampling steps. To further reduce the domain gap between pretrained image VAEs and 3D point maps, we adopt a simple two-stage training strategy. Using only 61K training samples, which merely 0.67\% of the data used by MoGe, and 0.1\% of that used by DepthAnythingv2, Control2Geo achieves competitive, and in several cases state-of-the-art, performance on multiple zero-shot benchmarks, including NYUv2 and iBims-1, for monocular point map and depth estimation.

![这是图片](/figure/c2geo_figure.png){:height="50%" width="50%"}

![这是图片](/figure/inference.png){:height="50%" width="50%"}
