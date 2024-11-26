# HeadRouter: A Training-free Image Editing Framework for MM-DiTs by Adaptively Routing Attention Heads

![](assets/teaser.png)

<a href='https://arxiv.org/abs/2411.15034'><img src='https://img.shields.io/badge/ArXiv-2403.16510-red'></a> 

## TL; DR
HeadRouter is a training-free text guided real image editing framework that based on MM-DiT (e.g. SD3 and Flux).

## Abstract
Diffusion Transformers (DiTs) have exhibited robust capabilities in image generation tasks. However, accurate text-guided image editing for multimodal DiTs (MM-DiTs) still poses a significant challenge. Unlike UNet-based structures that could utilize self/cross-attention maps for semantic editing, MM-DiTs inherently lack support for explicit and consistent incorporated text guidance, resulting in semantic misalignment between the edited results and texts. In this study, we disclose the sensitivity of different attention heads to different image semantics within MM-DiTs and introduce HeadRouter, a training-free image editing framework that edits the source image by adaptively routing the text guidance to different attention heads in MM-DiTs. Furthermore, we present a dual-token refinement module to refine text/image token representations for precise semantic guidance and accurate region expression. Experimental results on multiple benchmarks demonstrate HeadRouter's performance in terms of editing fidelity and image quality.

## Pipeline
![](assets/pipeline.png)

## Comparison with baselines
![](assets/main_compare.png)

## More of our results
![](assets/more_results.png)
