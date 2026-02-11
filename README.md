# vlm-crowd-risk-analyzer
Real-time crowd detection using YOLOv8 with vision-language–based risk analysis via a local VLM.

Findings and Model Capability Analysis

This project evaluates the practical capabilities of a lightweight Vision–Language Model (Qwen3-VL) when combined with a real-time object detection pipeline (YOLOv8) for crowd monitoring and safety analysis.

Model Capabilities Observed

Contextual understanding beyond detection
While YOLOv8 provides accurate people detection and counting, the vision–language model adds semantic interpretation, such as describing crowd density, spatial distribution, and potential safety concerns (e.g., congestion or clustering).

Robust performance with low-resolution inputs
The VLM maintained coherent and relevant descriptions even after frame resizing and JPEG compression, indicating suitability for edge or resource-constrained environments.

Prompt sensitivity and controllability
The quality of analysis is strongly influenced by prompt design. Explicit inclusion of structured metadata (e.g., people count) significantly improved the relevance and consistency of the generated descriptions.

Offline-first feasibility
Running the VLM locally via Ollama demonstrates that meaningful vision-language inference is possible without cloud dependency, which is critical for privacy-sensitive and edge deployments.

Engineering Changes and Performance Improvements

Several modifications were introduced to improve system stability, responsiveness, and inference quality:

Decoupling detection and semantic analysis
YOLOv8 is used exclusively for real-time people detection, while the VLM is invoked only for higher-level reasoning. This separation reduces unnecessary load on the language model.

Conditional VLM invocation
Vision-language inference is triggered only when at least one person is detected, avoiding redundant computation on empty frames and improving real-time performance.

Frame encoding optimization
Frames are resized and JPEG-encoded before transmission to the VLM, reducing payload size and inference latency without significantly degrading semantic output quality.

Explicit numerical grounding in prompts
Injecting the detected people count into the VLM prompt improved factual grounding and reduced hallucinated descriptions.

Model selection for efficiency
A lightweight YOLOv8 variant (yolov8n) and a compact vision-language model (~2B parameters) were intentionally chosen to balance accuracy and computational cost.

Practical Implications

These findings indicate that lightweight vision–language models, when paired with classical computer vision pipelines, can deliver meaningful situational awareness in real time. The approach is well-suited for:

Edge AI deployments

Campus and event crowd monitoring

Privacy-preserving surveillance systems

Research on hybrid CV + VLM architectures
